/**
 * 独占分配器
 * 用以解决以下问题：
 * 1. 实现tensor复用的问题
 * 2. 对于tensor使用的两个阶段实现并行，时间重叠
 *    阶段一：预处理准备
 *    阶段二：模型推理
 * 
 * 设计思路：
 * 以海底捞吃火锅为类比，座位分为两种：堂内吃饭的座位、厅外等候的座位
 * 
 * 1. 初始状态，堂内有10个座位，厅外有10个座位，全部空
 * 2. 来了30个人吃火锅
 * 3. 流程是，先安排10个人坐在厅外修整，20人个人排队等候
 * 4. 由于堂内没人，所以调度坐在厅外的10个人进入堂内，开始吃火锅。厅外的10个座位为空
 * 5. 由于厅外没人，所以可以让排队的20人中，取10个人在厅外修整
 * 6. 此时状态为，堂内10人，厅外10人，等候10人
 * 7. 经过60分钟后，堂内10人吃完，紧接着执行步骤4
 * 
 * 在实际工作中，通常图像输入过程有预处理、推理
 * 我们的目的是让预处理和推理时间进行重叠。因此设计了一个缓冲区，类似厅外等候区的那种形式
 * 当我们输入图像时，具有2倍batch的空间进行预处理用于缓存
 * 而引擎推理时，每次拿1个batch的数据进行推理
 * 当引擎推理速度慢而预处理速度快时，输入图像势必需要进行等候。否则缓存队列会越来越大
 * 而这里提到的几个点就是设计的主要目标
 **/

#ifndef MONOPOLY_ALLOCATOR_HPP
#define MONOPOLY_ALLOCATOR_HPP

#include <condition_variable>
#include <vector>
#include <mutex>
#include <memory>

template<class _ItemType>
class MonopolyAllocator{
public:
    /* Data是数据容器类
       允许query获取的item执行item->release释放自身所有权，该对象可以被复用
       通过item->data()获取储存的对象的指针
    */
    class MonopolyData{
    public:
        std::shared_ptr<_ItemType>& data(){ return data_; }
        void release(){manager_->release_one(this);}

    private:
        MonopolyData(MonopolyAllocator* pmanager){manager_ = pmanager;}

    private:
        friend class MonopolyAllocator;
        MonopolyAllocator* manager_ = nullptr;
        std::shared_ptr<_ItemType> data_;
        bool available_ = true;
    };
    typedef std::shared_ptr<MonopolyData> MonopolyDataPointer;

    MonopolyAllocator(int size){
        capacity_ = size;
        num_available_ = size;
        datas_.resize(size);

        for(int i = 0; i < size; ++i)
            datas_[i] = std::shared_ptr<MonopolyData>(new MonopolyData(this));
    }

    virtual ~MonopolyAllocator(){
        run_ = false;
        cv_.notify_all();
        
        std::unique_lock<std::mutex> l(lock_);
        cv_exit_.wait(l, [&](){
            return num_wait_thread_ == 0;
        });
    }

    /* 获取一个可用的对象
        timeout：超时时间，如果没有可用的对象，将会进入阻塞等待，如果等待超时则返回空指针
        请求得到一个对象后，该对象被占用，除非他执行了release释放该对象所有权
    */
    MonopolyDataPointer query(int timeout = 10000){

        std::unique_lock<std::mutex> l(lock_);
        if(!run_) return nullptr;
        
        if(num_available_ == 0){
            num_wait_thread_++;

            auto state = cv_.wait_for(l, std::chrono::milliseconds(timeout), [&](){
                return num_available_ > 0 || !run_;
            });

            num_wait_thread_--;
            cv_exit_.notify_one();

            // timeout, no available, exit program
            if(!state || num_available_ == 0 || !run_)
                return nullptr;
        }

        auto item = std::find_if(datas_.begin(), datas_.end(), [](MonopolyDataPointer& item){return item->available_;});
        if(item == datas_.end())
            return nullptr;
        
        (*item)->available_ = false;
        num_available_--;
        return *item;
    }

    int num_available(){
        return num_available_;
    }

    int capacity(){
        return capacity_;
    }

private:
    void release_one(MonopolyData* prq){
        std::unique_lock<std::mutex> l(lock_);
        if(!prq->available_){
            prq->available_ = true;
            num_available_++;
            cv_.notify_one();
        }
    }

private:
    std::mutex lock_;
    std::condition_variable cv_;
    std::condition_variable cv_exit_;
    std::vector<MonopolyDataPointer> datas_;
    int capacity_ = 0;
    volatile int num_available_ = 0;
    volatile int num_wait_thread_ = 0;
    volatile bool run_ = true;
};

#endif // MONOPOLY_ALLOCATOR_HPP