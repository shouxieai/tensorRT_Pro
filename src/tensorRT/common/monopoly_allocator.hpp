

#ifndef MONOPOLY_ALLOCATOR_HPP
#define MONOPOLY_ALLOCATOR_HPP

#include <condition_variable>
#include <vector>
#include <mutex>
#include <memory>

template<class _ItemType>
class MonopolyAllocator{
public:
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