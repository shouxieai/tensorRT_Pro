#include "multi_gpu.hpp"
#include <atomic>
#include <ostream>
#include <common/ilogger.hpp>

namespace Yolo{

    class MultiGPUInferImpl{
    public:
        virtual bool startup(
            const string& engine_file, Type type, const vector<int> gpuids, 
            float confidence_threshold, float nms_threshold,
            NMSMethod nms_method, int max_objects
        ){
            if(gpuids.empty()){
                INFOE("gpuids is empty");
                return false;
            }

            if(!iLogger::exists(engine_file)){
                INFOE("Engine file %s not exists", engine_file.c_str());
                return false;
            }

            infers_.resize(gpuids.size());

            #pragma omp parallel for num_threads(infers_.size())
            for(int i = 0; i < gpuids.size(); ++i){
                auto& gpuid = gpuids[i];
                infers_[i] = Yolo::create_infer(
                    engine_file, type, gpuid, confidence_threshold,
                    nms_threshold, nms_method, max_objects
                );
            }

            for(int i = 0; i < gpuids.size(); ++i){
                if(infers_[i] == nullptr){
                    INFOE("Infer create failed, gpuid = %d", gpuids[i]);
                    return false;
                }
            }
            return true;
        }

    protected:
        vector<shared_ptr<Infer>> infers_;
    };

    class BalancedImpl : public MultiGPUInfer, public MultiGPUInferImpl{
    public:
        int get_gpu_index(){
            int gpu_index = (cursor_ + 1) % infers_.size();
            cursor_++;
            return gpu_index;
        }

        virtual shared_future<BoxArray> commit(const cv::Mat& image) override{
            return infers_[get_gpu_index()]->commit(image);
        }

        virtual vector<shared_future<BoxArray>> commits(const vector<cv::Mat>& images) override{
            return infers_[get_gpu_index()]->commits(images);
        }

    private:
        atomic<unsigned int> cursor_{0};
    };

    class ThreadIDHashImpl : public MultiGPUInfer, public MultiGPUInferImpl{
    public:
        unsigned int get_hash_index(const void* key, size_t length, size_t table_size){

            const unsigned char* p = (const unsigned char*)key;
            unsigned int hash_value = 0;
            for(int i = 0; i < length; ++i)
                hash_value = (hash_value << 5) + *p++;
            
            return hash_value % table_size;
        }

        int get_gpu_index(){
            std::ostringstream oss;
            oss << std::this_thread::get_id();
            std::string stid = oss.str();
            return get_hash_index(stid.c_str(), stid.size(), infers_.size());
        }

        virtual shared_future<BoxArray> commit(const cv::Mat& image) override{
            return infers_[get_gpu_index()]->commit(image);
        }

        virtual vector<shared_future<BoxArray>> commits(const vector<cv::Mat>& images) override{
            return infers_[get_gpu_index()]->commits(images);
        }
    };

    // class ThreadPoolImpl : public MultiGPUInfer, public MultiGPUInferImpl{
    // public:
    //     virtual shared_future<BoxArray> commit(const cv::Mat& image) override{

    //     }

    //     virtual vector<shared_future<BoxArray>> commits(const vector<cv::Mat>& images) override{

    //     }
    // };

    shared_ptr<MultiGPUInfer> create_multi_gpu_infer(
        const string& engine_file, Type type, const vector<int> gpuids, 
        DispatchMethod dispatch_method,
        float confidence_threshold, float nms_threshold,
        NMSMethod nms_method, int max_objects
    ){
        shared_ptr<MultiGPUInfer> instance;
        switch(dispatch_method){
        case DispatchMethod::Balanced: instance.reset(new BalancedImpl());         break;
        case DispatchMethod::ThreadIDHash: instance.reset(new ThreadIDHashImpl()); break;
        //case DispatchMethod::ThreadPool: instance.reset(new ThreadPoolImpl());     break;
        case DispatchMethod::ThreadPool: 
            INFOE("Not implemented method");
            break;
        }

        if(instance == nullptr){
            INFOE("Unknow dispatch method");
            return instance;
        }

        auto impl = std::dynamic_pointer_cast<MultiGPUInferImpl>(instance);
        if(!impl->startup(
            engine_file, type, gpuids, confidence_threshold, nms_threshold, nms_method, max_objects
        )){
            instance.reset();
        }
        return instance;
    }

};