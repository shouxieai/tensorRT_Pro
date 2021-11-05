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
            return ((cursor_++) + 1) % infers_.size();
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

    shared_ptr<MultiGPUInfer> create_multi_gpu_infer(
        const string& engine_file, Type type, const vector<int> gpuids, 
        float confidence_threshold, float nms_threshold,
        NMSMethod nms_method, int max_objects
    ){
        shared_ptr<MultiGPUInfer> instance(new BalancedImpl());
        auto impl = std::dynamic_pointer_cast<MultiGPUInferImpl>(instance);
        if(!impl->startup(
            engine_file, type, gpuids, confidence_threshold, nms_threshold, nms_method, max_objects
        )){
            instance.reset();
        }
        return instance;
    }

};