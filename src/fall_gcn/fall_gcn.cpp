#include "fall_gcn.hpp"
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include <common/infer_controller.hpp>
#include <common/monopoly_allocator.hpp>
#include <common/preprocess_kernel.cuh>

namespace FallGCN{

    struct FallGCNInput{
        vector<Point3f> keys;
        Rect box;
    };

    const char* state_name(FallState state){
        switch(state){
            case FallState::Fall:      return "Fall";
            case FallState::Stand:     return "Stand";
            case FallState::UnCertain: return "UnCertain";
            default: return "Unknow";
        }
    }

    static void softmax(float* p, int size){

        float total = 0;
        for(int i = 0; i < size; ++i){
            p[i] = exp(p[i]);
            total += p[i];
        }

        for(int i = 0; i < size; ++i)
            p[i] /= total;
    }

    using ControllerImpl = InferController
    <
        FallGCNInput,              // input
        tuple<FallState, float>,   // output
        tuple<string, int>         // start param
    >;
    class InferImpl : public Infer, public ControllerImpl{
    public:
        bool startup(const string& file, int gpuid){
            gpuid_ = gpuid;
            return ControllerImpl::startup(make_tuple(file, gpuid));
        }
    
        virtual void worker(promise<bool>& result) override{
            
            string file = get<0>(start_param_);
            int gpuid   = get<1>(start_param_);

            TRT::set_device(gpuid);
            auto engine = TRT::load_infer(file);
            if(engine == nullptr){
                INFOE("Engine %s load failed", file.c_str());
                result.set_value(false);
                return;
            }

            engine->print();

            int max_batch_size = engine->get_max_batch_size();
            auto input         = engine->input();
            auto output        = engine->output();

            tensor_allocator_ = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
            stream_           = engine->get_stream();
            result.set_value(true);
            input->resize_single_dim(0, max_batch_size);

            vector<Job> fetch_jobs;
            while(get_jobs_and_wait(fetch_jobs, max_batch_size)){

                // 一次推理越多越好
                // 把图像批次丢引擎里边去
                int infer_batch_size = fetch_jobs.size();
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    auto& job = fetch_jobs[ibatch];
                    input->copy_from_gpu(input->offset(ibatch), job.mono_tensor->data()->gpu(), input->count(1));
                    job.mono_tensor->release();
                }
                // 模型推理
                engine->forward(false);

                // 收取结果，output->cpu里面存在一个同步操作
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    
                    auto& job                   = fetch_jobs[ibatch];
                    float* item_based_output   = output->cpu<float>(ibatch);
                    auto& output_state          = job.output;
                    auto& affine_matrix         = job.additional;
                    softmax(item_based_output, output->channel());

                    int label = std::max_element(item_based_output, item_based_output + output->channel()) - item_based_output;
                    output_state = make_tuple((FallState)label, item_based_output[label]);
                    job.pro->set_value(output_state);
                }
                fetch_jobs.clear();
            }
            INFOV("Engine destroy.");
        }

        virtual shared_future<tuple<FallState, float>> commit(const vector<Point3f>& keys, const Rect& box) override{
            FallGCNInput input;
            input.keys  = keys;
            input.box   = box;
            return ControllerImpl::commit(input);
        }

        virtual bool preprocess(Job& job, const FallGCNInput& input) override{

            job.mono_tensor = tensor_allocator_->query();
            if(job.mono_tensor == nullptr){
                INFOE("Tensor allocator query failed.");
                return false;
            }

            if(input.keys.size() != 16){
                INFOE("input.keys.size()[%d] != 16", input.keys.size());
                return false;
            }

            auto& tensor = job.mono_tensor->data();
            if(tensor == nullptr){
                // not init
                tensor = make_shared<TRT::Tensor>();
                tensor->set_workspace(make_shared<TRT::MixMemory>());
            }

            int num_points = input.keys.size();
            tensor->set_stream(stream_);
            tensor->resize(1, num_points, 3);

            tensor->to_cpu(false);
            float* inptr = tensor->cpu<float>();
            int box_max_line = max(input.box.width, input.box.height);
            for(int i = 0; i < num_points; ++i, inptr += 3){
                auto& point = input.keys[i];
                inptr[0] = (point.x - input.box.x) / box_max_line - 0.5f;
                inptr[1] = (point.y - input.box.y) / box_max_line - 0.5f;
                inptr[2] = point.z;
            }
            tensor->to_gpu();
            return true;
        }

    private:
        int gpuid_ = 0;
        TRT::CUStream stream_ = nullptr;
    };

    shared_ptr<Infer> create_infer(const string& engine_file, int gpuid){
        shared_ptr<InferImpl> instance(new InferImpl());
        if(!instance->startup(engine_file, gpuid)){
            instance.reset();
        }
        return instance;
    }
};