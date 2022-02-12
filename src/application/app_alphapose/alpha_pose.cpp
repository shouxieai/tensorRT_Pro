#include "alpha_pose.hpp"
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include <common/infer_controller.hpp>
#include <common/monopoly_allocator.hpp>
#include <common/preprocess_kernel.cuh>

namespace AlphaPose{

    struct AffineMatrix{
        float i2d[6];       // image to dst(network), 2x3 matrix
        float d2i[6];       // dst to image, 2x3 matrix

        void compute(const cv::Size& image_size, const cv::Rect& box, const cv::Size& net_size){
            Rect box_ = box;
            if(box_.width == 0 || box_.height == 0){
                box_.width  = image_size.width;
                box_.height = image_size.height;
                box_.x = 0;
                box_.y = 0;
            }

            float rate = box_.width > 100 ? 0.1f : 0.15f;
            float pad_width  = box_.width  * (1 + 2 * rate);
            float pad_height = box_.height * (1 + 1 * rate);
            float scale = min(net_size.width  / pad_width,  net_size.height / pad_height);
            i2d[0] = scale;  i2d[1] = 0;      i2d[2] = -(box_.x - box_.width  * 1 * rate + pad_width * 0.5)  * scale + net_size.width  * 0.5 + scale * 0.5 - 0.5;  
            i2d[3] = 0;      i2d[4] = scale;  i2d[5] = -(box_.y - box_.height * 1 * rate + pad_height * 0.5) * scale + net_size.height * 0.5 + scale * 0.5 - 0.5;

            cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
            cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
            cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
        }

        cv::Mat i2d_mat(){
            return cv::Mat(2, 3, CV_32F, i2d);
        }
    };

    static tuple<float, float> affine_project(float x, float y, float* pmatrix){

        float newx = x * pmatrix[0] + y * pmatrix[1] + pmatrix[2];
        float newy = x * pmatrix[3] + y * pmatrix[4] + pmatrix[5];
        return make_tuple(newx, newy);
    }

    using ControllerImpl = InferController
    <
        Input,                     // input
        vector<Point3f>,           // output
        tuple<string, int>,        // start param
        AffineMatrix               // additional
    >;
    class InferImpl : public Infer, public ControllerImpl{
    public:
        /** 要求在InferImpl里面执行stop，而不是在基类执行stop **/
        virtual ~InferImpl(){
            TRT::set_device(gpu_);
            stop();
        }
        
        bool startup(const string& file, int gpuid){
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
            input_width_       = input->width();
            input_height_      = input->height();
            gpu_               = gpuid;
            tensor_allocator_  = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
            stream_            = engine->get_stream();
            result.set_value(true);
            input->resize_single_dim(0, max_batch_size);

            int n = 0;
            vector<Job> fetch_jobs;
            while(get_jobs_and_wait(fetch_jobs, max_batch_size)){

                int infer_batch_size = fetch_jobs.size();
                input->resize_single_dim(0, infer_batch_size);

                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    auto& job = fetch_jobs[ibatch];
                    input->copy_from_gpu(input->offset(ibatch), job.mono_tensor->data()->gpu(), input->count(1));
                    job.mono_tensor->release();
                }
                
                engine->forward(false);
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    
                    auto& job                   = fetch_jobs[ibatch];
                    float* image_based_output   = output->cpu<float>(ibatch);
                    auto& image_based_keypoints = job.output;
                    auto& affine_matrix         = job.additional;

                    // output -> batch x 136 x 3
                    image_based_keypoints.resize(output->size(1));

                    for(int i = 0; i < output->size(1); ++i){
                        float* keyp = output->cpu<float>(ibatch, i);
                        float x = keyp[0];
                        float y = keyp[1];
                        float confidence = keyp[2];
                        auto& output_point = image_based_keypoints[i];
                        output_point.z = confidence;
                        tie(output_point.x, output_point.y) = affine_project(x, y, job.additional.d2i);
                    }
                    job.pro->set_value(job.output);
                }
                fetch_jobs.clear();
            }
            stream_ = nullptr;
            tensor_allocator_.reset();
            INFO("Engine destroy.");
        }

        virtual shared_future<vector<Point3f>> commit(const Input& input) override{
            return ControllerImpl::commit(input);
        }

        virtual vector<shared_future<vector<Point3f>>> commits(const vector<Input>& inputs) override{
            return ControllerImpl::commits(inputs);
        }

        virtual bool preprocess(Job& job, const Input& input) override{

            if(tensor_allocator_ == nullptr){
                INFOE("tensor_allocator_ is nullptr");
                return false;
            }

            job.mono_tensor = tensor_allocator_->query();
            if(job.mono_tensor == nullptr){
                INFOE("Tensor allocator query failed.");
                return false;
            }

            CUDATools::AutoDevice auto_device(gpu_);
            auto& tensor = job.mono_tensor->data();
            if(tensor == nullptr){
                // not init
                tensor = make_shared<TRT::Tensor>();
                tensor->set_workspace(make_shared<TRT::MixMemory>());
            }

            auto& image = get<0>(input);
            auto& box   = get<1>(input);
            Size input_size(input_width_, input_height_);
            job.additional.compute(image.size(), box, input_size);
            
            tensor->set_stream(stream_);
            tensor->resize(1, 3, input_height_, input_width_);
            float mean[]           = {0.406, 0.457, 0.480};
            float std[]            = {1, 1, 1};

            size_t size_image      = image.cols * image.rows * 3;
            size_t size_matrix     = iLogger::upbound(sizeof(job.additional.d2i), 32);
            auto workspace         = tensor->get_workspace();
            uint8_t* gpu_workspace = (uint8_t*)workspace->gpu(size_image + size_matrix);
            float*   affine_matrix_device = (float*)gpu_workspace;
            uint8_t* image_device         = gpu_workspace + size_matrix;
            checkCudaRuntime(cudaMemcpyAsync(image_device,         image.data,         size_image,                 cudaMemcpyHostToDevice, stream_));
            checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, job.additional.d2i, sizeof(job.additional.d2i), cudaMemcpyHostToDevice, stream_));

            auto normalize         = CUDAKernel::Norm::mean_std(mean, std, 1/255.0f, CUDAKernel::ChannelType::Invert);
            CUDAKernel::warp_affine_bilinear_and_normalize_plane(
                image_device,         image.cols * 3, image.cols, image.rows, 
                tensor->gpu<float>(), input_width_,     input_height_, 
                affine_matrix_device, 127, 
                normalize, stream_
            );
            return true;
        }

    private:
        int input_width_ = 0;
        int input_height_ = 0;
        int gpu_ = 0;
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