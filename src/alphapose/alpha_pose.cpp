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
            i2d[0] = scale;  i2d[1] = 0;      i2d[2] = -(box_.x - box_.width  * 1 * rate + pad_width * 0.5)  * scale + net_size.width  * 0.5;  
            i2d[3] = 0;      i2d[4] = scale;  i2d[5] = -(box_.y - box_.height * 1 * rate + pad_height * 0.5) * scale + net_size.height * 0.5;

            // 有了i2d矩阵，我们求其逆矩阵，即可得到d2i（用以解码时还原到原始图像分辨率上）
            cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
            cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
            cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
        }

        cv::Mat i2d_mat(){
            return cv::Mat(2, 3, CV_32F, i2d);
        }
    };

    struct PoseInput{
        Mat image;
        Rect box;
    };

    // 给定x、y坐标，经过仿射变换矩阵，进行映射
    static tuple<float, float> affine_project(float x, float y, float* pmatrix){

        float newx = x * pmatrix[0] + y * pmatrix[1] + pmatrix[2];
        float newy = x * pmatrix[3] + y * pmatrix[4] + pmatrix[5];
        return make_tuple(newx, newy);
    }

    using ControllerImpl = InferController
    <
        PoseInput,                 // input
        vector<Point3f>,           // output
        tuple<string, int>,        // start param
        AffineMatrix               // additional
    >;
    class InferImpl : public Infer, public ControllerImpl{
    public:
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
            float threshold    = 0.25;
            int stride         = input->width() / output->width();
            input_width_       = input->width();
            input_height_      = input->height();
            gpu_               = gpuid;
            tensor_allocator_  = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
            stream_            = engine->get_stream();
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
                    float* image_based_output   = output->cpu<float>(ibatch);
                    auto& image_based_keypoints = job.output;
                    auto& affine_matrix         = job.additional;
                    int begin_channel           = 17;
                    int area                    = output->width() * output->height();
                    image_based_keypoints.resize(output->channel() - begin_channel);

                    for(int i = begin_channel; i < output->channel(); ++i){
                        float* output_channel = output->cpu<float>(0, i);
                        int location = std::max_element(output_channel, output_channel + area) - output_channel;
                        float confidence = output_channel[location];
                        float x = (location % output->width()) * stride;
                        float y = (location / output->width()) * stride;
                        auto& output_point = image_based_keypoints[i-begin_channel];

                        output_point.z = confidence;
                        tie(output_point.x, output_point.y) = affine_project(x, y, job.additional.d2i);
                    }
                    job.pro->set_value(job.output);
                }
                fetch_jobs.clear();
            }
            INFOV("Engine destroy.");
        }

        virtual shared_future<vector<Point3f>> commit(const Mat& image, const Rect& box) override{
            return ControllerImpl::commit(PoseInput{.image=image, .box=box});
        }

        virtual bool preprocess(Job& job, const PoseInput& input) override{

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

            Size input_size(input_width_, input_height_);
            job.additional.compute(input.image.size(), input.box, input_size);
            
            tensor->set_stream(stream_);
            tensor->resize(1, 3, input_height_, input_width_);
            float mean[]           = {0.406, 0.457, 0.480};
            float std[]            = {1, 1, 1};

            size_t size_image      = input.image.cols * input.image.rows * 3;
            size_t size_matrix     = CUDAKernel::upbound(sizeof(job.additional.d2i), 32);
            auto workspace         = tensor->get_workspace();
            uint8_t* gpu_workspace = (uint8_t*)workspace->gpu(size_image + size_matrix);
            float*   affine_matrix_device = (float*)gpu_workspace;
            uint8_t* image_device         = gpu_workspace + size_matrix;
            checkCudaRuntime(cudaMemcpyAsync(image_device,         input.image.data,   size_image,                 cudaMemcpyHostToDevice, stream_));
            checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, job.additional.d2i, sizeof(job.additional.d2i), cudaMemcpyHostToDevice, stream_));

            auto normalize         = CUDAKernel::Norm::mean_std(mean, std) + CUDAKernel::NormType::ToRGB;
            CUDAKernel::warp_affine_bilinear_and_normalize(
                image_device,         input.image.cols * 3, input.image.cols, input.image.rows, 
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