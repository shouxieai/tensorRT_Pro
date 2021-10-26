#include "arcface.hpp"
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include <common/infer_controller.hpp>
#include <common/preprocess_kernel.cuh>
#include <common/monopoly_allocator.hpp>
#include <common/cuda_tools.hpp>

namespace Arcface{
    using namespace cv;
    using namespace std;

    struct AffineMatrix{
        float i2d[6];       // image to dst(network), 2x3 matrix
        float d2i[6];

        void compute(const landmarks& lands){

            // 112 x 112分辨率时的标准人脸关键点（训练用的是这个）
            // 96  x 112分辨率时的标准人脸关键点在下面基础上去掉x的偏移
            // 来源于论文和公开代码中训练用到的
            // https://github.com/wy1iu/sphereface/blob/f5cd440a2233facf46b6529bd13231bb82f23177/preprocess/code/face_align_demo.m
            float Sdata[] = {
                30.2946 + 8, 51.6963,
                65.5318 + 8, 51.5014,
                48.0252 + 8, 71.7366,
                33.5493 + 8, 92.3655,
                62.7299 + 8, 92.2041
            };

            // 以下代码参考自：http://www.zifuture.com/archives/face-alignment
            float Qdata[] = {
                lands.points[0],  lands.points[1], 1, 0,
                lands.points[1], -lands.points[0], 0, 1,
                lands.points[2],  lands.points[3], 1, 0,
                lands.points[3], -lands.points[2], 0, 1,
                lands.points[4],  lands.points[5], 1, 0,
                lands.points[5], -lands.points[4], 0, 1,
                lands.points[6],  lands.points[7], 1, 0,
                lands.points[7], -lands.points[6], 0, 1,
                lands.points[8],  lands.points[9], 1, 0,
                lands.points[9], -lands.points[8], 0, 1,
            };
            
            float Udata[4];
            Mat_<float> Q(10, 4, Qdata);
            Mat_<float> U(4, 1,  Udata);
            Mat_<float> S(10, 1, Sdata);

            U = (Q.t() * Q).inv() * Q.t() * S;
            i2d[0] = Udata[0];   i2d[1] = Udata[1];     i2d[2] = Udata[2];
            i2d[3] = -Udata[1];  i2d[4] = Udata[0];     i2d[5] = Udata[3];

            cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
            cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
            cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
        }
    };

    using ControllerImpl = InferController
    <
        commit_input,           // input
        feature,                // output
        tuple<string, int>,     // start param
        AffineMatrix            // additional
    >;
    class InferImpl : public Infer, public ControllerImpl{
    public:
        /** 要求在InferImpl里面执行stop，而不是在基类执行stop **/
        virtual ~InferImpl(){
            TRT::set_device(gpu_);
            stop();
        }
        
        virtual bool startup(const string& file, int gpuid){

            float mean[] = {0.5f, 0.5f, 0.5f};
            float std[]  = {0.5f, 0.5f, 0.5f};
            normalize_   = CUDAKernel::Norm::mean_std(mean, std, 1.0f / 255.0f, CUDAKernel::ChannelType::Invert);
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

            input_width_       = input->size(3);
            input_height_      = input->size(2);
            feature_length_    = output->size(1);
            tensor_allocator_  = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
            stream_            = engine->get_stream();
            gpu_               = gpuid;
            result.set_value(true);

            input->resize_single_dim(0, max_batch_size).to_gpu();
            output->resize_single_dim(0, max_batch_size).to_gpu();

            vector<Job> fetch_jobs;
            while(get_jobs_and_wait(fetch_jobs, max_batch_size)){

                int infer_batch_size = fetch_jobs.size();
                input->resize_single_dim(0, infer_batch_size);

                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    auto& job  = fetch_jobs[ibatch];
                    auto& mono = job.mono_tensor->data();
                    input->copy_from_gpu(input->offset(ibatch), mono->gpu(), mono->count());
                    job.mono_tensor->release();
                }

                engine->forward(false);
                CUDAKernel::norm_feature(output->gpu<float>(), output->size(0), output->size(1), stream_);

                output->to_cpu();
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    auto& job                 = fetch_jobs[ibatch];
                    float* image_based_output = output->cpu<float>(ibatch);

                    memcpy(job.output.ptr<float>(0), image_based_output, sizeof(float) * feature_length_);
                    job.pro->set_value(job.output);
                }
                fetch_jobs.clear();
            }
            stream_ = nullptr;
            tensor_allocator_.reset();
            INFO("Engine destroy.");
        }

        virtual bool preprocess(Job& job, const commit_input& input) override{

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
            Size input_size(input_width_, input_height_);
            job.additional.compute(get<1>(input));
            job.output = Mat_<float>(1, feature_length_);
            
            tensor->set_stream(stream_);
            tensor->resize(1, 3, input_height_, input_width_);

            size_t size_image      = image.cols * image.rows * 3;
            size_t size_matrix     = iLogger::upbound(sizeof(job.additional.d2i), 32);
            auto workspace         = tensor->get_workspace();
            uint8_t* gpu_workspace        = (uint8_t*)workspace->gpu(size_matrix + size_image);
            float*   affine_matrix_device = (float*)gpu_workspace;
            uint8_t* image_device         = size_matrix + gpu_workspace;

            uint8_t* cpu_workspace        = (uint8_t*)workspace->cpu(size_matrix + size_image);
            float* affine_matrix_host     = (float*)cpu_workspace;
            uint8_t* image_host           = size_matrix + cpu_workspace;

            //checkCudaRuntime(cudaMemcpyAsync(image_host,   image.data, size_image, cudaMemcpyHostToHost,   stream_));
            // speed up
            memcpy(image_host, image.data, size_image);
            memcpy(affine_matrix_host, job.additional.d2i,   sizeof(job.additional.d2i));
            checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_));
            checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(job.additional.d2i), cudaMemcpyHostToDevice, stream_));

            CUDAKernel::warp_affine_bilinear_and_normalize_plane(
                image_device,         image.cols * 3,       image.cols,       image.rows, 
                tensor->gpu<float>(), input_width_,         input_height_, 
                affine_matrix_device, 0, 
                normalize_, stream_
            );
            return true;
        }

        virtual vector<shared_future<feature>> commits(const vector<commit_input>& inputs) override{
            return ControllerImpl::commits(inputs);
        }

        virtual std::shared_future<feature> commit(const commit_input& input) override{
            return ControllerImpl::commit(input);
        }

    private:
        int input_width_            = 0;
        int input_height_           = 0;
        int feature_length_         = 0;
        int gpu_                    = 0;
        TRT::CUStream stream_       = nullptr;
        CUDAKernel::Norm normalize_;
    };

    Mat face_alignment(const cv::Mat& image, const landmarks& landmark){
        Size input_size(112, 112);
        AffineMatrix am;
        am.compute(landmark);

        Mat output;
        warpAffine(image, output, Mat_<float>(2, 3, am.i2d), input_size, cv::INTER_LINEAR);
        return output;
    }

    shared_ptr<Infer> create_infer(const string& engine_file, int gpuid){
        shared_ptr<InferImpl> instance(new InferImpl());
        if(!instance->startup(engine_file, gpuid)){
            instance.reset();
        }
        return instance;
    }
};