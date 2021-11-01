#include "scrfd.hpp"
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

namespace Scrfd{
    using namespace cv;
    using namespace std;

    void decode_kernel_invoker(
        float* predict, int num_bboxes, float confidence_threshold, 
        float nms_threshold, float* invert_affine_matrix, float* parray,
        int max_objects, float* prior,
        cudaStream_t stream
    );

    struct AffineMatrix{
        float i2d[6];       // image to dst(network), 2x3 matrix
        float d2i[6];       // dst to image, 2x3 matrix

        void compute(const cv::Size& from, const cv::Size& to){
            float scale_x = to.width / (float)from.width;
            float scale_y = to.height / (float)from.height;
            float scale = std::min(scale_x, scale_y);

            i2d[0] = scale;  i2d[1] = 0;  i2d[2] = -scale * from.width  * 0.5  + to.width * 0.5 + scale * 0.5 - 0.5;
            i2d[3] = 0;  i2d[4] = scale;  i2d[5] = -scale * from.height * 0.5 + to.height * 0.5 + scale * 0.5 - 0.5;

            // 有了i2d矩阵，我们求其逆矩阵，即可得到d2i（用以解码时还原到原始图像分辨率上）
            cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
            cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
            cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
        }

        cv::Mat i2d_mat(){
            return cv::Mat(2, 3, CV_32F, i2d);
        }
    };

    using ControllerImpl = InferController
    <
        Mat,                    // input
        BoxArray,              // output
        tuple<string, int>,     // start param
        AffineMatrix            // additional
    >;
    class InferImpl : public Infer, public ControllerImpl{
    public:
        /** 要求在InferImpl里面执行stop，而不是在基类执行stop **/
        virtual ~InferImpl(){
            stop();
        }
        
        virtual bool startup(const string& file, int gpuid, float confidence_threshold, float nms_threshold){

            float mean[] = {127.5, 127.5, 127.5};
            float std[]  = {128.0, 128.0, 128.0};
            normalize_   = CUDAKernel::Norm::mean_std(mean, std, 1.0f);
            confidence_threshold_ = confidence_threshold;
            nms_threshold_        = nms_threshold;
            return ControllerImpl::startup(make_tuple(file, gpuid));
        }

        size_t compute_prior_size(int input_width, int input_height, const vector<int>& strides={8, 16, 32}, int num_anchor_per_stage=2){

            int input_area = input_width * input_height;
            size_t total = 0;
            for(int s : strides){
                total += input_area / s / s * num_anchor_per_stage;
            }
            return total;
        }

        void init_prior_box(TRT::Tensor& prior, int input_width, int input_height){

            vector<int> strides{8, 16, 32};
            vector<vector<float>> min_sizes{
                vector<float>({16.0f,  32.0f }),
                vector<float>({64.0f,  128.0f}),
                vector<float>({256.0f, 512.0f})
            };
            prior.resize(1, compute_prior_size(input_width, input_height, strides), 4).to_cpu();
            
            int prior_row = 0;
            for(int istride = 0; istride < strides.size(); ++istride){
                int stride         = strides[istride];
                auto anchor_sizes  = min_sizes[istride];
                int feature_map_width  = input_width  / stride;
                int feature_map_height = input_height / stride;
                
                for(int y = 0; y < feature_map_height; ++y){
                    for(int x = 0; x < feature_map_width; ++x){
                        for(int isize = 0; isize < anchor_sizes.size(); ++isize){
                            float anchor_size = anchor_sizes[isize];
                            float dense_cx    = x * stride;
                            float dense_cy    = y * stride;
                            float s_kx        = stride;
                            float s_ky        = stride;
                            float* prow       = prior.cpu<float>(0, prior_row++);
                            prow[0] = dense_cx;
                            prow[1] = dense_cy;
                            prow[2] = s_kx;
                            prow[3] = s_ky;
                        }
                    }
                }
            }
            prior.to_gpu();
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

            const int MAX_IMAGE_BBOX = 1024;
            const int NUM_BOX_ELEMENT = 16;    // left, top, right, bottom, confidence, keepflag(1keep,0ignore), landmark(x, y) * 5
            TRT::Tensor affin_matrix_device(TRT::DataType::Float);
            TRT::Tensor output_array_device(TRT::DataType::Float);
            TRT::Tensor prior(TRT::DataType::Float);
            int max_batch_size = engine->get_max_batch_size();
            auto input         = engine->input();
            auto output        = engine->output();

            input_width_       = input->size(3);
            input_height_      = input->size(2);
            tensor_allocator_  = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
            stream_            = engine->get_stream();
            gpu_               = gpuid;
            result.set_value(true);

            init_prior_box(prior, input_width_, input_height_);
            input->resize_single_dim(0, max_batch_size).to_gpu();
            output->resize(max_batch_size, prior.size(1), 15).to_gpu();

            affin_matrix_device.set_stream(stream_);

            // 这里8个值的目的是保证 8 * sizeof(float) % 32 == 0
            affin_matrix_device.resize(max_batch_size, 8).to_gpu();

            // 这里的 1 + MAX_IMAGE_BBOX结构是，counter + bboxes ...
            output_array_device.resize(max_batch_size, 1 + MAX_IMAGE_BBOX * NUM_BOX_ELEMENT).to_gpu(); 

            vector<Job> fetch_jobs;
            while(get_jobs_and_wait(fetch_jobs, max_batch_size)){

                int infer_batch_size = fetch_jobs.size();
                input->resize_single_dim(0, infer_batch_size);

                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    auto& job  = fetch_jobs[ibatch];
                    auto& mono = job.mono_tensor->data();
                    affin_matrix_device.copy_from_gpu(affin_matrix_device.offset(ibatch), mono->get_workspace()->gpu(), 6);
                    input->copy_from_gpu(input->offset(ibatch), mono->gpu(), mono->count());
                    job.mono_tensor->release();
                }

                engine->forward(false);

                output_array_device.to_gpu(false);
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    auto& job                 = fetch_jobs[ibatch];
                    float* image_based_output = output->gpu<float>(ibatch);
                    float* output_array_ptr   = output_array_device.gpu<float>(ibatch);
                    auto affine_matrix        = affin_matrix_device.gpu<float>(ibatch);
                    checkCudaRuntime(cudaMemsetAsync(output_array_ptr, 0, sizeof(int), stream_));
                    decode_kernel_invoker(
                        image_based_output, 
                        output->size(1), confidence_threshold_, nms_threshold_, affine_matrix, 
                        output_array_ptr, MAX_IMAGE_BBOX, prior.gpu<float>(),
                        stream_
                    );
                }

                output_array_device.to_cpu();
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    float* parray = output_array_device.cpu<float>(ibatch);
                    int count     = min(MAX_IMAGE_BBOX, (int)*parray);
                    auto& job     = fetch_jobs[ibatch];
                    auto& image_based_boxes   = job.output;
                    for(int i = 0; i < count; ++i){
                        float* pbox = parray + 1 + i * NUM_BOX_ELEMENT;
                        int keepflag = pbox[5];
                        if(keepflag == 1){
                            Box box;
                            box.left       = pbox[0];
                            box.top        = pbox[1];
                            box.right      = pbox[2];
                            box.bottom     = pbox[3];
                            box.confidence = pbox[4];
                            memcpy(box.landmark, pbox + 6, sizeof(box.landmark));
                            image_based_boxes.emplace_back(box);
                        }
                    }
                    job.pro->set_value(image_based_boxes);
                }
                fetch_jobs.clear();
            }
            stream_ = nullptr;
            tensor_allocator_.reset();
            INFOV("Engine destroy.");
        }

        virtual bool preprocess(Job& job, const Mat& image) override{

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

            Size input_size(input_width_, input_height_);
            job.additional.compute(image.size(), input_size);
            
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
            memcpy(affine_matrix_host, job.additional.d2i, sizeof(job.additional.d2i));
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

        virtual vector<shared_future<BoxArray>> commits(const vector<Mat>& images) override{
            return ControllerImpl::commits(images);
        }

        virtual std::shared_future<BoxArray> commit(const Mat& image) override{
            return ControllerImpl::commit(image);
        }

    private:
        int input_width_            = 0;
        int input_height_           = 0;
        int gpu_                    = 0;
        float confidence_threshold_ = 0;
        float nms_threshold_        = 0;
        TRT::CUStream stream_       = nullptr;
        CUDAKernel::Norm normalize_;
    };

    tuple<cv::Mat, Box> crop_face_and_landmark(const cv::Mat& image, const Box& box, float scale_box){
        
        float padding_x = (scale_box - 1) * box.width() * 0.5f;
        float padding_y = (scale_box - 1) * box.height() * 0.5f;
        int left   = std::round(box.left   - padding_x);
        int top    = std::round(box.top    - padding_y);
        int right  = std::round(box.right  + padding_x);
        int bottom = std::round(box.bottom + padding_y);

        Rect rbox(left, top, right-left, bottom-top);
        rbox = rbox & Rect(0, 0, image.cols, image.rows);

        auto box_copy = box;
        for(int i = 0; i < 10; ++i){
            if(i % 2 == 0){
                // x
                box_copy.landmark[i] -= left;
            }else{
                box_copy.landmark[i] -= top;
            }
        }

        box_copy.left   -= left;
        box_copy.top    -= top;
        box_copy.right  -= left;
        box_copy.bottom -= top;

        if(rbox.width < 1 || rbox.height < 1)
            return make_tuple(Mat(), box_copy);

        return make_tuple(image(rbox).clone(), box_copy);
    }

    shared_ptr<Infer> create_infer(const string& engine_file, int gpuid, float confidence_threshold, float nms_threshold){
        shared_ptr<InferImpl> instance(new InferImpl());
        if(!instance->startup(engine_file, gpuid, confidence_threshold, nms_threshold)){
            instance.reset();
        }
        return instance;
    }
};