#include "yolo_fast.hpp"
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

namespace YoloFast{
    using namespace cv;
    using namespace std;

    DecodeMeta DecodeMeta::x_default_meta(){
        DecodeMeta meta;
        meta.num_anchor = 1;
        meta.num_level = 3;

        const int strides[] = {8, 16, 32};
        memcpy(meta.strides, strides, sizeof(meta.strides));
        return meta;
    }

    DecodeMeta DecodeMeta::v5_p6_default_meta(){
        DecodeMeta meta;
        meta.num_anchor = 3;
        meta.num_level = 4;

        float anchors[] = {
            19, 27,   44, 40,   38, 94,
            96, 68,   86, 152,  180,137,
            140,301,  303,264,  238,542,
            436,615,  739,380,  925,792
        };  

        int abs_index = 0;
        for(int i = 0; i < meta.num_level; ++i){
            for(int j = 0; j < meta.num_anchor; ++j){
                int aidx = i * meta.num_anchor + j;
                meta.w[aidx] = anchors[abs_index++];
                meta.h[aidx] = anchors[abs_index++];
            }
        }

        const int strides[] = {8, 16, 32, 64};
        memcpy(meta.strides, strides, sizeof(meta.strides));
        return meta;
    }

    DecodeMeta DecodeMeta::v5_p5_default_meta(){
        DecodeMeta meta;
        meta.num_anchor = 3;
        meta.num_level = 3;

        float anchors[] = {
            10.000000, 13.000000, 16.000000, 30.000000, 33.000000, 23.000000,
            30.000000, 61.000000, 62.000000, 45.000000, 59.000000, 119.000000,
            116.000000, 90.000000, 156.000000, 198.000000, 373.000000, 326.000000
        };  

        int abs_index = 0;
        for(int i = 0; i < meta.num_level; ++i){
            for(int j = 0; j < meta.num_anchor; ++j){
                int aidx = i * meta.num_anchor + j;
                meta.w[aidx] = anchors[abs_index++];
                meta.h[aidx] = anchors[abs_index++];
            }
        }

        const int strides[] = {8, 16, 32};
        memcpy(meta.strides, strides, sizeof(meta.strides));
        return meta;
    }

    const char* type_name(Type type){
        switch(type){
        case Type::V5_P5: return "YoloV5_P5";
        case Type::V5_P6: return "YoloV5_P6";
        case Type::X: return "YoloX";
        default: return "Unknow";
        }
    }

    void yolov5_decode_kernel_invoker(
        float* predict, int num_bboxes, int fm_area, int num_classes, float confidence_threshold, 
        float nms_threshold, float* invert_affine_matrix, float* parray, const float* prior_box,
        int max_objects, cudaStream_t stream
    );

    void yolox_decode_kernel_invoker(
        float* predict, int num_bboxes, int fm_area, int num_classes, float confidence_threshold, 
        float nms_threshold, float* invert_affine_matrix, float* parray, const float* prior_box,
        int max_objects, cudaStream_t stream
    );

    struct AffineMatrix{
        float i2d[3];       // image to dst(network)
        float d2i[3];       // dst to image

        void compute(const cv::Size& from, const cv::Size& to){
            float scale_x = to.width / (float)from.width;
            float scale_y = to.height / (float)from.height;
            float scale = std::min(scale_x, scale_y);
            i2d[0] = scale;  i2d[1] = -scale * from.width  * 0.5  + to.width * 0.5 + scale * 0.5 - 0.5;
                             i2d[2] = -scale * from.height * 0.5 + to.height * 0.5 + scale * 0.5 - 0.5;

            // y = kx + b
            // x = (y-b)/k = y*1/k + (-b)/k
            d2i[0] = 1/scale; d2i[1] = -i2d[1] / scale;
                              d2i[2] = -i2d[2] / scale;
        }
    };

    using ControllerImpl = InferController
    <
        Mat,                    // input
        BoxArray,         // output
        tuple<string, int>,     // start param
        AffineMatrix            // additional
    >;
    class InferImpl : public Infer, public ControllerImpl{
    public:

        /** 要求在InferImpl里面执行stop，而不是在基类执行stop **/
        virtual ~InferImpl(){
            stop();
        }

        virtual bool startup(const string& file, Type type, int gpuid, float confidence_threshold, float nms_threshold, const DecodeMeta& meta){

            meta_ = meta;
            type_ = type;

            if(type == Type::V5_P5 || type == Type::V5_P6){
                normalize_ = CUDAKernel::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDAKernel::ChannelType::Invert);
            }else if(type == Type::X){
                //float mean[] = {0.485, 0.456, 0.406};
                //float std[]  = {0.229, 0.224, 0.225};
                //normalize_ = CUDAKernel::Norm::mean_std(mean, std, 1/255.0f, CUDAKernel::ChannelType::Invert);
                meta_.num_anchor = 1;
                normalize_ = CUDAKernel::Norm::None();
            }else{
                INFOE("Unsupport type %d", type);
            }
            
            confidence_threshold_ = confidence_threshold;
            nms_threshold_        = nms_threshold;
            return ControllerImpl::startup(make_tuple(file, gpuid));
        }

        void init_yolox_prior_box(TRT::Tensor& prior_box){
            
            // 8400(lxaxhxw) x 3
            float* prior_ptr = prior_box.cpu<float>();
            for(int ianchor = 0; ianchor < meta_.num_anchor; ++ianchor){
                for(int ilevel = 0; ilevel < meta_.num_level; ++ilevel){
                    int stride    = meta_.strides[ilevel];
                    int fm_width  = input_width_ / stride;
                    int fm_height = input_height_ / stride;
                    int anchor_abs_index = ilevel * meta_.num_anchor + ianchor;
                    for(int ih = 0; ih < fm_height; ++ih){
                        for(int iw = 0; iw < fm_width; ++iw){
                            *prior_ptr++ = iw;
                            *prior_ptr++ = ih;
                            *prior_ptr++ = stride;
                        }
                    }
                }
            }
            prior_box.to_gpu();
        }

        void init_yolov5_prior_box(TRT::Tensor& prior_box){
            
            // 25200(lxaxhxw) x 5
            float* prior_ptr = prior_box.cpu<float>();
            for(int ianchor = 0; ianchor < meta_.num_anchor; ++ianchor){
                for(int ilevel = 0; ilevel < meta_.num_level; ++ilevel){
                    int stride    = meta_.strides[ilevel];
                    int fm_width  = input_width_ / stride;
                    int fm_height = input_height_ / stride;
                    int anchor_abs_index = ilevel * meta_.num_anchor + ianchor;
                    for(int ih = 0; ih < fm_height; ++ih){
                        for(int iw = 0; iw < fm_width; ++iw){
                            *prior_ptr++ = iw;
                            *prior_ptr++ = ih;
                            *prior_ptr++ = meta_.w[anchor_abs_index];
                            *prior_ptr++ = meta_.h[anchor_abs_index];
                            *prior_ptr++ = stride;
                        }
                    }
                }
            }
            prior_box.to_gpu();
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

            const int MAX_IMAGE_BBOX  = 1024;
            const int NUM_BOX_ELEMENT = 7;      // left, top, right, bottom, confidence, class, keepflag
            TRT::Tensor affin_matrix_device(TRT::DataType::Float);
            TRT::Tensor output_array_device(TRT::DataType::Float);
            TRT::Tensor prior_box(TRT::DataType::Float);
            int max_batch_size = engine->get_max_batch_size();
            auto input         = engine->tensor("images");
            auto output        = engine->tensor("output");
            int num_classes    = output->size(2) - 5;

            input_width_       = input->size(3) * 2;  /** 移除focus后要乘以2 **/
            input_height_      = input->size(2) * 2;  /** 移除focus后要乘以2 **/
            tensor_allocator_  = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
            stream_            = engine->get_stream();
            gpu_               = gpuid;
            result.set_value(true);

            input->resize_single_dim(0, max_batch_size).to_gpu();
            affin_matrix_device.set_stream(stream_);

            // 这里8个值的目的是保证 8 * sizeof(float) % 32 == 0
            affin_matrix_device.resize(max_batch_size, 8).to_gpu();

            // 这里的 1 + MAX_IMAGE_BBOX结构是，counter + bboxes ...
            output_array_device.resize(max_batch_size, 1 + MAX_IMAGE_BBOX * NUM_BOX_ELEMENT).to_gpu();

            // 25200(axhxw) x 5
            bool is_v5 = type_ == Type::V5_P5 || type_ == Type::V5_P6;
            if(is_v5){
                prior_box.resize(output->size(1) * output->size(3), 5).to_cpu();
                init_yolov5_prior_box(prior_box);
            }else{
                prior_box.resize(output->size(1) * output->size(3), 3).to_cpu();
                init_yolox_prior_box(prior_box);
            }

            auto decode_kernel_invoker = is_v5 ? yolov5_decode_kernel_invoker : yolox_decode_kernel_invoker;
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
                        output->size(1) * output->size(3),
                        output->size(3), 
                        num_classes, 
                        confidence_threshold_, 
                        nms_threshold_, 
                        affine_matrix, 
                        output_array_ptr, 
                        prior_box.gpu<float>(),
                        MAX_IMAGE_BBOX, 
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
                        float* pbox  = parray + 1 + i * NUM_BOX_ELEMENT;
                        int label    = pbox[5];
                        int keepflag = pbox[6];
                        if(keepflag == 1){
                            image_based_boxes.emplace_back(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label);
                        }
                    }
                    job.pro->set_value(image_based_boxes);
                }
                fetch_jobs.clear();
            }
            stream_ = nullptr;
            tensor_allocator_.reset();
            INFO("Engine destroy.");
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

            /** 移除focus后，宽高是1/2 **/
            tensor->resize(1, 12, input_height_ / 2, input_width_ / 2);

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

            CUDAKernel::warp_affine_bilinear_and_normalize_focus(
                image_device,         image.cols * 3,       image.cols,       image.rows, 
                tensor->gpu<float>(), input_width_,         input_height_, 
                affine_matrix_device, 114, 
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
        DecodeMeta meta_;
        Type type_;
    };

    shared_ptr<Infer> create_infer(
        const string& engine_file, 
        Type type, 
        int gpuid, 
        float confidence_threshold, 
        float nms_threshold, 
        const DecodeMeta& meta
    ){
        shared_ptr<InferImpl> instance(new InferImpl());
        if(!instance->startup(engine_file, type, gpuid, confidence_threshold, nms_threshold, meta)){
            instance.reset();
        }
        return instance;
    }

    void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, Type type, int ibatch){

        CUDAKernel::Norm normalize;
        if(type == Type::V5_P5 || type == Type::V5_P6){
            normalize = CUDAKernel::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDAKernel::ChannelType::Invert);
        }else if(type == Type::X){
            //float mean[] = {0.485, 0.456, 0.406};
            //float std[]  = {0.229, 0.224, 0.225};
            //normalize_ = CUDAKernel::Norm::mean_std(mean, std, 1/255.0f, CUDAKernel::ChannelType::Invert);
            normalize = CUDAKernel::Norm::None();
        }else{
            INFOE("Unsupport type %d", type);
        }
        
        Size input_size(tensor->size(3) * 2, tensor->size(2) * 2);
        AffineMatrix affine;
        affine.compute(image.size(), input_size);

        size_t size_image      = image.cols * image.rows * 3;
        size_t size_matrix     = iLogger::upbound(sizeof(affine.d2i), 32);
        auto workspace         = tensor->get_workspace();
        uint8_t* gpu_workspace        = (uint8_t*)workspace->gpu(size_matrix + size_image);
        float*   affine_matrix_device = (float*)gpu_workspace;
        uint8_t* image_device         = size_matrix + gpu_workspace;

        uint8_t* cpu_workspace        = (uint8_t*)workspace->cpu(size_matrix + size_image);
        float* affine_matrix_host     = (float*)cpu_workspace;
        uint8_t* image_host           = size_matrix + cpu_workspace;
        auto stream                   = tensor->get_stream();

        memcpy(image_host, image.data, size_image);
        memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
        checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream));
        checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i), cudaMemcpyHostToDevice, stream));

        CUDAKernel::warp_affine_bilinear_and_normalize_focus(
            image_device,               image.cols * 3,       image.cols,       image.rows, 
            tensor->gpu<float>(ibatch), input_size.width,     input_size.height, 
            affine_matrix_device, 114, 
            normalize, stream
        );
        tensor->synchronize();
    }
};