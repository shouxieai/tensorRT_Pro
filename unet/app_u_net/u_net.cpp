#include "u_net.hpp"
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

namespace U_net{
    using namespace cv;
    using namespace std;

    
    static string outname;
    void decode_kernel_u_net_invoker(
        float* predict, 
        int num_classes,  
        float* parray, 
        Size src_size, 
        int channels,
        cudaStream_t stream);

    // 传的有原数据的地址(CPU)的指针, 类别数, 输出数据内存的指针
    void decode_kernel_invoker(
        shared_ptr<TRT::Tensor> predict, int num_classes, float* parray
    );

    Mat add_mask(float* parray);

    Mat new_mask(shared_ptr<TRT::Tensor> mask_ptr, float* affine_matrix_ptr, Size size_ori);
    

    // 因为图像需要进行预处理，这里采用仿射变换warpAffine进行处理，因此在这里计算仿射变换的矩阵
    struct AffineMatrix{
        float i2d[6];       // image to dst(network), 2x3 matrix
        float d2i[6];       // dst to image, 2x3 matrix

        void compute(const cv::Size& from, const cv::Size& to){
            float scale_x = to.width / (float)from.width;      
            float scale_y = to.height / (float)from.height;    

            // 这里取min的理由是
            // 1. M矩阵是 from * M = to的方式进行映射，因此scale的分母一定是from
            // 2. 取最小，即根据宽高比，算出最小的比例，如果取最大，则势必有一部分超出图像范围而被裁剪掉，这不是我们要的
            float scale = std::min(scale_x, scale_y);

            /**
            这里的仿射变换矩阵实质上是2x3的矩阵，具体实现是
            scale, 0, -scale * from.width * 0.5 + to.width * 0.5
            0, scale, -scale * from.height * 0.5 + to.height * 0.5
            
            这里可以想象成，是经历过缩放、平移、平移三次变换后的组合，M = TPS
            例如第一个S矩阵，定义为把输入的from图像，等比缩放scale倍，到to尺度下
            S = [
            scale,     0,      0
            0,     scale,      0
            0,         0,      1
            ]
            
            P矩阵定义为第一次平移变换矩阵，将图像的原点，从左上角，移动到缩放(scale)后图像的中心上
            P = [
            1,        0,      -scale * from.width * 0.5
            0,        1,      -scale * from.height * 0.5
            0,        0,                1
            ]

            T矩阵定义为第二次平移变换矩阵，将图像从原点移动到目标（to）图的中心上
            T = [
            1,        0,      to.width * 0.5,
            0,        1,      to.height * 0.5,
            0,        0,            1
            ]

            通过将3个矩阵顺序乘起来，即可得到下面的表达式：
            M = [
            scale,    0,     -scale * from.width * 0.5 + to.width * 0.5
            0,     scale,    -scale * from.height * 0.5 + to.height * 0.5
            0,        0,                     1
            ]
            去掉第三行就得到opencv需要的输入2x3矩阵
            **/
            // 1   0  center  0  1  center 
            i2d[0] = scale;  i2d[1] = 0;  i2d[2] = -scale * from.width  * 0.5  + to.width * 0.5;
            i2d[3] = 0;  i2d[4] = scale;  i2d[5] = -scale * from.height * 0.5 + to.height * 0.5;

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
        SegmentResult,          // output
        tuple<string, int>,     // start param
        AffineMatrix            // additional
    >;
    class InferImpl : public Infer, public ControllerImpl{
    public:
        virtual bool startup(const string& file, int gpuid){

            normalize_ = CUDAKernel::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDAKernel::ChannelType::Invert);
            return ControllerImpl::startup(make_tuple(file, gpuid));
        }

        virtual void worker(promise<bool>& result) override{

            string file = get<0>(start_param_);   // file   u_net.fp32.trt_model
            int gpuid   = get<1>(start_param_);   // gpuid  0

            TRT::set_device(gpuid);               // 0
            auto engine = TRT::load_infer(file);  // 加载 trtmodel
            if(engine == nullptr){
                INFOE("Engine %s load failed", file.c_str());
                result.set_value(false);          // pro 设置为 false
                return;
            }

            engine->print();                      // 打印推理细节

            TRT::Tensor affin_matrix_device(TRT::DataType::Float);
            int max_batch_size = engine->get_max_batch_size();
            auto input         = engine->tensor("images");
            auto output        = engine->tensor("output");  // (1, 21, 512, 512)
            // bool dynamic_batch = engine->is_dynamic_batch_dimension();
            int num_classes    = output->size(1);  // 21

            input_width_       = input->size(3);  // 512
            input_height_      = input->size(2);  // 512 
            tensor_allocator_  = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);  // 设置独占分配器的大小
            stream_            = engine->get_stream();
            gpu_               = gpuid;
            result.set_value(true);      // 启动成功

            // 预先分配好内存
            input->resize_single_dim(0, max_batch_size).to_gpu();  
            affin_matrix_device.set_stream(stream_);

            // 这里8个值的目的是保证 8 * sizeof(float) % 32 == 0
            affin_matrix_device.resize(max_batch_size, 8).to_gpu();
            shared_ptr<TRT::Tensor> parray_ptr = make_shared<TRT::Tensor>(TRT::DataType::Float);  // 定义的一个掩码输出目前：512 512 3

            parray_ptr -> resize(512, 512, 3);   //h, w, c  1800 810  3
            // 仿射变换之后的输出图像
            parray_ptr -> to_gpu(false);  // GPU 
            vector<Job> fetch_jobs; 
            Size src_size(output -> shape(2), output -> shape(3));  // w, h
            while(get_jobs_and_wait(fetch_jobs, max_batch_size)){    

                int infer_batch_size = fetch_jobs.size();
                INFO("拿到的任务一共 %d 个", infer_batch_size);
                // 取任务, 拿到预处理后的image，并且拷贝affin_matrix_device到CPU
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    auto& job  = fetch_jobs[ibatch];
                    auto& mono = job.mono_tensor->data();   // 取到 tensor 即: 预处理后的 image
                    // 先判断 affin_matrix_device 在GPU上的位置，
                    affin_matrix_device.copy_from_gpu(affin_matrix_device.offset(ibatch), mono->get_workspace()->gpu(), 6);
                    input->copy_from_gpu(input->offset(ibatch), mono->gpu(), mono->count());
                    job.mono_tensor->release();
                }
                
                // 模型推理
                engine->forward(false);

                // output -> save_to_file("out_data");  // 此时在 gpu
                // GPU 解码  适用于单张图的inference
                parray_ptr -> to_gpu(false);  // GPU 
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    // 得到job
                    auto& job                 = fetch_jobs[ibatch];  // ibatch = 1
                    output -> resize(21, 512, 512);
                    auto image_based_output = output->gpu<float>();  // 得到输出的图像
                    auto affine_matrix        = affin_matrix_device.gpu<float>(ibatch);  // 得到仿射矩阵 可以直接用

                    int channels = int(ori_image.channels());
                    float* parray_gpu_ptr = parray_ptr -> gpu<float>(0);
                    // GPU 核函数进行解码 以及 warpAffine
                    decode_kernel_u_net_invoker(image_based_output, num_classes, parray_gpu_ptr, src_size, channels, stream_);
                    }

                parray_ptr -> to_cpu();
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    auto& job     = fetch_jobs[ibatch];
                    auto& img_matrix = job.output;
                    float* network_out  = parray_ptr -> cpu<float>(ibatch);
                    float* invertMatrix_ptr = affin_matrix_device.gpu<float>(ibatch);
                    INFO("仿射矩阵的第一个数字: %f ", invertMatrix_ptr[0]);
                    Mat final_image(src_size, CV_32FC3, network_out);
                    Mat invertMatrix(2, 3, CV_32F, invertMatrix_ptr);
                    INFO("当前size: %d, %d ",final_image.cols, final_image.rows);

                    
                    final_image.convertTo(final_image, CV_8UC3);
                    cv::Mat final;
                    cv::imwrite("img_img.jpg", final_image);
                    if (ibatch==2)
                        cv::imwrite("img_img_1.jpg", final_image);
                    img_matrix.emplace_back(final_image, invertMatrix);
                    job.pro->set_value(img_matrix);
                }
                fetch_jobs.clear();
            }
            INFO("Engine destroy.");
        }

        virtual bool preprocess(Job& job, const Mat& image) override{
            ori_image = image.clone();
            make_tuple(image);
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

            Size input_size(input_width_, input_height_);  // 从trtmodel 得到的网络输入  512   512  
            job.additional.compute(image.size(), input_size);  // 计算仿射矩阵 得到 长边为 512 的 matrix
            tensor->set_stream(stream_);  // 设置 stream 异步流
            tensor->resize(1, 3, input_height_, input_width_);   // 增加一维

            size_t size_image      = image.cols * image.rows * 3;  // 设置图像大小空间
            size_t size_matrix     = iLogger::upbound(sizeof(job.additional.d2i), 32);  // 设置仿射矩阵所需空间大小
            auto workspace         = tensor->get_workspace();  
            uint8_t* gpu_workspace        = (uint8_t*)workspace->gpu(size_matrix + size_image);  // 放到 gpu
            float*   affine_matrix_device = (float*)gpu_workspace;    // 指针类型强转
            uint8_t* image_device         = size_matrix + gpu_workspace;    // 定义总的 空间大小

            uint8_t* cpu_workspace        = (uint8_t*)workspace->cpu(size_matrix + size_image);  //  定义总的 CPU空间大小
            float* affine_matrix_host     = (float*)cpu_workspace;
            uint8_t* image_host           = size_matrix + cpu_workspace;

            checkCudaRuntime(cudaMemcpyAsync(image_host,   image.data, size_image, cudaMemcpyHostToHost,   stream_));
            checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_));
            checkCudaRuntime(cudaMemcpyAsync(affine_matrix_host, job.additional.d2i, sizeof(job.additional.d2i), cudaMemcpyHostToHost, stream_));
            checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(job.additional.d2i), cudaMemcpyHostToDevice, stream_));
            // image_device     image.cols * 3     image.cols:512     image.rows:480  gpu<float>  512   512 matrix_device, 114
            CUDAKernel::warp_affine_bilinear_and_normalize_plane(
                image_device,         image.cols * 3,       image.cols,       image.rows, 
                tensor->gpu<float>(), input_width_,         input_height_, 
                affine_matrix_device, 114, 
                normalize_, stream_
            );
            return true;
        }

        virtual vector<shared_future<SegmentResultArray>> commits(const vector<Mat>& images) override{
            return ControllerImpl::commits(images);
        }

        virtual std::shared_future<SegmentResultArray> commit(const Mat& image) override{
            return ControllerImpl::commit(image);
        }

    private:
        int input_width_            = 0;
        int input_height_           = 0;
        int gpu_                    = 0;
        float confidence_threshold_ = 0;
        float nms_threshold_        = 0;
        Mat ori_image;
        Mat i2d_device;
        Size old_size;
        Size new_size;
        TRT::CUStream stream_       = nullptr;
        CUDAKernel::Norm normalize_;
    };
    
    // U-net.fp32.trtmodel 开始做推理前的准备 构建， InferImpl
    shared_ptr<Infer> create_infer(const string& engine_file, int gpuid, float ){
        shared_ptr<InferImpl> instance(new InferImpl());
        if(!instance->startup(engine_file, gpuid)){
            instance.reset();
        }
        return instance;
    }
};