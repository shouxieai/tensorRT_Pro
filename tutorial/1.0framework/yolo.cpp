#include "yolo.hpp"
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


/* 
中文指南请下滑

Overview and Guide:
    Please focus on shared_ptr<Infer> create_infer(......) first.
    When we create an infer module by the above code, we pass in **.trtmodel, gpuid and some threshold etc.
    Then we allocate a new storage space on heap by operator new. Note that the type we offer is InferImpl, which is 
    different from the returned type of the create_infer(.....). To put it in a nutshell, We offer implmentation type(child class),
    but we return interface type(parent class), which avoids misoperation in many scenario. For this advantage, we will discuss later.

    !It's always recommended to have a glance at the code involved by using ctrl/alt F to use full-text search 
    better understanding and try to compare the idea of text and the impl of the code. So it's fine to jump to the code 
    and jump back to the text and continue reading)
    Now it's time to introduce a key concept --RAII. (Tips: search for the RAII keywords to check the code involved)

    RAII in the context can be unsterstood in the following mapping:
        create infer module  <------>  Resource acquisition
        instance->startup    <------>  is Initialization


    Now let's look at the startup(). It's also good to check other member functions that class InferImpl has such that
    you might have a rough idea of how infer module works. Simplily speaking, 
        - an infer module needs to be started up. 
        - It receives some inputs like images by using commit func
        - and gives it to worker func
        - within worker func, we need to preprocess the input.
    
    
    After the rough idea, let's move on to startup details. When we unfold the function startup, 
    we go for a yolo type and a custom normalize method. Then we return ControllerImpl::startup(....), which 
    might be a little confusing. But let's do a summary and preview for other related concepts now.

    For a better encapsulation and more user-friendly, which means users don't need to care about too much details when
    they use it, We design the following 3 classes
        - class Infer      (parent class)
        - class InferImpl  (child class)
        - class InferController (which is also named as class InferImpl by typedef) (parent class)
        
        Class Infer is the interface for user to call. Class InferImpl and class InferController share some attributes
        and methods. The reason class InferController was introduced is that we might have different models (e.g. yolo 
        , centernet or unet etc) and we integrate their common method into class InferController, for example, startup, somtimes preprocessing method.

    Now let's continue and go into class InferController<.....> (src/tensorRT/common/infer_controller.hpp)


概述和指南:
请先关注shared_ptr<Infer> create_infer(......)。
当我们通过上面的代码创建一个infer模块时，我们传入**.Trtmodel, gpuid和一些阈值等。然后我们用new来分配一个新的空间, 注意, 我们提供的类型是InferImpl,
与create_infer(.....)的返回类型不同。简而言之，我们提供实现类型(子类)，但是我们返回接口类型(父类)，这在许多场景中避免了误操作。这个优势，我们以后再讨论。

推荐使用ctrl/alt F来进行全文搜索以便更好地理解，同时尝试比较文本的idea和代码的实现。所以十分建议在代码和注释间反复跳跃阅读

现在是时候介绍一个关键概念——RAII了。(提示:搜索RAII关键字以检查所涉及的代码)
RAII在上下文中可以通过以下对应关系来理解:
create infer module <------>资源获取
instance->startup<------>初始化


现在让我们看看startup()。建议也同时看一下类InferImpl的其他成员函数
您可能对infer模块的工作原理有了一个粗略的了解。简单地说,
- 需要启动infer模块。
- 它通过使用commit func提交一些输入，比如图像
- 并将其赋给worker func
- 在worker func中，我们需要对输入进行预处理。


在粗略的概念之后，让我们进入startup的细节。当我们展开函数startup时，
我们选择yolo类型和normalize 方法，然后返回ControllerImpl::startup(....)
这里可能会让人有点困惑，但现在让我们做一个总结和预览其他相关概念。
为了更好的封装和更人性化，这意味着用户不需要关心太多的细节，

我们设计了以下3个类
    - class Infer      (parent class)
    - class InferImpl  (child class)
    - class InferController (which is also named as class InferImpl by typedef) (parent class)

类Infer是用户调用的接口。类InferImpl和类InferController共享一些属性和方法。引入类InferController的原因是我们可能有不同的模型(例如yolo
我们将它们的常用方法集成到类InferController中，例如，startup，有时是预处理方法。
现在让我们继续，进入类InferController<.....> (src / tensorRT /common/ infer_controller.hpp)



 */



namespace Yolo{
    using namespace cv;
    using namespace std;

    const char* type_name(Type type){
        switch(type){
        case Type::V5: return "YoloV5";
        case Type::X: return "YoloX";
        default: return "Unknow";
        }
    }

    void decode_kernel_invoker(
        float* predict, int num_bboxes, int num_classes, float confidence_threshold, 
        float nms_threshold, float* invert_affine_matrix, float* parray,
        int max_objects, cudaStream_t stream
    );

    struct AffineMatrix{
        float i2d[6];       // image to dst(network), 2x3 matrix
        float d2i[6];       // dst to image, 2x3 matrix

        void compute(const cv::Size& from, const cv::Size& to){
            float scale_x = to.width / (float)from.width;
            float scale_y = to.height / (float)from.height;

            // 这里取min的理由是
            // 1. M矩阵是 from * M = to的方式进行映射，因此scale的分母一定是from
            // 2. 取最小，即根据宽高比，算出最小的比例，如果取最大，则势必有一部分超出图像范围而被裁剪掉，这不是我们要的
            // **
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

            i2d[0] = scale;  i2d[1] = 0;  i2d[2] = -scale * from.width  * 0.5  + to.width * 0.5;
            i2d[3] = 0;  i2d[4] = scale;  i2d[5] = -scale * from.height * 0.5 + to.height * 0.5;

            cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
            cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
            cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
        }

        cv::Mat i2d_mat(){
            return cv::Mat(2, 3, CV_32F, i2d);
        }
    };

    //* kp: alias declaration  ref: https://en.cppreference.com/w/cpp/language/type_alias
    using ControllerImpl = InferController
    <
        Mat,                    // input
        ObjectBoxArray,         // output
        tuple<string, int>,     // start param
        AffineMatrix            // additional
    >;

    
    class InferImpl : public Infer, public ControllerImpl{
    public:
        virtual bool startup(const string& file, Type type, int gpuid, float confidence_threshold, float nms_threshold){

            if(type == Type::V5 || type == Type::V3 || type == Type::V7){
                normalize_ = CUDAKernel::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDAKernel::ChannelType::Invert);
            }else if(type == Type::X){
                float mean[] = {0.485, 0.456, 0.406};
                float std[]  = {0.229, 0.224, 0.225};
                normalize_ = CUDAKernel::Norm::mean_std(mean, std, 1/255.0f, CUDAKernel::ChannelType::Invert);
                // normalize_ = CUDAKernel::Norm::None();
            }else{
                INFOE("Unsupport type %d", type);
            }
            
            confidence_threshold_ = confidence_threshold;
            nms_threshold_        = nms_threshold;
            return ControllerImpl::startup(make_tuple(file, gpuid)); //* kp: make_tupe. ref: https://en.cppreference.com/w/cpp/utility/tuple/make_tuple
        }

        virtual void worker(promise<bool>& result) override{ // the result var is passed as a promise, which will be returned by setting value.
            
            string file = get<0>(start_param_); //* kp: std::get(std::tuple)  https://en.cppreference.com/w/cpp/utility/tuple/get
            int gpuid   = get<1>(start_param_); // we can access start_praram_ here because we pass 'this (the parent class)' as a param in @infer_controller.hpp 

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
            int max_batch_size = engine->get_max_batch_size();
            auto input         = engine->tensor("images");
            auto output        = engine->tensor("output");
            int num_classes    = output->size(2) - 5;

            input_width_       = input->size(3);
            input_height_      = input->size(2);
            tensor_allocator_  = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
            stream_            = engine->get_stream();
            gpu_               = gpuid;
            result.set_value(true);     // true will be returned in promise-future way.

            input->resize_single_dim(0, max_batch_size).to_gpu(); // resize_single_dim can be seen as a lightweigth version of resize, to which you offer fewer paramterss.
            affin_matrix_device.set_stream(stream_);

            // why 8 ? : 8 * sizeof(float) % 32 == 0
            affin_matrix_device.resize(max_batch_size, 8).to_gpu(); // e.g. 16 images as a batch. Each image goes with an affine_matrix

            // 1 image has n detected bboxexs, which can be expressed as counter, bbox0, bbox1...bboxn
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
                    decode_kernel_invoker(image_based_output, output->size(1), num_classes, confidence_threshold_, nms_threshold_, affine_matrix, output_array_ptr, MAX_IMAGE_BBOX, stream_);
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
            INFO("Engine destroy.");
        }

        virtual bool preprocess(Job& job, const Mat& image) override{
            job.mono_tensor = tensor_allocator_->query(); //todo 
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

            checkCudaRuntime(cudaMemcpyAsync(image_host,   image.data, size_image, cudaMemcpyHostToHost,   stream_));
            checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_));
            checkCudaRuntime(cudaMemcpyAsync(affine_matrix_host, job.additional.d2i, sizeof(job.additional.d2i), cudaMemcpyHostToHost, stream_));
            checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(job.additional.d2i), cudaMemcpyHostToDevice, stream_));

            CUDAKernel::warp_affine_bilinear_and_normalize( //todo
                image_device,         image.cols * 3,       image.cols,       image.rows, 
                tensor->gpu<float>(), input_width_,         input_height_, 
                affine_matrix_device, 114, 
                normalize_, stream_
            );
            return true;
        }

        virtual vector<shared_future<ObjectBoxArray>> commits(const vector<Mat>& images) override{
            return ControllerImpl::commits(images);
        }

        virtual std::shared_future<ObjectBoxArray> commit(const Mat& image) override{
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

    shared_ptr<Infer> create_infer(const string& engine_file, Type type, int gpuid, float confidence_threshold, float nms_threshold){
        shared_ptr<InferImpl> instance(new InferImpl()); //* kp: RAII Resource Acquision Is Initialization.
        if(!instance->startup(engine_file, type, gpuid, confidence_threshold, nms_threshold)){
            instance.reset();
        }
        return instance;
    }
};