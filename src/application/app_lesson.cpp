
#include <common/ilogger.hpp>
#include <infer/trt_infer.hpp>
#include <builder/trt_builder.hpp>
#include "app_yolo/yolo.hpp"
#include <cuda_runtime.h>

using namespace std;

// static void test_tensor1(){

//     size_t cpu_bytes = 1024;
//     size_t gpu_bytes = 2048;

//     ///////////////////////////////////////////////////////////////////
//     // 封装效果，自动分配和释放
//     TRT::MixMemory memory;
//     void* host_ptr   = memory.cpu(cpu_bytes);
//     void* device_ptr = memory.gpu(gpu_bytes);

//     ///////////////////////////////////////////////////////////////////
//     // 不封装效果
//     void* host_ptr   = nullptr;
//     void* device_ptr = nullptr;
//     cudaMallocHost(&host_ptr, cpu_bytes);
//     cudaMalloc(&device_ptr, gpu_bytes);

//     cudaFreeHost(&host_ptr);
//     cudaFree(&device_ptr);
//     ///////////////////////////////////////////////////////////////////
// }

static void test_tensor2(){

    ///////////////////////////////////////////////////////////////////
    /* 内存的自动复制，依靠head属性标记数据最新的位置
       若访问的数据不是最新的，则会自动发生复制操作 */
    TRT::Tensor tensor({1, 3, 5, 5}, TRT::DataType::Float);
    INFO("tensor.head = %s", TRT::data_head_string(tensor.head()));   /* 输出 Init，内存没有分配 */

    tensor.cpu<float>()[0] = 512;               /* 访问cpu时，分配cpu内存 */
    INFO("tensor.head = %s", TRT::data_head_string(tensor.head()));   /* 输出 Host */

    float* device_ptr = tensor.gpu<float>();    /* 访问gpu时，最新数据在Host，发生复制动作并标记最新数据在Device */
    INFO("tensor.head = %s", TRT::data_head_string(tensor.head()));   /* 输出 Device */
    //INFO("device_ptr[0] = %f", device_ptr[0]);                        /* 输出 512.00000，由于gpu内存修改为cudaMalloc，这里无法直接访问 */
}

static void test_tensor3(){

    ///////////////////////////////////////////////////////////////////
    /* 计算维度的偏移量 */
    TRT::Tensor tensor({1, 3, 5, 5, 2, 5}, TRT::DataType::Float);
    auto ptr_origin   = tensor.cpu<float>();
    auto ptr_channel2 = tensor.cpu<float>(0, 2, 3, 2, 1, 3);

    INFO("Offset = %d", ptr_channel2 - ptr_origin);                          /* 输出678 */
    INFO("Offset = %d", tensor.offset(0, 2, 3, 2, 1, 3));                    /* 输出678 */

    int offset_compute = ((((0 * 3 + 2) * 5 + 3) * 5 + 2) * 2 + 1) * 5 + 3;  
    INFO("Compute = %d", offset_compute);                                    /* 输出678 */
}

static void lesson1(){

    /** 模型编译，onnx到trtmodel **/
    TRT::compile(
        TRT::Mode::FP32,            /** 模式, fp32 fp16 int8  **/
        1,                          /** 最大batch size        **/
        "lesson1.onnx",             /** onnx文件，输入         **/
        "lesson1.fp32.trtmodel"     /** trt模型文件，输出      **/
    );

    /** 加载编译好的引擎 **/
    auto infer = TRT::load_infer("lesson1.fp32.trtmodel");

    /** 设置输入的值 **/
    infer->input(0)->set_to(1.0f);

    /** 引擎进行推理 **/
    infer->forward();

    /** 取出引擎的输出并打印 **/
    auto out = infer->output(0);
    INFO("out.shape = %s", out->shape_string());
    for(int i = 0; i < out->channel(); ++i)
        INFO("%f", out->at<float>(0, i));
}

/** 动态batch **/
static void lesson2(){

    int max_batch_size = 5;
    /** 模型编译，onnx到trtmodel **/
    TRT::compile(
        TRT::Mode::FP32,            /** 模式, fp32 fp16 int8  **/
        max_batch_size,             /** 最大batch size        **/
        "lesson1.onnx",             /** onnx文件，输入         **/
        "lesson1.fp32.trtmodel"     /** trt模型文件，输出      **/
    );

    /** 加载编译好的引擎 **/
    auto infer = TRT::load_infer("lesson1.fp32.trtmodel");

    /** 设置输入的值 **/
    /** 修改input的0维度为1，最大可以是5 **/
    infer->input(0)->resize_single_dim(0, 2);
    infer->input(0)->set_to(1.0f);

    /** 引擎进行推理 **/
    infer->forward();

    /** 取出引擎的输出并打印 **/
    auto out = infer->output(0);
    INFO("out.shape = %s", out->shape_string());
}

/** 动态宽高-相对的，仅仅调整onnx输入大小为目的 **/
static void lesson3(){

    TRT::set_layer_hook_reshape([](const string& name, const vector<int64_t>& shape)->vector<int64_t>{
        INFO("name: %s,  shape: %s", name.c_str(), iLogger::join_dims(shape).c_str());
        return {-1, 25};
    });

    /** 模型编译，onnx到trtmodel **/
    TRT::compile(
        TRT::Mode::FP32,            /** 模式, fp32 fp16 int8  **/
        1,                          /** 最大batch size        **/
        "lesson1.onnx",             /** onnx文件，输入         **/
        "lesson1.fp32.trtmodel",    /** trt模型文件，输出      **/
        {{1, 1, 5, 5}}              /** 对输入的重定义         **/
    );

    auto infer = TRT::load_infer("lesson1.fp32.trtmodel");
    auto out = infer->output(0);
    INFO("out.shape = %s", out->shape_string());
}

void lesson_cache1frame(){

    iLogger::set_log_level(iLogger::LogLevel::Info);
    auto model_file = "yolox_s.FP32.trtmodel";
    auto onnx_file  = "yolox_s.onnx";

    if(not iLogger::exists(model_file)){
        TRT::compile(
            TRT::Mode::FP32,            // FP32、FP16、INT8
            16,                         // max batch size
            onnx_file,                  // source 
            model_file                  // save to
        );
    }

    auto yolo = Yolo::create_infer(model_file, Yolo::Type::X, 0, 0.4f);
    if(yolo == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    //////////////////基础耗时////////////////////////
    {
        cv::VideoCapture cap("exp/face_tracker.mp4");
        cv::Mat image;
        int iframe = 0;
        auto t0 = iLogger::timestamp_now_float();

        while(iframe < 300 && cap.read(image)){
            /** 模拟读取摄像头的延迟 **/
            iLogger::sleep(40);
            iframe++;
        }

        auto fee = iLogger::timestamp_now_float() - t0;
        INFO("fee %.2f ms, fps = %.2f", fee, iframe / fee * 1000);
    };

    //////////////////传统做法////////////////////////
    {
        cv::VideoCapture cap("exp/face_tracker.mp4");
        cv::Mat image;
        int iframe = 0;
        auto t0 = iLogger::timestamp_now_float();

        while(iframe < 300 && cap.read(image)){
            /** 模拟读取摄像头的延迟 **/
            iLogger::sleep(40);
            iframe++;

            /** 立即拿结果，时序图效果差，耗时5.7ms **/
            auto bboxes = yolo->commit(image).get();
            //for(auto& box : bboxes)
            //    cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), cv::Scalar(0, 255, 0), 2);

            if(iframe % 100 == 0)
                INFO("%d. %d objects", iframe++, bboxes.size());
        }

        auto fee = iLogger::timestamp_now_float() - t0;
        INFO("fee %.2f ms, fps = %.2f", fee, iframe / fee * 1000);
    };

    //////////////////优化做法////////////////////////
    {
        cv::VideoCapture cap("exp/face_tracker.mp4");
        shared_future<Yolo::BoxArray> prev_future;
        cv::Mat image;
        cv::Mat prev_image;
        int iframe = 0;
        auto t0 = iLogger::timestamp_now_float();

        while(iframe < 300 && cap.read(image)){
            /** 模拟读取摄像头的延迟 **/
            iLogger::sleep(40);
            iframe++;

            if(prev_future.valid()){
                auto bboxes = prev_future.get();
                //for(auto& box : bboxes)
                //    cv::rectangle(prev_image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), cv::Scalar(0, 255, 0), 2);

                if(iframe % 100 == 0)
                    INFO("%d. %d objects", iframe++, bboxes.size());
            }

            image.copyTo(prev_image);
            prev_future = yolo->commit(image);
        }

        auto fee = iLogger::timestamp_now_float() - t0;
        INFO("fee %.2f ms, fps = %.2f", fee, iframe / fee * 1000);
    };
}

int app_lesson(){

    iLogger::set_log_level(iLogger::LogLevel::Verbose);
    test_tensor3();
    // lesson1();
    // lesson2();
    // lesson3();
    // lesson_cache1frame();
    return 0;
}

