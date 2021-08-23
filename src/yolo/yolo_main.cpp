
/**
 * @file _main.cpp
 * @author 手写AI (zifuture.com:8090)
 * @date 2021-07-26
 * 
 *   实现了基于TensorRT对yolox的推理工作 
 *   1. 基于FP32的模型编译、和推理执行
 *   2. 基于INT8的模型编译、和推理执行
 *   3. 自定义插件的实现，从pytorch导出到推理编译，并支持FP16
 * 
 *   我们是一群热血的个人组织者，力图发布免费高质量内容
 *   我们的博客地址：http://zifuture.com:8090
 *   我们的B站地址：https://space.bilibili.com/1413433465
 *   
 *   如果想要深入学习关于tensorRT的技术栈，请通过博客中的二维码联系我们（免费崔更即可）
 *   请关注B站，我们根据情况发布相关教程视频（免费）
 */

// 模型编译时使用的头文件
#include <builder/trt_builder.hpp>

// 模型推理时使用的头文件
#include <infer/trt_infer.hpp>

// 日志打印使用的文件
#include <common/ilogger.hpp>

// 封装的方式测试
#include "yolo.hpp"

using namespace std;


// 展示效果时使用的coco标签名称 
static const char* cocolabels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

bool requires(const char* name);

static void forward_engine(const string& engine_file, Yolo::Type type){

    auto engine = Yolo::create_infer(engine_file, type, 0, 0.4f);
    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    string root = iLogger::format("detect_result_%s", Yolo::type_name(type));
    iLogger::rmtree(root);
    iLogger::mkdir(root);

    auto files = iLogger::find_files("inference", "*.jpg;*.jpeg;*.png;*.gif;*.tif");
    for(int i = 0; i < files.size(); ++i){
        auto image = cv::imread(files[i]);

        auto t0    = iLogger::timestamp_now_float();
        auto boxes = engine->commit(image).get();
        float inference_time = iLogger::timestamp_now_float() - t0;

        // 框给画到图上
        for(auto& obj : boxes){

            // 使用根据类别计算的随机颜色填充
            uint8_t b, g, r;
            tie(r, g, b) = iLogger::random_color(obj.class_label);
            cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

            // 绘制类别名字
            auto name = cocolabels[obj.class_label];
            int width = cv::getTextSize(name, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(image, cv::Point(obj.left-3, obj.top-33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
            cv::putText(image, iLogger::format("%s", name), cv::Point(obj.left, obj.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
        }

        string file_name = iLogger::file_name(files[i], false);
        string save_path = iLogger::format("%s/%s.jpg", root.c_str(), file_name.c_str());
        INFO("Save to %s, %d object, %.2f ms", save_path.c_str(), boxes.size(), inference_time);
        cv::imwrite(save_path, image);
    }
}

static void forward_engine_dynamic_batch(const string& engine_file, Yolo::Type type){

    auto engine = Yolo::create_infer(engine_file, type, 0, 0.4f);
    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    string root = iLogger::format("detect_result_%s", Yolo::type_name(type));
    iLogger::rmtree(root);
    iLogger::mkdir(root);

    auto files = iLogger::find_files("inference", "*.jpg;*.jpeg;*.png;*.gif;*.tif");
    vector<cv::Mat> images;
    for(int i = 0; i < files.size(); ++i){
        auto image = cv::imread(files[i]);
        images.emplace_back(image);
    }

    // warmup
    engine->commit(images[0]).get();
    
    auto t0     = iLogger::timestamp_now_float();
    auto boxes_array = engine->commits(images);
    
    // wait batch result
    boxes_array[0].get();

    float inference_average_time = (iLogger::timestamp_now_float() - t0) / boxes_array.size();
    for(int i = 0; i < boxes_array.size(); ++i){

        auto& image = images[i];
        auto boxes  = boxes_array[i].get();
        
        // 框给画到图上
        for(auto& obj : boxes){

            // 使用根据类别计算的随机颜色填充
            uint8_t b, g, r;
            tie(r, g, b) = iLogger::random_color(obj.class_label);
            cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

            // 绘制类别名字
            auto name = cocolabels[obj.class_label];
            int width = cv::getTextSize(name, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(image, cv::Point(obj.left-3, obj.top-33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
            cv::putText(image, iLogger::format("%s", name), cv::Point(obj.left, obj.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
        }

        string file_name = iLogger::file_name(files[i], false);
        string save_path = iLogger::format("%s/%s.jpg", root.c_str(), file_name.c_str());
        INFO("Save to %s, %d object, average time %.2f ms", save_path.c_str(), boxes.size(), inference_average_time);
        cv::imwrite(save_path, image);
    }
}

static void test_plugin(){

    // 通过以下代码即可生成plugin.onnx
    // cd workspace
    // python test_plugin.py
    TRT::set_device(0);

    // plugin.onnx是通过test_plugin.py生成的
    TRT::compile(
        TRT::TRTMode_FP32, {}, 3, "plugin.onnx", "plugin.fp32.trtmodel", {}, false
    );
 
    auto engine = TRT::load_infer("plugin.fp32.trtmodel");
    engine->print();

    auto input0 = engine->input(0);
    auto input1 = engine->input(1);
    auto output = engine->output(0);

    // 推理使用到了插件
    INFO("input0: %s", input0->shape_string());
    INFO("input1: %s", input1->shape_string());
    INFO("output: %s", output->shape_string());
    input0->set_to(0.80);

    // output = input1 + hswish(input0)
    // output = 0 + hswish(0.8)
    // output = 0 + 0.8 * relu6(0.8 + 3) / 6
    // output = 0.8 * 3.8f / 6
    float output_real = 0.8f * 3.8f / 6.0f;
    engine->forward(true);
    INFO("output %f, output_real = %f", output->at<float>(0), output_real);
}

static void test_int8(Yolo::Type type){

    INFO("===================== test %s int8 ==================================", Yolo::type_name(type));
    auto int8process = [](int current, int count, vector<string>& images, shared_ptr<TRT::Tensor>& tensor){

        INFO("Int8 %d / %d", current, count);

        // 按道理，这里的输入，应该按照推理的方式进行。这里我简单模拟了resize的方式输入
        for(int i = 0; i < images.size(); ++i){
            auto image = cv::imread(images[i]);
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
            cv::resize(image, image, cv::Size(tensor->size(3), tensor->size(2)));
            float mean[] = {0.485, 0.456, 0.406};
            float std[]  = {0.229, 0.224, 0.225};
            tensor->set_norm_mat(i, image, mean, std);
        }
    };

    const char* name = nullptr;
    if(type == Yolo::Type::V5){
        name = "yolov5m";
    }else if(type == Yolo::Type::X){
        name = "yolox_m";
    }

    if(not requires(name))
        return;

    string onnx_file = iLogger::format("%s.onnx", name);
    string model_file = iLogger::format("%s.int8.trtmodel", name);
    int test_batch_size = 1;  // 当你需要修改batch大于1时，请查看yolox.cpp:260行备注

    if(not iLogger::exists(model_file)){
        TRT::compile(
            TRT::TRTMode_INT8,   // 编译方式有，FP32、FP16、INT8
            {},                         // onnx时无效，caffe的输出节点标记
            test_batch_size,            // 指定编译的batch size
            onnx_file,                  // 需要编译的onnx文件
            model_file,                 // 储存的模型文件
            {},                         // 指定需要重定义的输入shape，这里可以对onnx的输入shape进行重定义
            false,                      // 是否采用动态batch维度，true采用，false不采用，使用静态固定的batch size
            int8process,                // int8标定时的数据输入处理函数
            "inference"                 // 图像数据的路径，在当前路径下找到用以标定的图像，图像可以随意给，不需要标注
        );
    }

    forward_engine(model_file, type);
}

static void test_fp32(Yolo::Type type){

    TRT::set_device(0);
    INFO("===================== test %s fp32 ==================================", Yolo::type_name(type));

    const char* name = nullptr;
    if(type == Yolo::Type::V5){
        name = "yolov5m";
    }else if(type == Yolo::Type::X){
        name = "yolox_m";
    }

    if(not requires(name))
        return;

    string onnx_file = iLogger::format("%s.onnx", name);
    string model_file = iLogger::format("%s.fp32.trtmodel", name);
    int test_batch_size = 5;  // 当你需要修改batch大于1时，请查看yolox.cpp:260行备注
    
    // 动态batch和静态batch，如果你想要弄清楚，请打开http://www.zifuture.com:8090/
    // 找到右边的二维码，扫码加好友后进群交流（免费哈，就是技术人员一起沟通）
    if(not iLogger::exists(model_file)){
        TRT::compile(
            TRT::TRTMode_FP32,   // 编译方式有，FP32、FP16、INT8
            {},                         // onnx时无效，caffe的输出节点标记
            test_batch_size,            // 指定编译的batch size
            onnx_file,                  // 需要编译的onnx文件
            model_file,                 // 储存的模型文件
            {},                         // 指定需要重定义的输入shape，这里可以对onnx的输入shape进行重定义
            false                       // 是否采用动态batch维度，true采用，false不采用，使用静态固定的batch size
        );
    }

    forward_engine(model_file, type);
}

static void test_dynamic_batch(Yolo::Type type){

    TRT::set_device(0);
    INFO("===================== test %s dynamic batch fp32 ==================================", Yolo::type_name(type));

    const char* name = nullptr;
    if(type == Yolo::Type::V5){
        name = "yolov5m";
    }else if(type == Yolo::Type::X){
        name = "yolox_m";
    }

    if(not requires(name))
        return;

    string onnx_file = iLogger::format("%s.onnx", name);
    string model_file = iLogger::format("%s.dynamic.batch.fp32.trtmodel", name);
    int test_batch_size = 5;
    
    // 动态batch和静态batch，如果你想要弄清楚，请打开http://www.zifuture.com:8090/
    // 找到右边的二维码，扫码加好友后进群交流（免费哈，就是技术人员一起沟通）
    if(not iLogger::exists(model_file)){
        TRT::compile(
            TRT::TRTMode_FP32,   // 编译方式有，FP32、FP16、INT8
            {},                         // onnx时无效，caffe的输出节点标记
            test_batch_size,            // 指定编译的batch size
            onnx_file,                  // 需要编译的onnx文件
            model_file,                 // 储存的模型文件
            {},                         // 指定需要重定义的输入shape，这里可以对onnx的输入shape进行重定义
            true                        // 是否采用动态batch维度，true采用，false不采用，使用静态固定的batch size
        );
    }

    forward_engine_dynamic_batch(model_file, type);
}

int yolo_main(){

    test_dynamic_batch(Yolo::Type::V5);
    test_dynamic_batch(Yolo::Type::X);
    // test_plugin();
    // test_int8(Yolo::Type::X);
    return 0;
}