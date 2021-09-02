
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
#include "app_yolo/yolo.hpp"

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

    string root = iLogger::format("%s_result", Yolo::type_name(type));
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
    boxes_array.back().get();

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
            auto name    = cocolabels[obj.class_label];
            auto caption = iLogger::format("%s %.2f", name, obj.confidence);
            int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(image, cv::Point(obj.left-3, obj.top-33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
            cv::putText(image, caption, cv::Point(obj.left, obj.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
        }

        string file_name = iLogger::file_name(files[i], false);
        string save_path = iLogger::format("%s/%s.jpg", root.c_str(), file_name.c_str());
        INFO("Save to %s, %d object, average time %.2f ms", save_path.c_str(), boxes.size(), inference_average_time);
        cv::imwrite(save_path, image);
    }
}

static void test_plugin(TRT::Mode mode){

    // 通过以下代码即可生成plugin.onnx
    // cd workspace
    // python test_plugin.py
    iLogger::set_log_level(iLogger::LogLevel::Verbose);
    TRT::set_device(0);

    auto mode_name = TRT::mode_string(mode);
    auto engine_name = iLogger::format("plugin.%s.trtmodel", mode_name);
    // plugin.onnx是通过test_plugin.py生成的
    TRT::compile(
        mode, 3, "plugin.onnx", engine_name, {}
    );
 
    auto engine = TRT::load_infer(engine_name);
    engine->print();

    auto input0 = engine->input(0);
    auto input1 = engine->input(1);
    auto output = engine->output(0);

    // 推理使用到了插件
    INFO("input0: %s", input0->shape_string());
    INFO("input1: %s", input1->shape_string());
    INFO("output: %s", output->shape_string());
    
    float input0_val = 0.8;
    float input1_val = 2;
    input0->set_to(input0_val);
    input1->set_to(input1_val);

    auto hswish = [](float x){float a = x + 3; a=a<0?0:(a>=6?6:a); return x * a / 6;};
    auto sigmoid = [](float x){return 1 / (1 + exp(-x));};
    auto relu   = [](float x){return max(0.0f, x);};
    float output_real = relu(hswish(input0_val) * input1_val);
    engine->forward(true);

    INFO("output %f, output_real = %f", output->at<float>(0, 0), output_real);
}

static void test_int8(Yolo::Type type, const string& model){

    const char* name = model.c_str();
    INFO("===================== test %s int8 %s ==================================", Yolo::type_name(type), name);
    auto int8process = [=](int current, int count, const vector<string>& files, shared_ptr<TRT::Tensor>& tensor){

        INFO("Int8 %d / %d", current, count);

        // 按道理，这里的输入，应该按照推理的方式进行。这里我简单模拟了resize的方式输入
        for(int i = 0; i < files.size(); ++i){
            auto image = cv::imread(files[i]);

            if(type == Yolo::Type::V5){
                cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
                cv::resize(image, image, cv::Size(tensor->size(3), tensor->size(2)));
                image.convertTo(image, CV_32F, 1 / 255.0f);
                tensor->set_mat(i, image);
            }else if(type == Yolo::Type::X){
                cv::resize(image, image, cv::Size(tensor->size(3), tensor->size(2)));
                image.convertTo(image, CV_32F);
                tensor->set_mat(i, image);
            }
        }
    };

    if(not requires(name))
        return;

    string onnx_file = iLogger::format("%s.onnx", name);
    string model_file = iLogger::format("%s.int8.trtmodel", name);
    int test_batch_size = 6;  // 当你需要修改batch大于1时，请查看yolox.cpp:260行备注

    if(not iLogger::exists(model_file)){
        TRT::compile(
            TRT::Mode::INT8,   // 编译方式有，FP32、FP16、INT8
            test_batch_size,            // 指定编译的batch size
            onnx_file,                  // 需要编译的onnx文件
            model_file,                 // 储存的模型文件
            {},                         // 指定需要重定义的输入shape，这里可以对onnx的输入shape进行重定义
            int8process,                // int8标定时的数据输入处理函数
            "inference"                 // 图像数据的路径，在当前路径下找到用以标定的图像，图像可以随意给，不需要标注
        );
    }

    forward_engine(model_file, type);
}

static void test(Yolo::Type type, TRT::Mode mode, const string& model){

    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(0);

    const char* name = model.c_str();
    INFO("===================== test %s %s %s ==================================", Yolo::type_name(type), mode_name, name);

    if(not requires(name))
        return;

    string onnx_file = iLogger::format("%s.onnx", name);
    string model_file = iLogger::format("%s.%s.trtmodel", name, mode_name);
    int test_batch_size = 6;
    
    // 动态batch和静态batch，如果你想要弄清楚，请打开http://www.zifuture.com:8090/
    // 找到右边的二维码，扫码加好友后进群交流（免费哈，就是技术人员一起沟通）
    if(not iLogger::exists(model_file)){
        TRT::compile(
            mode,                       // 编译方式有，FP32、FP16、INT8
            test_batch_size,            // 指定编译的batch size
            onnx_file,                  // 需要编译的onnx文件
            model_file                  // 储存的模型文件
        );
    }

    forward_engine(model_file, type);
}

int app_yolo(){

    //iLogger::set_log_level(iLogger::LogLevel::Info);
    test(Yolo::Type::X, TRT::Mode::FP32, "yolox_s");
    test(Yolo::Type::V5, TRT::Mode::FP32, "yolov5s");
    test(Yolo::Type::X, TRT::Mode::FP16, "yolox_s");
    test(Yolo::Type::V5, TRT::Mode::FP16, "yolov5s");
    test_int8(Yolo::Type::X, "yolox_s");
    test_int8(Yolo::Type::V5, "yolov5s");
    // test_plugin(TRT::Mode::FP32);
    // test_plugin(TRT::Mode::FP16);
    return 0;
}