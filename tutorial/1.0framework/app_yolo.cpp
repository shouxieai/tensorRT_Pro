
/**
 * @file _main.cpp
 * @author 手写AI shouxieai (zifuture.com:8090)
 * @date 2021-09-17
 * 
 * 中文版注释请下滑(建议优先阅读英文版)
 * ################################################# ENGLISH VERSION ############################################################
 * This script of yolo version contains a lot of comments for the purpose of learning, especially for those who
 * are interesed in high performance deployment but are still not familiar with c++ and cuda programmming.Therefore, 
 * the comments sometimes might be a little bit verbose.
 * 
 * The comments are generally demonstrated in code sample, drawing or any other types that is vivid. The common
 * symbol used is listed as follow:
 *  - //* kp : the knowledge point involved in the code snippest  
 *  - //* ref : the reference for deeper understanding
 *  - //* Overview: an overview of a file or a section of code etc.
 *  - //todo means the part will be detailed in the comming days 
 *  - @xxx.hpp or @xxx.cpp indicates the file in the discussion
 * 
 *
 * Afterword
 *   We're enthusastic self-organized AI coders, striving to release high quality AI tutorial and toolbox.
 *   Our blog：http://zifuture.com:8090 
 *   Bilibli：https://space.bilibili.com/1413433465
 *   Youtube: (comming soon)
 * 
 *   Now the resource is only Chinese-version-available. English version is comming soon.
 *   
 *   Welcome to contact us through the QR code in github if you want to deep dive into the techonology stack.
 *   Please subscibe our blibli and youtube.
 *   Just be bossy and push us to release more for free but don't forget to give us a thumb!!!!
 */


/* 
* Overview:
In this framework, for any model(e.g. Yolo), we usually follow this workflow:
(Tips: use full-text search for the following key words in order to get a general idea of what is going on)

Basic: (usually files in src/application)
- build our model by TRT::compile
- create our infer module by Yolo::create_infer
- commit the input in a std::vector form
- wait for the result and draw the bboxes

Advanced: (ususally files in src/tensorRT)
- trt_builder
- trt_infer
- iLogger
- common etc

Tips: if you use vscode, it is recommended to install Better Comments extensions to see the more vivid
comments.

For beginner:
More attention should be paid to learn the above basic parts to take advantage of the high performance framework. Just see other
code from the advanced part as some useful API. Details can be neglected for now.
For veteran:
Explore the framework as possible as you could.
 
Now go to the yolo.cpp and read the overview of it. 

################################################# CHINESE VERSION ############################################################
这个yolo的脚本包含了很多以为学习目的的注释，特别是那些对高性能部署感兴趣，但仍不熟悉c++和cuda编程的小伙伴。因此,注释有时可能有点冗长。
注释通常以代码示例、绘图或任何其他生动的类型来表现。
常见使用的符号如下所示:
- //* kp:代码片段中涉及的知识点
- //* ref:有助于更深层次理解的参考资料（可能需要梯子翻出去）
- //* Overview:一个文件或一段代码的概述等。
- //todo表示该部分将在未来进行详细说明
- @xxx.hpp或@xxx.cpp表示正在讨论的文件

*Overview:
在这个框架中，对于任何模型(例如 Yolo)，我们通常遵循以下工作流:
(提示:使用全文搜索下面的关键词，以便对正在进行的事情有一个大致的了解)

基础部分 (通常是src/application中的文件)
- 通过TRT::compile构建我们的模型
- 通过Yolo::create_infer创建infer模块
- 以std::vector形式提交输入
- 等待结果并绘制框
    
高级部分 (通常是src/tensorRT中的文件)
- trt_builder
- trt_infer
- iLogger
- common 等等

Tips:如果你使用vscode，建议安装Better Comments扩展以看到更生动的注释

对于初学者:
    为了更好地利用高性能框架，应该更多地注意学习上述基本部分。先暂时把高级部分的代码作为一些有用的API。细节现在可以忽略不计。

对于经验丰富的开发者:
    尽可能地探索这个框架！

现在去yolo.cpp阅读它的概述吧


 */





#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include "app_yolo/yolo.hpp"

using namespace std;

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
        images.emplace_back(image); //* kp: emplace_back. Save the extra copy or move operation required when using push_back.refer to https://en.cppreference.com/w/cpp/container/vector/emplace_back
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
        
        for(auto& obj : boxes){

            uint8_t b, g, r;
            tie(r, g, b) = iLogger::random_color(obj.class_label);
            cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

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

static void test_int8(Yolo::Type type, const string& model){

    const char* name = model.c_str();
    INFO("===================== test %s int8 %s ==================================", Yolo::type_name(type), name);
    auto int8process = [=](int current, int count, const vector<string>& files, shared_ptr<TRT::Tensor>& tensor){

        INFO("Int8 %d / %d", current, count);

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
    int test_batch_size = 16; 

    if(not iLogger::exists(model_file)){
        TRT::compile(
            TRT::Mode::INT8,            // FP32、FP16、INT8
            test_batch_size,            // max batch size
            onnx_file,                  // source
            model_file,                 // save to
            {},                         // reset input dims
            int8process,                // int8 function
            "inference"                 // int8 image directory
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
    int test_batch_size = 16;
    
    if(not iLogger::exists(model_file)){
        TRT::compile(
            mode,                       // FP32、FP16、INT8
            test_batch_size,            // max batch size
            onnx_file,                  // source 
            model_file                  // save to
        );
    }

    forward_engine(model_file, type);
}

int app_yolo(){

    //iLogger::set_log_level(iLogger::LogLevel::Info);
    test(Yolo::Type::X, TRT::Mode::FP32, "yolox_l");
    //test(Yolo::Type::V5, TRT::Mode::FP32, "yolov5s");
    // test(Yolo::Type::X, TRT::Mode::FP16, "yolox_s");
    // test(Yolo::Type::V5, TRT::Mode::FP16, "yolov5s");
    // test_int8(Yolo::Type::X, "yolox_s");
    // test_int8(Yolo::Type::V5, "yolov5s");
    return 0;
}