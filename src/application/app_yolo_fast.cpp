
#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include "app_yolo_fast/yolo_fast.hpp"

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

YoloFast::DecodeMeta type_of_meta(YoloFast::Type type){
    switch(type){
        case YoloFast::Type::V5_P5: return YoloFast::DecodeMeta::v5_p5_default_meta();
        case YoloFast::Type::V5_P6: return YoloFast::DecodeMeta::v5_p6_default_meta();
        case YoloFast::Type::X: return YoloFast::DecodeMeta::x_default_meta();
        default: return YoloFast::DecodeMeta::v5_p5_default_meta();
    }
}

static void append_to_file(const string& file, const string& data){
    FILE* f = fopen(file.c_str(), "a+");
    if(f == nullptr){
        INFOE("Open %s failed.", file.c_str());
        return;
    }

    fprintf(f, "%s\n", data.c_str());
    fclose(f);
}

static void inference_and_performance(int deviceid, const string& engine_file, TRT::Mode mode, YoloFast::Type type, const string& model_name){

    auto meta = type_of_meta(type);
    auto engine = YoloFast::create_infer(engine_file, type, deviceid, 0.4f, 0.5f, meta);
    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    auto files = iLogger::find_files("inference", "*.jpg;*.jpeg;*.png;*.gif;*.tif");
    vector<cv::Mat> images;
    for(int i = 0; i < files.size(); ++i){
        auto image = cv::imread(files[i]);
        images.emplace_back(image);
    }

    // warmup
    vector<shared_future<YoloFast::BoxArray>> boxes_array;
    for(int i = 0; i < 10; ++i)
        boxes_array = engine->commits(images);
    boxes_array.back().get();
    boxes_array.clear();
    
    /////////////////////////////////////////////////////////
    const int ntest = 100;
    auto begin_timer = iLogger::timestamp_now_float();

    for(int i  = 0; i < ntest; ++i)
        boxes_array = engine->commits(images);
    
    // wait all result
    boxes_array.back().get();

    float inference_average_time = (iLogger::timestamp_now_float() - begin_timer) / ntest / images.size();
    auto type_name = YoloFast::type_name(type);
    auto mode_name = TRT::mode_string(mode);
    INFO("%s[%s] average: %.2f ms / image, FPS: %.2f", engine_file.c_str(), type_name, inference_average_time, 1000 / inference_average_time);
    append_to_file("perf.result.log", iLogger::format("%s,%s,%s,%f", model_name.c_str(), type_name, mode_name, inference_average_time));

    string root = iLogger::format("%s_%s_%s_result", model_name.c_str(), type_name, mode_name);
    iLogger::rmtree(root);
    iLogger::mkdir(root);

    for(int i = 0; i < boxes_array.size(); ++i){

        auto& image = images[i];
        auto boxes  = boxes_array[i].get();
        
        for(auto& obj : boxes){
            uint8_t b, g, r;
            tie(b, g, r) = iLogger::random_color(obj.class_label);
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

static void test(YoloFast::Type type, TRT::Mode mode, const string& model){

    int deviceid = 0;
    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(deviceid);

    auto int8process = [=](int current, int count, const vector<string>& files, shared_ptr<TRT::Tensor>& tensor){

        INFO("Int8 %d / %d", current, count);

        for(int i = 0; i < files.size(); ++i){
            auto image = cv::imread(files[i]);
            YoloFast::image_to_tensor(image, tensor, type, i);
        }
    };

    const char* name = model.c_str();
    INFO("===================== test %s %s %s ==================================", YoloFast::type_name(type), mode_name, name);

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
            model_file,                 // save to
            {},
            int8process,
            "inference"
        );
    }

    inference_and_performance(deviceid, model_file, mode, type, name);
}

int app_yolo_fast(){

    test(YoloFast::Type::X, TRT::Mode::FP32, "yolox_s_fast");
    

    //iLogger::set_log_level(iLogger::LogLevel::Info);
    // test(YoloFast::Type::X, TRT::Mode::FP32, "yolox_x_fast");
    // test(YoloFast::Type::X, TRT::Mode::FP32, "yolox_l_fast");
    // test(YoloFast::Type::X, TRT::Mode::FP32, "yolox_m_fast");
    // test(YoloFast::Type::X, TRT::Mode::FP32, "yolox_s_fast");
    // test(YoloFast::Type::X, TRT::Mode::FP16, "yolox_x_fast");
    // test(YoloFast::Type::X, TRT::Mode::FP16, "yolox_l_fast");
    // test(YoloFast::Type::X, TRT::Mode::FP16, "yolox_m_fast");
    // test(YoloFast::Type::X, TRT::Mode::FP16, "yolox_s_fast");
    // test(YoloFast::Type::X, TRT::Mode::INT8, "yolox_x_fast");
    // test(YoloFast::Type::X, TRT::Mode::INT8, "yolox_l_fast");
    // test(YoloFast::Type::X, TRT::Mode::INT8, "yolox_m_fast");
    // test(YoloFast::Type::X, TRT::Mode::INT8, "yolox_s_fast");

    // test(YoloFast::Type::V5_P6, TRT::Mode::FP32, "yolov5x6_fast");
    // test(YoloFast::Type::V5_P6, TRT::Mode::FP32, "yolov5l6_fast");
    // test(YoloFast::Type::V5_P6, TRT::Mode::FP32, "yolov5m6_fast");
    // test(YoloFast::Type::V5_P6, TRT::Mode::FP32, "yolov5s6_fast");
    // test(YoloFast::Type::V5_P5, TRT::Mode::FP32, "yolov5x_fast");
    // test(YoloFast::Type::V5_P5, TRT::Mode::FP32, "yolov5l_fast");
    // test(YoloFast::Type::V5_P5, TRT::Mode::FP32, "yolov5m_fast");
    // test(YoloFast::Type::V5_P5, TRT::Mode::FP32, "yolov5s_fast");

    // test(YoloFast::Type::V5_P6, TRT::Mode::FP16, "yolov5x6_fast");
    // test(YoloFast::Type::V5_P6, TRT::Mode::FP16, "yolov5l6_fast");
    // test(YoloFast::Type::V5_P6, TRT::Mode::FP16, "yolov5m6_fast");
    // test(YoloFast::Type::V5_P6, TRT::Mode::FP16, "yolov5s6_fast");
    // test(YoloFast::Type::V5_P5, TRT::Mode::FP16, "yolov5x_fast");
    // test(YoloFast::Type::V5_P5, TRT::Mode::FP16, "yolov5l_fast");
    // test(YoloFast::Type::V5_P5, TRT::Mode::FP16, "yolov5m_fast");
    // test(YoloFast::Type::V5_P5, TRT::Mode::FP16, "yolov5s_fast");

    // test(YoloFast::Type::V5_P6, TRT::Mode::INT8, "yolov5x6_fast");
    // test(YoloFast::Type::V5_P6, TRT::Mode::INT8, "yolov5l6_fast");
    // test(YoloFast::Type::V5_P6, TRT::Mode::INT8, "yolov5m6_fast");
    // test(YoloFast::Type::V5_P6, TRT::Mode::INT8, "yolov5s6_fast");
    // test(YoloFast::Type::V5_P5, TRT::Mode::INT8, "yolov5x_fast");
    // test(YoloFast::Type::V5_P5, TRT::Mode::INT8, "yolov5l_fast");
    // test(YoloFast::Type::V5_P5, TRT::Mode::INT8, "yolov5m_fast");
    // test(YoloFast::Type::V5_P5, TRT::Mode::INT8, "yolov5s_fast");
    return 0;
}