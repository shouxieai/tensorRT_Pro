
#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include "app_yolo/yolo.hpp"
#include "app_yolo/multi_gpu.hpp"

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

static void append_to_file(const string& file, const string& data){
    FILE* f = fopen(file.c_str(), "a+");
    if(f == nullptr){
        INFOE("Open %s failed.", file.c_str());
        return;
    }

    fprintf(f, "%s\n", data.c_str());
    fclose(f);
}

static void inference_and_performance(int deviceid, const string& engine_file, TRT::Mode mode, Yolo::Type type, const string& model_name){

    auto engine = Yolo::create_infer(
        engine_file,                // engine file
        type,                       // yolo type, Yolo::Type::V5 / Yolo::Type::X
        deviceid,                   // gpu id
        0.25f,                      // confidence threshold
        0.45f,                      // nms threshold
        Yolo::NMSMethod::FastGPU,   // NMS method, fast GPU / CPU
        1024,                       // max objects
        false                       // preprocess use multi stream
    );
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
    vector<shared_future<Yolo::BoxArray>> boxes_array;
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
    auto type_name = Yolo::type_name(type);
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
    engine.reset();
}

static void test(Yolo::Type type, TRT::Mode mode, const string& model){

    int deviceid = 0;
    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(deviceid);

    auto int8process = [=](int current, int count, const vector<string>& files, shared_ptr<TRT::Tensor>& tensor){

        INFO("Int8 %d / %d", current, count);

        for(int i = 0; i < files.size(); ++i){
            auto image = cv::imread(files[i]);
            Yolo::image_to_tensor(image, tensor, type, i);
        }
    };

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
            model_file,                 // save to
            {},
            int8process,
            "inference"
        );
    }

    inference_and_performance(deviceid, model_file, mode, type, name);
}

void multi_gpu_test(){
    
    vector<int> devices{0, 1, 2};
    auto multi_gpu_infer = Yolo::create_multi_gpu_infer(
        "yolov5s-6.0.FP32.trtmodel", Yolo::Type::V5, devices
    );

    auto files = iLogger::find_files("inference", "*.jpg");
    #pragma omp parallel for num_threads(devices.size())
    for(int i = 0; i < devices.size(); ++i){

        auto image = cv::imread(files[i]);
        for(int j = 0; j < 1000; ++j){
            multi_gpu_infer->commit(image).get();
        }
    }
    INFO("Done");
}

int app_yolo(){
 
    test(Yolo::Type::V7, TRT::Mode::FP32, "yolov7");
    //test(Yolo::Type::V5, TRT::Mode::FP32, "yolov5s");
    //test(Yolo::Type::V3, TRT::Mode::FP32, "yolov3");

    // multi_gpu_test();
    //iLogger::set_log_level(iLogger::LogLevel::Debug);
    //test(Yolo::Type::X, TRT::Mode::FP32, "yolox_s");

    //iLogger::set_log_level(iLogger::LogLevel::Info);
    // test(Yolo::Type::X, TRT::Mode::FP32, "yolox_x");
    // test(Yolo::Type::X, TRT::Mode::FP32, "yolox_l");
    // test(Yolo::Type::X, TRT::Mode::FP32, "yolox_m");
    // test(Yolo::Type::X, TRT::Mode::FP32, "yolox_s");
    // test(Yolo::Type::X, TRT::Mode::FP16, "yolox_x");
    // test(Yolo::Type::X, TRT::Mode::FP16, "yolox_l");
    // test(Yolo::Type::X, TRT::Mode::FP16, "yolox_m");
    // test(Yolo::Type::X, TRT::Mode::FP16, "yolox_s");
    // test(Yolo::Type::X, TRT::Mode::INT8, "yolox_x");
    // test(Yolo::Type::X, TRT::Mode::INT8, "yolox_l");
    // test(Yolo::Type::X, TRT::Mode::INT8, "yolox_m");
    // test(Yolo::Type::X, TRT::Mode::INT8, "yolox_s");

    // test(Yolo::Type::V5, TRT::Mode::FP32, "yolov5x6");
    // test(Yolo::Type::V5, TRT::Mode::FP32, "yolov5l6");
    // test(Yolo::Type::V5, TRT::Mode::FP32, "yolov5m6");
    // test(Yolo::Type::V5, TRT::Mode::FP32, "yolov5s6");
    // test(Yolo::Type::V5, TRT::Mode::FP32, "yolov5x");
    // test(Yolo::Type::V5, TRT::Mode::FP32, "yolov5l");
    // test(Yolo::Type::V5, TRT::Mode::FP32, "yolov5m");

    // test(Yolo::Type::V5, TRT::Mode::FP16, "yolov5x6");
    // test(Yolo::Type::V5, TRT::Mode::FP16, "yolov5l6");
    // test(Yolo::Type::V5, TRT::Mode::FP16, "yolov5m6");
    // test(Yolo::Type::V5, TRT::Mode::FP16, "yolov5s6");
    // test(Yolo::Type::V5, TRT::Mode::FP16, "yolov5x");
    // test(Yolo::Type::V5, TRT::Mode::FP16, "yolov5l");
    // test(Yolo::Type::V5, TRT::Mode::FP16, "yolov5m");
    //test(Yolo::Type::V5, TRT::Mode::FP32, "yolov5s");
    // test(Yolo::Type::V5, TRT::Mode::INT8, "yolov5x6");
    // test(Yolo::Type::V5, TRT::Mode::INT8, "yolov5l6");
    // test(Yolo::Type::V5, TRT::Mode::INT8, "yolov5m6");
    // test(Yolo::Type::V5, TRT::Mode::INT8, "yolov5s6");
    // test(Yolo::Type::V5, TRT::Mode::INT8, "yolov5x");
    // test(Yolo::Type::V5, TRT::Mode::INT8, "yolov5l");
    // test(Yolo::Type::V5, TRT::Mode::INT8, "yolov5m");
    // test(Yolo::Type::V5, TRT::Mode::INT8, "yolov5s");
    return 0;
}