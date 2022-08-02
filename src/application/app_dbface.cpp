//dbface git: https://github.com/dlunion/DBFace
#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include "app_dbface/dbface.hpp"

using namespace std;

bool requires(const char* name);

static void forward_engine(const string& engine_file){

    auto engine = DBFace::create_infer(engine_file, 0, 0.35f);
    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    string root = "DBFace_result";
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
        
        for(auto& obj : boxes){

            uint8_t b, g, r;
            tie(b, g, r) = iLogger::random_color(0);
            cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

            auto caption = iLogger::format("%.2f", obj.confidence);
            int width    = cv::getTextSize(caption, 0, 1, 1, nullptr).width;
            cv::rectangle(image, cv::Point(obj.left-3, obj.top-33), cv::Point(obj.left + width - 5, obj.top), cv::Scalar(b, g, r), -1);

            for(int k = 0; k < 5; ++k){
                float x = obj.landmark[k * 2 + 0];
                float y = obj.landmark[k * 2 + 1];
                cv::circle(image, cv::Point(x, y), 3, cv::Scalar(0, 255, 0), -1, 16);
            }
            cv::putText(image, caption, cv::Point(obj.left, obj.top-5), 0, 0.8, cv::Scalar::all(0), 2, 16);
        }

        string file_name = iLogger::file_name(files[i], false);
        string save_path = iLogger::format("%s/%s.jpg", root.c_str(), file_name.c_str());
        INFO("Save to %s, %d object, average time %.2f ms", save_path.c_str(), boxes.size(), inference_average_time);
        cv::imwrite(save_path, image);
    }
}


static void test(TRT::Mode mode, const string& model){

    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(0);

    const char* name = model.c_str();
    INFO("===================== test %s %s ==================================", mode_name, name);

    if(not requires(name))
        return;

    string onnx_file = iLogger::format("%s.onnx", name);
    string model_file = iLogger::format("%s.%s.trtmodel", name, mode_name);
    int test_batch_size = 16;
    
    if(not iLogger::exists(model_file)){
        TRT::compile(
            mode,                       // FP32
            test_batch_size,            // max batch size
            onnx_file,                  // source 
            model_file,                  // save to
            {{1, 3, 800, 800}}
        );
    }

    forward_engine(model_file);
}

int app_dbface(){
    
    //test(TRT::Mode::FP32, "dbface_cvt"); // Type indicates the task. name string indicates the model size and structure.
    test(TRT::Mode::FP32, "dbfaceSmallH"); // DBFaceSmallH, FP32
    //test(TRT::Mode::FP16, "dbfaceSmallH");   // DBFaceSmallH, FP16
    return 0;
}
