

#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include "app_scrfd/scrfd.hpp"

using namespace std;
using namespace cv;

bool requires(const char* name);

bool compile_scrfd(int input_width, int input_height, string& out_model_file, TRT::Mode mode = TRT::Mode::FP32){

    const char* name = "scrfd_2.5g_bnkps";
    if(not requires(name))
        return false;

    string onnx_file    = iLogger::format("%s.onnx", name);
    string model_file   = iLogger::format("%s.%dx%d.%s.trtmodel", name, input_width, input_height, TRT::mode_string(mode));
    int test_batch_size = 6;
    out_model_file      = model_file;
    
    if(iLogger::exists(model_file))
        return true;

    input_width  = iLogger::upbound(input_width);
    input_height = iLogger::upbound(input_height);
    TRT::set_layer_hook_reshape([&](const string& name, const std::vector<int64_t>& shape){
        
        INFOV("%s, %s", name.c_str(), iLogger::join_dims(shape).c_str());
        vector<string> layerset{
            "Reshape_108", "Reshape_110", "Reshape_112", 
            "Reshape_126", "Reshape_128", "Reshape_130", 
            "Reshape_144", "Reshape_146", "Reshape_148"
        };
        vector<int> strides{8, 8, 8, 16, 16, 16, 32, 32, 32};
        auto layer_iter = std::find_if(layerset.begin(),layerset.end(), [&](const string& item){return item==name;});
        if(layer_iter  != layerset.end()){
            int pos     = layer_iter - layerset.begin();
            int stride  = strides[pos];
            return vector<int64_t>{-1, input_height * input_width / stride / stride * 2, shape[2]};
        }
        return shape;
    });

    return TRT::compile(
        TRT::Mode::FP32,            // FP32、FP16、INT8
        test_batch_size,            // max batch size
        onnx_file,                  // source
        model_file,                 // save to
        {TRT::InputDims({1, 3, input_height, input_width})}
    );
}

static void scrfd_performance(shared_ptr<Scrfd::Infer> infer){
    
    auto files = iLogger::find_files("inference", "*.jpg;*.jpeg;*.png;*.gif;*.tif");
    vector<Mat> images;

    for(int i = 0; i < files.size(); ++i)
        images.emplace_back(imread(files[i]));

    int ntest  = 1000;
    INFO("Do scrfd performance test..., %d testing, %d images, take a moment...", ntest, images.size());

    // warmup
    infer->commits(images).back().get();

    auto begin = iLogger::timestamp_now_float();
    for(int i = 0; i < ntest; ++i){
        infer->commits(images).back().get();
    }

    auto time_fee_per_image = (iLogger::timestamp_now_float() - begin) / ntest / images.size();
    INFO("***********************************************************************************************");
    INFO("Retinaface limit case performance: %.3f ms / image, fps = %.2f", time_fee_per_image, 1000 / time_fee_per_image);
    INFO("***********************************************************************************************");
}

int app_scrfd(){

    TRT::set_device(0);
    INFO("===================== test scrfd FP32 ==================================");

    string model_file;
    if(!compile_scrfd(640, 640, model_file))
        return 0;

    auto engine = Scrfd::create_infer(model_file, 0, 0.7);
    scrfd_performance(engine);

    auto save_root = "scrfd_result";
    iLogger::rmtree(save_root);
    iLogger::mkdirs(save_root);
    auto files = iLogger::find_files("inference", "*.jpg;*.jpeg;*.png;*.gif;*.tif");
    vector<Mat> images;

    for(int i = 0; i < files.size(); ++i)
        images.emplace_back(imread(files[i]));

    // warmup
    engine->commits(images).back().get();
    
    auto time_begin   = iLogger::timestamp_now_float();
    auto images_faces = engine->commits(images);

    images_faces.back().get();
    float average_time = (iLogger::timestamp_now_float() - time_begin) / files.size();
    for(int i = 0; i < images.size(); ++i){
        
        auto& image = images[i];
        auto faces  = images_faces[i].get();
        for(int i = 0; i < faces.size(); ++i){
            auto& face = faces[i];
            rectangle(image, cv::Point(face.left, face.top), cv::Point(face.right, face.bottom), Scalar(0, 255, 0), 2);

            for(int j = 0; j < 5; ++j)
                circle(image, Point(face.landmark[j*2+0], face.landmark[j*2+1]), 3, Scalar(0, 255, 0), -1, 16);

            putText(image, iLogger::format("%.3f", face.confidence), cv::Point(face.left, face.top), 0, 1, Scalar(0, 255, 0), 1, 16);
        }

        auto save_file = iLogger::format("%s/%s", save_root, iLogger::file_name(files[i]).c_str());

        INFO("Save to %s, %d faces, average time: %.2f ms", save_file.c_str(), faces.size(), average_time);
        imwrite(save_file, image);
    }
    INFO("Done");
    return 0;
}