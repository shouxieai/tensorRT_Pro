
/**
 * onnx导出项目在这里：
 * https://github.com/shouxieai/tensorrt-pro-sample-python-classifier
 */

#include <infer/trt_infer.hpp>
#include <builder/trt_builder.hpp>
#include <common/ilogger.hpp>

int direct_classifier(){

    if(!iLogger::exists("classifier.onnx")){
        INFOE("classifier.onnx not found, reference: https://github.com/shouxieai/tensorrt-pro-sample-python-classifier");
        return -1;
    }

    TRT::set_device(0);
    if(!iLogger::exists("classifier.trtmodel")){
        TRT::compile(
            TRT::Mode::FP32,
            1,
            "classifier.onnx", 
            "classifier.trtmodel"
        );
        INFO("Compile done");
    }

    auto engine = TRT::load_infer("classifier.trtmodel");
    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return -1;
    }
    
    auto image = cv::imread("dog.jpg");
    float mean[] = {0.485, 0.456, 0.406};
    float std[]  = {0.229, 0.224, 0.225};
    engine->input()->set_norm_mat(0, image, mean, std);
    engine->forward();

    float* prob       = engine->output()->cpu<float>();
    int num_classes   = engine->output()->channel();
    int predict_label = std::max_element(prob, prob + num_classes) - prob;
    auto labels       = iLogger::split_string(iLogger::load_text_file("labels.imagenet.txt"), "\n");
    auto predict_name = labels[predict_label];
    float confidence  = prob[predict_label];

    INFO("Predict: %s, confidence = %f, label = %d", predict_name.c_str(), confidence, predict_label);
    return 0;
}