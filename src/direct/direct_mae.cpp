
/**
 * onnx导出项目在这里：
 * https://github.com/shouxieai/MAE-pytorch
 *  实验mae的特征提取功能
 * Masked Autoencoders Are Scalable Vision Learners
 * 
 * onnx下载：
 * 链接：https://pan.baidu.com/s/1r0e82KQj99ue7sNBawNvUQ 
 * 提取码：sxai 
 */

#include <infer/trt_infer.hpp>
#include <builder/trt_builder.hpp>
#include <common/ilogger.hpp>

int direct_mae(){

    TRT::set_device(0);
    TRT::compile(
        TRT::Mode::FP32,
        1,
        "mae.onnx", "mae.trtmodel"
    );
    INFO("Done");

    auto engine = TRT::load_infer("mae.trtmodel");
    auto image = cv::imread("test.jpg");
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    float mean[] = {0.485, 0.456, 0.406};
    float std[]  = {0.229, 0.224, 0.225};
    engine->input()->set_norm_mat(0, image, mean, std);
    engine->forward();

    std::cout << engine->output()->shape_string() << std::endl; 
    engine->output()->save_to_file("test.binary");
    return 0;
}