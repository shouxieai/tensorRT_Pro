
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
 *   预处理、后处理采用CPU实现（若想GPU可以自行实现）
 *   一次推理5张图获取结果
 * 
 *   我们是一群热血的个人组织者，力图发布免费高质量内容
 *   我们的博客地址：http://zifuture.com:8090
 *   我们的B站地址：https://space.bilibili.com/1413433465
 *   
 *   如果想要深入学习关于tensorRT的技术栈，请通过博客中的二维码联系我们（免费崔更即可）
 *   请关注B站，我们根据情况发布相关教程视频（免费）
 */

#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include "app_alphapose/alpha_pose.hpp"

using namespace std;
using namespace cv;

bool requires(const char* name);

int app_alphapose(){

    TRT::set_device(0);
    INFO("===================== test alphapose fp32 ==================================");

    const char* name = "alpha-pose-136";
    if(not requires(name))
        return 0;

    string onnx_file = iLogger::format("%s.onnx", name);
    string model_file = iLogger::format("%s.FP32.trtmodel", name);
    int test_batch_size = 16;  
    
    if(!iLogger::exists(model_file)){
        TRT::compile(
            TRT::Mode::FP32,            // FP32、FP16、INT8
            test_batch_size,            // max_batch_size
            onnx_file,                  // source
            model_file                  // save to
        );
    }
   
    Mat image = imread("inference/gril.jpg");
    auto engine = AlphaPose::create_infer(model_file, 0);
    auto box = Rect(158, 104, 176, 693);
    auto keys = engine->commit(make_tuple(image, box)).get();
    for(int i = 0; i < keys.size(); ++i){
        float x = keys[i].x;
        float y = keys[i].y;
        if(keys[i].z > 0.05){
            cv::circle(image, Point(x, y), 1, Scalar(0, 255, 0), -1, 16);
        }
    }

    auto save_file = "pose.show.jpg";
    INFO("Save to %s", save_file);
    
    imwrite(save_file, image);
    INFO("Done");
    return 0;
}