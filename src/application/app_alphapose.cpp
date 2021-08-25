
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

// 模型编译时使用的头文件
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

    const char* name = "sppe";
    if(not requires(name))
        return 0;

    string onnx_file = iLogger::format("%s.onnx", name);
    string model_file = iLogger::format("%s.fp32.trtmodel", name);
    int test_batch_size = 1;  // 当你需要修改batch大于1时，请查看yolox.cpp:260行备注
    
    // 动态batch和静态batch，如果你想要弄清楚，请打开http://www.zifuture.com:8090/
    // 找到右边的二维码，扫码加好友后进群交流（免费哈，就是技术人员一起沟通）
    if(!iLogger::exists(model_file)){
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

    Mat image = imread("inference/gril.jpg");
    auto engine = AlphaPose::create_infer(model_file, 0);
    auto box = Rect(158, 104, 176, 693);
    auto keys = engine->commit(image, box).get();
    for(int i = 0; i < keys.size(); ++i){
        float x = keys[i].x;
        float y = keys[i].y;
        cv::circle(image, Point(x, y), 5, Scalar(0, 255, 0), -1, 16);
    }

    auto save_file = "pose.show.jpg";
    INFO("Save to %s", save_file);
    
    imwrite(save_file, image);
    INFO("Done");
    return 0;
}