
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
#include "app_retinaface/retinaface.hpp"
#include "app_arcface/arcface.hpp"

using namespace std;
using namespace cv;

bool requires(const char* name);
bool compile_retinaface(int input_width, int input_height, string& out_model_file);

static bool compile_models(){

    TRT::set_device(0);
    string model_file;

    if(!compile_retinaface(640, 480, model_file))
        return false;

    const char* onnx_files[]{"arcface_iresnet50"};
    for(auto& name : onnx_files){
        if(not requires(name))
            return false;

        string onnx_file = iLogger::format("%s.onnx", name);
        string model_file = iLogger::format("%s.fp32.trtmodel", name);
        int test_batch_size = 1;  // 当你需要修改batch大于1时，请查看yolox.cpp:260行备注
        
        // 动态batch和静态batch，如果你想要弄清楚，请打开http://www.zifuture.com:8090/
        // 找到右边的二维码，扫码加好友后进群交流（免费哈，就是技术人员一起沟通）
        if(not iLogger::exists(model_file)){
            bool ok = TRT::compile(
                TRT::TRTMode_FP32,   // 编译方式有，FP32、FP16、INT8
                {},                         // onnx时无效，caffe的输出节点标记
                test_batch_size,            // 指定编译的batch size
                onnx_file,                  // 需要编译的onnx文件
                model_file,                 // 储存的模型文件
                {},                         // 指定需要重定义的输入shape，这里可以对onnx的输入shape进行重定义
                true                        // 是否采用动态batch维度，true采用，false不采用，使用静态固定的batch size
            );

            if(!ok) return false;
        }
    }
    return true;
}

tuple<Mat, vector<string>> build_library(shared_ptr<RetinaFace::Infer> detector, shared_ptr<Arcface::Infer> arcface){
    
    Mat_<float> features(0, 512);
    vector<string> names;
    auto libs = iLogger::find_files("face/library");
    INFO("Build library, %d images", libs.size());

    for(auto& file : libs){
        auto name  = iLogger::file_name(file, false);
        Mat image  = imread(file);
        INFO("%d x %d", image.rows, image.cols);
        auto faces = detector->commit(image).get();

        for(int i = 0; i < faces.size(); ++i){
            auto& face = faces[i];
            auto box   = Rect(face.left, face.top, face.right-face.left, face.bottom-face.top);
            box        = box & Rect(0, 0, image.cols, image.rows);

            if(box.width < 80 or box.height < 80)
                continue;

            if(box.area() == 0){
                INFOE("Invalid box, %d, %d, %d, %d", box.x, box.y, box.width, box.height);
                continue;
            }

            auto crop  = image(box).clone();
            Arcface::landmarks landmarks;
            for(int j = 0; j < 10; ++j)
                landmarks.points[j] = face.landmark[j] - (j % 2 == 0 ? face.left : face.top);

            auto feature     = arcface->commit(make_tuple(crop, landmarks)).get();
            string face_name = iLogger::format("%s.%02d", name.c_str(), i);
            features.push_back(feature);
            names.push_back(face_name);

            INFO("New face [%s], %d feature, %.5f", face_name.c_str(), feature.cols, face.confidence);

            rectangle(image, cv::Point(face.left, face.top), cv::Point(face.right, face.bottom), Scalar(0, 255, 0), 2);
            for(int j = 0; j < 5; ++j)
                circle(image, Point(face.landmark[j*2+0], face.landmark[j*2+1]), 3, Scalar(0, 255, 0), -1, 16);
            putText(image, face_name, cv::Point(face.left, face.top), 0, 1, Scalar(0, 255, 0), 1, 16);
        }

        string save_file = iLogger::format("face/library_draw/%s.jpg", name.c_str());
        imwrite(save_file, image);
    }
    return make_tuple(features, names);
}

int app_arcface(){

    TRT::set_device(0);
    INFO("===================== test arcface fp32 ==================================");

    if(!compile_models())
        return 0;

    iLogger::rmtree("face/library_draw");
    iLogger::rmtree("face/result");
    iLogger::mkdirs("face/library_draw");
    iLogger::mkdirs("face/result");

    auto detector = RetinaFace::create_infer("mb_retinaface.640x480.fp32.trtmodel", 0, 0.5f);
    auto arcface  = Arcface::create_infer("arcface_iresnet50.fp32.trtmodel", 0);
    auto library  = build_library(detector, arcface);

    auto files    = iLogger::find_files("face/recognize");
    for(auto& file : files){

        auto image  = imread(file);
        auto faces  = detector->commit(image).get();
        vector<string> names(faces.size());
        for(int i = 0; i < faces.size(); ++i){
            auto& face = faces[i];
            auto box   = Rect(face.left, face.top, face.right-face.left, face.bottom-face.top);
            box        = box & Rect(0, 0, image.cols, image.rows);
            auto crop  = image(box).clone();
            
            Arcface::landmarks landmarks;
            for(int j = 0; j < 10; ++j)
                landmarks.points[j] = face.landmark[j] - (j % 2 == 0 ? face.left : face.top);

            auto out          = arcface->commit(make_tuple(crop, landmarks)).get();
            auto scores       = Mat(get<0>(library) * out.t());
            float* pscore     = scores.ptr<float>(0);
            int label         = std::max_element(pscore, pscore + scores.rows) - pscore;
            float match_score = max(0.0f, pscore[label]);
            INFO("%f, %s", match_score, get<1>(library)[label].c_str());

            if(match_score > 0.3f){
                names[i] = iLogger::format("%s[%.3f]", get<1>(library)[label].c_str(), match_score);
            }
        }

        for(int i = 0; i < faces.size(); ++i){
            auto& face = faces[i];

            auto color = Scalar(0, 255, 0);
            if(names[i].empty()){
                color = Scalar(0, 0, 255);
                names[i] = "Unknow";
            }
            
            rectangle(image, cv::Point(face.left, face.top), cv::Point(face.right, face.bottom), color, 3);
            putText(image, names[i], cv::Point(face.left, face.top), 0, 1, color, 1, 16);
        }

        auto save_file = iLogger::format("face/result/%s.jpg", iLogger::file_name(file, false).c_str());
        imwrite(save_file, image);
    }
    INFO("Done");
    return 0;
}