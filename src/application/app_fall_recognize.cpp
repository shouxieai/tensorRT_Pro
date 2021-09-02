
#include <stdio.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>

#include "app_yolo/yolo.hpp"
#include "app_alphapose/alpha_pose.hpp"
#include "app_fall_gcn/fall_gcn.hpp"
#include "tools/zmq_remote_show.hpp"
#include "tools/deepsort.hpp"

using namespace cv;
using namespace std;

bool requires(const char* name);

static bool compile_models(){

    TRT::set_device(0);
    const char* onnx_files[]{"yolox_m", "sppe", "fall_bp"};
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
                TRT::Mode::FP32,   // 编译方式有，FP32、FP16、INT8
                test_batch_size,            // 指定编译的batch size
                onnx_file,                  // 需要编译的onnx文件
                model_file,                 // 储存的模型文件
                {},                         // 指定需要重定义的输入shape，这里可以对onnx的输入shape进行重定义
                false                       // 是否采用动态batch维度，true采用，false不采用，使用静态固定的batch size
            );

            if(!ok) return false;
        }
    }
    return true;
}

int app_fall_recognize(){
    cv::setNumThreads(0);

    INFO("===================== test alphapose fp32 ==================================");
    if(!compile_models())
        return 0;
    
    auto pose_model_file     = "sppe.fp32.trtmodel";
    auto detector_model_file = "yolox_m.fp32.trtmodel";
    auto gcn_model_file      = "fall_bp.fp32.trtmodel";
    
    auto pose_model     = AlphaPose::create_infer(pose_model_file, 0);
    auto detector_model = Yolo::create_infer(detector_model_file, Yolo::Type::X, 0, 0.4f);
    auto gcn_model      = FallGCN::create_infer(gcn_model_file, 0);

    Mat image;
    VideoCapture cap("fall_video.mp4");
    INFO("Video fps=%d, Width=%d, Height=%d", 
        (int)cap.get(cv::CAP_PROP_FPS), 
        (int)cap.get(cv::CAP_PROP_FRAME_WIDTH), 
        (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT)
    );

    auto remote_show = create_zmq_remote_show();
    INFO("这个程序需要展示，请使用tools/show.py做客户端，然后启用这里的remote_show进行实时展示");

    auto config  = DeepSORT::TrackerConfig();
    config.set_initiate_state({
        0.1,  0.1,  0.1,  0.1,
        0.2,  0.2,  1,    0.2
    });

    config.set_per_frame_motion({
        0.1,  0.1,  0.1,  0.1,
        0.2,  0.2,  1,    0.2
    });

    auto tracker = DeepSORT::create_tracker(config);
    // VideoWriter writer("fall_video.result.avi", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 
    //     30,
    //     Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT))
    // );
    // if(!writer.isOpened()){
    //     INFOE("Writer failed.");
    //     return 0;
    // }
    while(cap.read(image)){
        auto objects = detector_model->commit(image).get();

        vector<DeepSORT::Box> boxes;
        for(int i = 0; i < objects.size(); ++i){
            auto& obj = objects[i];
            if(obj.class_label != 0) continue;
            boxes.emplace_back(std::move(DeepSORT::convert_to_box(obj)));
        }
        tracker->update(boxes);

        auto final_objects = tracker->get_objects();
        for(int i = 0; i < final_objects.size(); ++i){
            auto& person = final_objects[i];
            if(person->time_since_update() == 0 && person->state() == DeepSORT::State::Confirmed){
                Rect box = DeepSORT::convert_box_to_rect(person->last_position());
                auto keys   = pose_model->commit(make_tuple(image, box)).get();
                auto statev = gcn_model->commit(make_tuple(keys, box)).get();

                FallGCN::FallState state = get<0>(statev);
                float confidence         = get<1>(statev);
                const char* label_name   = FallGCN::state_name(state);
                rectangle(image, DeepSORT::convert_box_to_rect(person->predict_box()), Scalar(0, 255, 0), 1);
                rectangle(image, box, Scalar(0, 255, 255), 1);

                auto line = person->trace_line();
                for(int j = 0; j < (int)line.size() - 1; ++j){
                    auto& p = line[j];
                    auto& np = line[j + 1];
                    cv::line(image, p, np, Scalar(255, 128, 60), 2, 16);
                }

                putText(image, iLogger::format("%d. [%s] %.2f %%", person->id(), label_name, confidence * 100), box.tl(), 0, 1, Scalar(0, 255, 0), 2, 16);
                //INFO("Predict is [%s], %.2f %%", label_name, confidence * 100);
           }
        }
        remote_show->post(image);
        //writer.write(image);
    }
    INFO("Done");
    return 0;
}