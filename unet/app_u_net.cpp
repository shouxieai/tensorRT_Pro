
#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include "app_u_net/u_net.hpp"

using namespace std;


bool requires(const char* name);

static void forward_engine(const string& engine_file, bool flag){


    auto engine = U_net::create_infer(engine_file, 0, 0.4f);
    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    string root = iLogger::format("%s_result", "U_net");
    iLogger::rmtree(root);
    iLogger::mkdir(root);

    auto files = iLogger::find_files("inference", "*.jpg;*.jpeg;*.png;*.gif;*.tif");
    auto t_start    = iLogger::timestamp_now_float();
    
    cv::Mat srcImg;
    U_net::SegmentResultArray tSegmentOutput;
    cv::Mat mask_img;
    cv::Mat final_image;
    string file_name;
    string save_path;
    // for(int j=0; j<20; ++j){
    for(int i = 0; i < files.size(); ++i){

        srcImg = cv::imread(files[i]);
        tSegmentOutput = engine->commit(srcImg).get();
        printf("成功得到 future！！！%d/%d \n", i, files.size()-1);
        file_name = iLogger::file_name(files[i], false);
        save_path = iLogger::format("%s/%s.jpg", root.c_str(), file_name.c_str());
        // 框给画到图上
        if (flag == true){
            printf("%d, %d \n", tSegmentOutput[0].out_img.cols, tSegmentOutput[0].out_img.rows);
            cv::warpAffine(tSegmentOutput[0].out_img, mask_img, tSegmentOutput[0].invertMatrix, srcImg.size());
            cv::addWeighted(srcImg, 0.7, mask_img, 0.3, 0.0, final_image);
            cv::imwrite(save_path.c_str(), final_image);
            // INFO("Save to %s, %.2f ms", save_path.c_str(), inference_time);
            INFO("Save to %s", save_path.c_str());
            }
        else{
            printf("%s\n", save_path.c_str());
            cv::imwrite(save_path.c_str(), tSegmentOutput[0].out_img);
            INFO("Save to %s", save_path.c_str());
            // INFO("Save to %s, %.2f ms", save_path.c_str(), inference_time);
        }

    }
    // }
    float inference_time = iLogger::timestamp_now_float() - t_start;
    INFO("测试轮数: %d,平均一轮时间: %.2f ms", 20 * files.size(), inference_time / (20 * files.size()));
}

static void forward_engine_batch(const string& engine_file, bool flag){


    auto engine = U_net::create_infer(engine_file, 0, 0.4f);
    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    string root = iLogger::format("%s_result", "U_net");
    iLogger::rmtree(root);
    iLogger::mkdir(root);

    auto files = iLogger::find_files("inference", "*.jpg;*.jpeg;*.png;*.gif;*.tif");   

    cv::Mat mask_img;
    cv::Mat final_image;
    vector<cv::Mat> images;

    string file_name;
    string save_path;

    for(int i = 0; i < files.size(); ++i)
        images.emplace_back(cv::imread(files[i]));   // 得到的 batch

    // warmup
    engine->commits(images).back().get();
    
    auto time_begin   = iLogger::timestamp_now_float();
    auto vecSegmentoutput = engine->commits(images);

    vecSegmentoutput.back().get();     // 得到的
    float average_time = (iLogger::timestamp_now_float() - time_begin) / files.size();

    INFO("future 数量: %d", vecSegmentoutput.size());
    for(int i = 0; i < images.size(); ++i){

        printf("成功得到 future！！！%d/%d \n", i, files.size()-1);
        file_name = iLogger::file_name(files[i], false);
        save_path = iLogger::format("%s/%s.jpg", root.c_str(), file_name.c_str());
        
        auto& image = images[i];
        auto tSegmentOutput  = vecSegmentoutput[i].get(); 
        if (flag == true){
            cv::warpAffine(tSegmentOutput[i].out_img, mask_img, tSegmentOutput[i].invertMatrix, image.size());
            INFO("原图: %d, %d , mask_image: %d, %d\n", image.cols, image.rows, mask_img.cols, mask_img.rows);
            cv::addWeighted(image, 0.7, mask_img, 0.3, 0.0, final_image);
            cv::imwrite(save_path.c_str(), final_image);
            INFO("Save to %s, %.2f ms", save_path.c_str(), average_time);
            // INFO("Save to %s", save_path.c_str());
        }
        else{
            printf("%s\n", save_path.c_str());
            cv::imwrite(save_path.c_str(), tSegmentOutput[i].out_img);
            // INFO("Save to %s", save_path.c_str());
            INFO("Save to %s, %.2f ms", save_path.c_str(), average_time);
        }
    }
    INFO("Done");
}


static void test_fp32(const string& str, bool flag){

    TRT::set_device(0);
    // 是否需要渲染
    INFO("===================== test %s fp32 ==================================", str.c_str());

    const char* name = str.c_str();

    // if(not requires(name))
    //     return;

    string onnx_file = iLogger::format("%s.onnx", name);
    string model_file = iLogger::format("%s.fp16.trtmodel", name);
    int test_batch_size = 2;  // 当你需要修改batch大于1时，请注意你的模型是否修改（看readme.md代码修改部分），否则会有错误
    
    // 动态batch和静态batch，如果你想要弄清楚，请打开http://www.zifuture.com:8090/
    // 找到右边的二维码，扫码加好友后进群交流（免费哈，就是技术人员一起沟通）
    if(not iLogger::exists(model_file)){
        TRT::compile(
            TRT::Mode::FP16,            // 编译方式有，FP32、FP16、INT8
            test_batch_size,            // 指定编译的batch size
            onnx_file,                  // 需要编译的onnx文件
            model_file                  // 储存的模型文件
        );
    }

    // forward_engine(model_file, flag);
    // forward_engine_batch(model_file, flag);


}


int app_u_net(){

    // 测试模型的名字, 以及是否选择渲染
    test_fp32("u_net", false);

    return 0;
}