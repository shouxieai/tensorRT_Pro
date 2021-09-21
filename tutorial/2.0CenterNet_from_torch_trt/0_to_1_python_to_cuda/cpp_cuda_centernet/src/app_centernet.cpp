#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include "centernet.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

// the following includes are for learning purpose
#include "cpp_utils.hpp"
#include "cuda_utils.cuh"


using namespace std;
using namespace cv;

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

static void forward_engine(const string& engine_file, Centernet::Type type){
    // *********************************** c++ version *************************************

    #pragma region c++ version
    #if 0
    auto engine = TRT::load_infer(engine_file);

    // 1. basic info
    #pragma region define input, output and info of image
    
    auto input = engine->input(0);
    auto hm = engine->output(0);
    auto pooled_hm = engine->output(1);
    auto wh = engine->output(2);
    auto regxy = engine->output(3);
    
    Mat src_img = cv::imread("car.jpg");
    Mat M_2x3;
    Mat dst_img;
    auto image = src_img.clone();

    int net_w = 512; int net_h = 512;
    int img_w = image.cols; int img_h = image.rows;

    float scale = std::min(
        net_w / (float)img_w,
        net_h / (float)img_h
    );

    #pragma endregion define input and output

    // 2. warpaffine and normalization
    warpaffine_and_normalize_opencv_cpp(image, img_w, img_h, M_2x3, dst_img, net_w, net_h);
    
    // 3. image.shape :hwc -> bchw
    resize_into(dst_img, input);  // modify the dst into the shape of the input and put it into the input tensor.

    // 4. save the input to a tensor to check whether the preprocessing is correct
    // engine->input(0)->save_to_file("input_after_preprocessing.binary");

    // 5. forward 
    engine->forward();
    // engine->output(0)->save_to_file("hm.data");
    // engine->output(1)->save_to_file("wh.data");
    // engine->output(2)->save_to_file("reg_xy.data");
    // INFO("The output are saved.");
    
    // 6. decode
    float conf_T = 0.3;
    decode(src_img, engine, conf_T, M_2x3);
    INFO("inference done");
    
    #endif

    #pragma endregion c++ version
 
    // ********************************** cuda version *************************************
    #pragma region cuda version
    int gpuid = 0;
    TRT::set_device(gpuid);
    auto engine = TRT::load_infer(engine_file);
    auto stream_ = engine->get_stream();

    #pragma region 1.load image, define input&output and do preprocessing
    
    Mat src_image = cv::imread("car.jpg");
    
    auto input         = engine->tensor("images");
    auto output        = engine->tensor("output");
    auto reg           = engine->output(0);
    auto wh            = engine->output(1);
    auto hm            = engine->output(2);
    auto pool_hm       = engine->output(3);
    
    // preprocess the image and put the result into input
    float* d2i_ptr_device = nullptr;      // move the d2i mat to GPU
    // cudaMalloc(&d2i_ptr_device, 8);

    preprocess_on_gpu(src_image, d2i_ptr_device, input);
    
    // input->save_to_file("affined_result_gpu.tensor");
    // debug: check the preprocessed image
    // input->resize(3, 512, 512);
    // auto input_gpu_ptr = input->gpu<float>(0);
    // INFO(input->shape_string());

    // Size input_size(512, 512);
    // Mat preprocessed_img(input_size, CV_32FC3, input_gpu_ptr);
    
    // Scalar mean(0.408, 0.447, 0.470);
    // Scalar std (0.289, 0.274, 0.278);
    // auto img = pre_img * std + mean;
    // Mat affined_img;
    // preprocessed_img.convertTo(affined_img, CV_8U);
    
    #pragma region debug
    #if 0
    unsigned char img_arr[]{0,0,0,255,255,255};
    Mat img_mat(Size(2, 1), CV_8UC3, img_arr);
    // cv::imwrite("test.jpg", img_mat);
    for(int row=0; row<1; ++row){
        for(int col=0; col<2; ++col){
            for (int c = 0; c < 3; ++c)
            printf("%d \n", img_mat.at<Vec3b>(row, col)[c]);
        }
    }
    #endif
    #pragma endregion debug


    int a = 1;
    #if 1

    Mat image = src_image.clone();
    
    // input->set_to(1.5);
    // bool ret = input->save_to_file("trt_i_car.tensor");
    
    #pragma endregion 1.

    #pragma region 2. inference
    engine->forward(); // false means no sync before decoding. If set it to false, you can't access the output.
    // output->save_to_file("output.tensor");
    #pragma endregion

    // output ->save_to_file("trt_5_output_car.tensor");
    // reg    ->save_to_file("trt_reg_car.tensor");
    // wh     ->save_to_file("trt_wh_car.tensor");
    // hm     ->save_to_file("trt_hm_car.tensor");
    // pool_hm->save_to_file("trt_pool_hm_car.tensor");


    #pragma region 3. decode
    float conf_T = 0.3;
    // float de_conf_T = desigmoid(conf_T);
    int num_channels = 164;
    float* output_ptr = output->gpu<float>(0);


    // hardcoded in tutorial for simplicity
    float inv_M[8] = {2.109375,    0.,       -135.,
                        0.,          2.109375,    0.  };

    float* inv_M_ptr_device = nullptr; // the ptr var is on cpu but stores the gpu addr.
    cudaMalloc(&inv_M_ptr_device, sizeof(inv_M));
    cudaMemcpy(inv_M_ptr_device, inv_M, sizeof(inv_M), cudaMemcpyHostToDevice);

    int* array_ptr = new(int); // e.g. 2065667216, 20, 30
    int* item_ptr = array_ptr + 1;


    int MAX_IMAGE_BBOX = 1000;
    TRT::Tensor output_array_device (TRT::DataType::Float); // create a tensor to store resulting bboxes.
    output_array_device.resize(1, 1 + 100 * 7).to_gpu();
    int ibatch = 0 ;
    float* output_array_ptr = output_array_device.gpu<float>(ibatch);
    checkCudaRuntime(cudaMemsetAsync(output_array_ptr, 0, sizeof(int), stream_));
    

    Centernet::decode_kernel_invoker(output_ptr, output->size(1), num_channels,
                    conf_T, inv_M_ptr_device, output_array_ptr, 1000, stream_);

    output_array_device.to_cpu();
    

    float* parray = output_array_device.cpu<float>(ibatch);

            
    for(int i = 0; i < *parray; ++i){
        float* begin = parray + 1 + i * 7; // the begining idx of the current bbox.
        float left = *begin;          float top = *(begin + 1);     float right = *(begin + 2); float bottom = *(begin + 3);
        float confidence = *(begin + 4); int label = (int)*(begin + 5); float status = *(begin + 6);

        uint8_t b, g, r;
        tie(r, g, b) = iLogger::random_color(label);
        cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(b,g,r), 5);

        auto name = cocolabels[label];
        auto caption = iLogger::format("%s %.2f", name, confidence);
        int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(left-3, top-33), cv::Point(left + width, top), cv::Scalar(b, g, r), -1);
        cv::putText(image, caption, cv::Point(left, top-5), 0, 1, cv::Scalar::all(0), 2, 16);
    }
    
    string file_name = iLogger::file_name("final_result_cuda", false);
    string save_path = iLogger::format("%s/%s.jpg", "/datav/shared/tensorRT_cpp/tutorial/2.0CenterNet_from_torch_trt/0_to_1_python_to_cuda/cpp_cuda_centernet/workspace", file_name.c_str());
    INFO("Save to %s", save_path.c_str());
    cv::imwrite(save_path, image);
    #endif
    
    
    
    #pragma endregion 3.

    #pragma endregion cuda version
    

    #pragma region DEBUG_SNIPPEST
    // engine->print();
    // engine->input(0)->set_to(1);     // input tensor , 让这个tensor 全为1  torch.ones((1,3,512,512))
    // engine->forward();
    // engine->output(0)->save_to_file("output0.data");
    // engine->output(1)->save_to_file("output1.data");
    // engine->output(2)->save_to_file("output2.data");
    #pragma endregion DEBUG_SNIPPEST


}


static void test(Centernet::Type type, TRT::Mode mode, const string& model){
    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(0);

    const char* name = model.c_str();
    INFO("===================== test %s %s %s ==================================", Centernet::type_name(type), mode_name, name);


    string onnx_file = iLogger::format("%s.onnx", name);
    string model_file = iLogger::format("%s.%s.trtmodel", name, mode_name);
    int test_batch_size = 16;

    if(not iLogger::exists(model_file)){
        TRT::compile(
            mode,                       // FP32、FP16、INT8
            test_batch_size,            // max batch size
            onnx_file,                  // source 
            model_file                  // save to
        );
    }

    forward_engine(model_file, type);
}

int app_centernet(){
    iLogger::set_log_level(iLogger::LogLevel::Info);
    // test(Centernet::Type::DET, TRT::Mode::FP32, "ctnet_r18_dcn_4outs_hm_pool_wh_reg"); // Type indicates the task. name string indicates the model size and structure.
    test(Centernet::Type::DET, TRT::Mode::FP32, "ctnet_r18_dcn_5outs_reg_wh_hm_pool_output"); // Type indicates the task. name string indicates the model size and structure.

}