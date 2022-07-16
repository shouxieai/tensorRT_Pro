
#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/preprocess_kernel.cuh>
#include <common/ilogger.hpp>

using namespace cv;
using namespace std;

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

enum class Type : int{
    V5 = 0,
    X  = 1,
    V3 = 2,
    V7 = 3
};

struct AffineMatrix{
    float i2d[6];       // image to dst(network), 2x3 matrix
    float d2i[6];       // dst to image, 2x3 matrix

    void compute(const cv::Size& from, const cv::Size& to){
        float scale_x = to.width / (float)from.width;
        float scale_y = to.height / (float)from.height;
        float scale = std::min(scale_x, scale_y);
        /* 
                + scale * 0.5 - 0.5 的主要原因是使得中心更加对齐，下采样不明显，但是上采样时就比较明显
            参考：https://www.iteye.com/blog/handspeaker-1545126
        */
        i2d[0] = scale;  i2d[1] = 0;  i2d[2] = -scale * from.width  * 0.5  + to.width * 0.5 + scale * 0.5 - 0.5;
        i2d[3] = 0;  i2d[4] = scale;  i2d[5] = -scale * from.height * 0.5 + to.height * 0.5 + scale * 0.5 - 0.5;
        
        cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
        cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
        cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
    }

    cv::Mat i2d_mat(){
        return cv::Mat(2, 3, CV_32F, i2d);
    }
};

bool requires(const char* name);

// code in application/app_yolo/yolo_decode.cu
namespace Yolo{
    void decode_kernel_invoker(
        float* predict, int num_bboxes, int num_classes, float confidence_threshold, 
        float* invert_affine_matrix, float* parray,
        int max_objects, cudaStream_t stream
    );

    void nms_kernel_invoker(
        float* parray, float nms_threshold, int max_objects, cudaStream_t stream
    );
};

static const char* type_name(Type type){
    switch(type){
    case Type::V5: return "YoloV5";
    case Type::X: return "YoloX";
    default: return "Unknow";
    }
}

static void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, Type type, int ibatch){

    CUDAKernel::Norm normalize;
    if(type == Type::V5 || type == Type::V3 || type == Type::V7){
        normalize = CUDAKernel::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDAKernel::ChannelType::Invert);
    }else if(type == Type::X){
        //float mean[] = {0.485, 0.456, 0.406};
        //float std[]  = {0.229, 0.224, 0.225};
        //normalize_ = CUDAKernel::Norm::mean_std(mean, std, 1/255.0f, CUDAKernel::ChannelType::Invert);
        normalize = CUDAKernel::Norm::None();
    }else{
        INFOE("Unsupport type %d", type);
    }
    
    Size input_size(tensor->size(3), tensor->size(2));
    AffineMatrix affine;
    affine.compute(image.size(), input_size);

    size_t size_image      = image.cols * image.rows * 3;
    size_t size_matrix     = iLogger::upbound(sizeof(affine.d2i), 32);
    auto workspace         = tensor->get_workspace();
    uint8_t* gpu_workspace        = (uint8_t*)workspace->gpu(size_matrix + size_image);
    float*   affine_matrix_device = (float*)gpu_workspace;
    uint8_t* image_device         = size_matrix + gpu_workspace;

    uint8_t* cpu_workspace        = (uint8_t*)workspace->cpu(size_matrix + size_image);
    float* affine_matrix_host     = (float*)cpu_workspace;
    uint8_t* image_host           = size_matrix + cpu_workspace;
    auto stream                   = tensor->get_stream();

    memcpy(image_host, image.data, size_image);
    memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
    checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream));
    checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i), cudaMemcpyHostToDevice, stream));

    CUDAKernel::warp_affine_bilinear_and_normalize_plane(
        image_device,               image.cols * 3,       image.cols,       image.rows, 
        tensor->gpu<float>(ibatch), input_size.width,     input_size.height, 
        affine_matrix_device, 114, 
        normalize, stream
    );
}

static void inference(Type type, TRT::Mode mode, const string& model_file){

    auto engine = TRT::load_infer(model_file);
    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    auto image      = cv::imread("inference/gril.jpg");
    if(image.empty()){
        INFOE("Load image failed.");
        return;
    }

    /* 
        engine的workspace与input、output等获取的workspace是同一个
        engine的stream与input、output等获取的stream是同一个
    */
    auto input      = engine->tensor("images");   // engine->input(0);
    auto output     = engine->tensor("output");  // engine->output(0);
    int num_bboxes  = output->size(1);
    int num_classes = output->size(2) - 5;
    float confidence_threshold = 0.25;
    float nms_threshold        = 0.5;
    int MAX_IMAGE_BBOX         = 1024;
    int NUM_BOX_ELEMENT        = 7;  // left, top, right, bottom, confidence, class, keepflag
    TRT::Tensor output_array_device(TRT::DataType::Float);

    // use max = 1 batch to inference.
    int max_batch_size = 1;
    input->resize_single_dim(0, max_batch_size).to_gpu();  
    output_array_device.resize(max_batch_size, 1 + MAX_IMAGE_BBOX * NUM_BOX_ELEMENT).to_gpu();
    output_array_device.set_stream(engine->get_stream());

    // set batch = 1  image
    int ibatch = 0;
    image_to_tensor(image, input, type, ibatch);

    // do async
    engine->forward(false);
    
    float* d2i_affine_matrix = static_cast<float*>(input->get_workspace()->gpu());
    Yolo::decode_kernel_invoker(
        output->gpu<float>(ibatch),
        num_bboxes, num_classes,
        confidence_threshold,
        d2i_affine_matrix, output_array_device.gpu<float>(ibatch),
        MAX_IMAGE_BBOX, engine->get_stream()
    );

    Yolo::nms_kernel_invoker(
        output_array_device.gpu<float>(ibatch),
        nms_threshold, 
        MAX_IMAGE_BBOX, engine->get_stream()
    );

    float* parray = output_array_device.cpu<float>();
    int num_box = min(static_cast<int>(*parray), MAX_IMAGE_BBOX);

    for(int i = 0; i < num_box; ++i){
        float* pbox  = parray + 1 + i * NUM_BOX_ELEMENT;
        int keepflag = pbox[6];
        if(keepflag == 1){
            // left,      top,     right,  bottom, confidence,class, keepflag
            // pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], pbox[5], pbox[6]
            float left       = pbox[0];
            float top        = pbox[1];
            float right      = pbox[2];
            float bottom     = pbox[3];
            float confidence = pbox[4];
            int label        = static_cast<int>(pbox[5]);

            uint8_t b, g, r;
            tie(b, g, r) = iLogger::random_color(label);
            cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(b, g, r), 5);

            auto name    = cocolabels[label];
            auto caption = iLogger::format("%s %.2f", name, confidence);
            int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(image, cv::Point(left-3, top-33), cv::Point(left + width, top), cv::Scalar(b, g, r), -1);
            cv::putText(image, caption, cv::Point(left, top-5), 0, 1, cv::Scalar::all(0), 2, 16);
        }
    }

    INFO("Done, save to detect.out.jpg");
    cv::imwrite("detect.out.jpg", image);
}

static void test(Type type, TRT::Mode mode, const string& model, int deviceid = 0){

    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(deviceid);

    auto int8process = [=](int current, int count, const vector<string>& files, shared_ptr<TRT::Tensor>& tensor){

        INFO("Int8 %d / %d", current, count);

        for(int i = 0; i < files.size(); ++i){
            auto image = cv::imread(files[i]);
            image_to_tensor(image, tensor, type, i);
            tensor->synchronize();
        }
    };

    const char* name = model.c_str();
    INFO("===================== test %s %s %s ==================================", type_name(type), mode_name, name);

    if(!requires(name))
        return;

    string onnx_file = iLogger::format("%s.onnx", name);
    string model_file = iLogger::format("%s.%s.trtmodel", name, mode_name);
    int test_batch_size = 16;
    
    if(!iLogger::exists(model_file)){
        TRT::compile(
            mode,                       // FP32、FP16、INT8
            test_batch_size,            // max batch size
            onnx_file,                  // source 
            model_file,                 // save to
            {},
            int8process,
            "inference"
        );
    }
    inference(type, mode, model_file);
}

int direct_yolo(){
 
    test(Type::V5, TRT::Mode::FP32, "yolov5s");
    return 0;
}