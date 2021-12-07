// https://github.com/shouxieai/unet-pytorch
#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/preprocess_kernel.cuh>
#include <common/ilogger.hpp>

using namespace cv;
using namespace std;

bool requires(const char* name);

static vector<int> _classes_colors = {
    0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 
    128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 
    64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 12
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

    cv::Mat d2i_mat(){
        return cv::Mat(2, 3, CV_32F, d2i);
    }
};

static cv::Mat image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, int ibatch){

    CUDAKernel::Norm normalize = CUDAKernel::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDAKernel::ChannelType::Invert);
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
        affine_matrix_device, 128, 
        normalize, stream
    );
    return affine.d2i_mat().clone();
}

static tuple<cv::Mat, cv::Mat> post_process(shared_ptr<TRT::Tensor>& tensor, int ibatch){

    cv::Mat output_prob(tensor->size(1), tensor->size(2), CV_32F);
    cv::Mat output_index(tensor->size(1), tensor->size(2), CV_8U);

    int num_class = tensor->size(3);
    float* pnet   = tensor->cpu<float>(ibatch);
    float* prob   = output_prob.ptr<float>(0);
    uint8_t* pidx = output_index.ptr<uint8_t>(0);

    for(int k = 0; k < output_prob.cols * output_prob.rows; ++k, pnet+=num_class, ++prob, ++pidx){
        int ic = std::max_element(pnet, pnet + num_class) - pnet;
        *prob  = pnet[ic];
        *pidx  = ic;
    }
    return make_tuple(output_prob, output_index);
}

static void render(cv::Mat& image, const cv::Mat& prob, const cv::Mat& iclass){

    auto pimage = image.ptr<Vec3b>(0);
    auto pprob  = prob.ptr<float>(0);
    auto pclass = iclass.ptr<uint8_t>(0);

    for(int i = 0; i < image.cols*image.rows; ++i, ++pimage, ++pprob, ++pclass){

        int iclass        = *pclass;
        float probability = *pprob;
        auto& pixel       = *pimage;
        float foreground  = min(0.6f + probability * 0.2f, 0.8f);
        float background  = 1 - foreground;
        for(int c = 0; c < 3; ++c){
            auto value = pixel[c] * background + foreground * _classes_colors[iclass * 3 + 2-c];
            pixel[c] = min((int)value, 255);
        }
    }
}

static void inference(TRT::Mode mode, const string& model_file){

    auto engine = TRT::load_infer(model_file);
    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    auto image      = cv::imread("street.jpg");
    if(image.empty()){
        INFOE("Load image failed.");
        return;
    }

    /* 
        engine的workspace与input、output等获取的workspace是同一个
        engine的stream与input、output等获取的stream是同一个
    */
    auto input      = engine->tensor("images");   // engine->input(0);
    auto output     = engine->tensor("output");   // engine->output(0);

    // use max = 1 batch to inference.
    int max_batch_size = 1;
    input->resize_single_dim(0, max_batch_size).to_gpu();  

    // set batch = 1  image
    int ibatch = 0;
    auto invert_matrix = image_to_tensor(image, input, ibatch);

    // do async
    engine->forward(false);
    
    cv::Mat prob, iclass;
    tie(prob, iclass) = post_process(output, ibatch);

    cv::warpAffine(prob, prob, invert_matrix, image.size(), cv::INTER_LINEAR);
    cv::warpAffine(iclass, iclass, invert_matrix, image.size(), cv::INTER_NEAREST);
    render(image, prob, iclass);

    INFO("Done, Save to unet.predict.jpg");
    cv::imwrite("unet.predict.jpg", image);
}   

static void test(TRT::Mode mode, const string& model, int deviceid = 0){

    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(deviceid);

    const char* name = model.c_str();
    INFO("===================== test %s %s ==================================", mode_name, name);

    if(!requires(name))
        return;

    string onnx_file    = iLogger::format("%s.onnx", name);
    string model_file   = iLogger::format("%s.%s.trtmodel", name, mode_name);
    int test_batch_size = 1;
    
    if(!iLogger::exists(model_file)){
        TRT::compile(
            mode,                       // FP32、FP16、INT8
            test_batch_size,            // max batch size
            onnx_file,                  // source 
            model_file,                 // save to
            {},
            nullptr,
            ""
        );
    }
    inference(mode, model_file);
}

int direct_unet(){
    test(TRT::Mode::FP32, "unet");
    return 0;
}