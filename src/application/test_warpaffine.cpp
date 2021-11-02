#include <common/preprocess_kernel.cuh>
#include <common/trt_tensor.hpp>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

using namespace cv;

void verify_result(TRT::Tensor& gpu, cv::Mat& cpu){

    cv::Mat_<float> b(gpu.height(), gpu.width(), gpu.cpu<float>(0, 0));
    cv::Mat_<float> g(gpu.height(), gpu.width(), gpu.cpu<float>(0, 1));
    cv::Mat_<float> r(gpu.height(), gpu.width(), gpu.cpu<float>(0, 2));
    cv::Mat gpu3c;
    cv::merge(std::vector<cv::Mat>{b, g, r}, gpu3c);

    gpu3c.convertTo(gpu3c, CV_8U);

    auto pg = gpu3c.data;
    auto pc = cpu.data;
    int count = gpu3c.rows * gpu3c.cols * 3;
    int count0 = 0;
    int count1 = 0;
    int count2 = 0;
    int count3 = 0;
    int count5 = 0;
    for(int i = 0; i < count; ++i, ++pg, ++pc){
        
        int gval = *pg;
        int cval = *pc;
        int absdiff = abs(gval - cval);
        if(absdiff > 0)
            count0++;
        
        if(absdiff > 1)
            count1++;
        
        if(absdiff > 2)
            count2++;

        if(absdiff > 3)
            count3++;
        
        if(absdiff > 5)
            count5++;
    }
    INFO("absdiff count0 = %d, count1 = %d, count2 = %d, count3 = %d, count5 = %d", count0, count1, count2, count3, count5);

    cv::Mat absdiff_image;
    cv::absdiff(gpu3c, cpu, absdiff_image);

    absdiff_image.convertTo(absdiff_image, CV_8U, 30);
    cv::imwrite("absdiff_image.png", absdiff_image);
}

cv::Mat load_image(){
    auto file = "inference/gril.jpg";
    auto image = cv::imread(file);
    if(image.empty()){
        INFOE("Load image failed %s", file);
        exit(0);
    }

    /* 结论：对于尺寸能够整除的情况时，且image大小>=50，结果完全一致
     否则，会有细微偏差，比如count0会比较异常
     比如 800 / 50能够整除
    */
    cv::resize(image, image, cv::Size(50, 100));
    return image;
}

int test_warpaffine(){

    cudaStream_t stream = nullptr;
    TRT::MixMemory memory, matrix;
    TRT::Tensor my_gpu_warpaffine;
    cv::Mat opencv_cpu_warpaffine;
    cv::Mat_<float> i2d_matrix(2, 3), d2i_matrix(2, 3);
    cv::Size test_size(800, 800);

    auto image  = load_image();
    float sx    = test_size.width / (float)image.cols;
    float sy    = test_size.height / (float)image.rows;
    float scale = min(sx, sy);
    float i2d_matrix_values[] = {
        scale, 0, -scale * image.cols * 0.5f + test_size.width * 0.5f + scale * 0.5f - 0.5f,
        0, scale, -scale * image.rows * 0.5f + test_size.height * 0.5f + scale * 0.5f - 0.5f,
    };

    memcpy(i2d_matrix.ptr<float>(0), i2d_matrix_values, sizeof(i2d_matrix_values));
    cv::warpAffine(image, opencv_cpu_warpaffine, i2d_matrix, test_size);
    cv::imwrite("opencv_cpu_warpaffine.png", opencv_cpu_warpaffine);
    cv::invertAffineTransform(i2d_matrix, d2i_matrix);

    uint8_t* image_data_device = (uint8_t*)memory.gpu(image.rows * image.cols * 3);

    cudaStreamCreate(&stream);
    cudaMemcpyAsync(image_data_device, image.data, memory.gpu_size(), cudaMemcpyHostToDevice, stream);

    auto norm = CUDAKernel::Norm::None();
    float* d2i_matrix_device = (float*)matrix.gpu(6 * sizeof(float));
    cudaMemcpyAsync(d2i_matrix_device, d2i_matrix.ptr<float>(0), matrix.gpu_size(), cudaMemcpyHostToDevice, stream);

    my_gpu_warpaffine.resize(1, 3, test_size.height, test_size.width);
    CUDAKernel::warp_affine_bilinear_and_normalize_plane(
        image_data_device, image.cols * 3, image.cols, image.rows,
        my_gpu_warpaffine.gpu<float>(), my_gpu_warpaffine.width(), my_gpu_warpaffine.height(), d2i_matrix_device, 0, 
        norm, stream
    );
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    my_gpu_warpaffine.save_to_file("my_gpu_warpaffine.tensor");

    verify_result(my_gpu_warpaffine, opencv_cpu_warpaffine);
    INFO("done to warp.tensor");
    return 0;
}