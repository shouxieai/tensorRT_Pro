
#include "centernet.hpp" // for accessing Affinematrix.compute()
#include <common/cuda_tools.hpp> // for accessing CUDAcheck
#include <common/preprocess_kernel.cuh> // for accessing CUDAKernel
#include <stdio.h>


using namespace std;
using namespace cv;

void preprocess_on_gpu(Mat& image);


void preprocess_on_gpu(Mat& image, float* d2i_ptr_device, std::shared_ptr<TRT::Tensor> input){
    // 1. basic settings
    int input_width_ = 512;
    int input_height_ = 512;
    int gpu_ = 0;

    float conf_T_ = 0.3;
    TRT::CUStream stream_ = nullptr;
    CUDATools::AutoDevice auto_device(gpu_);
    Size input_size(input_width_, input_height_);


    float mean[] = {0.408, 0.447, 0.470};//! 坑
    float std[] = {0.289, 0.274, 0.278};
    CUDAKernel::Norm normalize_ = CUDAKernel::Norm::mean_std(mean, std, 1/255.0f, CUDAKernel::ChannelType::None);//! 坑

    //2. get the addr of the input tensor(the space is not assigned yet)
    // auto tensor = make_shared<TRT::Tensor>();
    // tensor->set_workspace(make_shared<TRT::MixMemory>());
    auto& tensor = input;//! 坑

    // 3. get the M_2x3 and open up a space for tensor
    Centernet::AffineMatrix affine_M;


    affine_M.compute(image.size(), input_size); 
    
    tensor->set_stream(stream_);
    tensor->resize(1, 3, input_height_, input_width_); // only open up a space. No input data to be filled in

    // 4. define the addr and the space for images and affine_matrix on host and device
    size_t size_image = image.cols * image.rows * 3;   // num_channels are also considered.
    size_t size_matrix = iLogger::upbound(sizeof(affine_M.d2i), 32); // memory assignment
    auto workspace = tensor->get_workspace();

    uint8_t* cpu_workspace        = (uint8_t*)workspace->cpu(size_matrix + size_image);
    float* affine_matrix_host     = (float*)cpu_workspace;
    uint8_t* image_host           = size_matrix + cpu_workspace;
    
    uint8_t* gpu_workspace        = (uint8_t*)workspace->gpu(size_matrix + size_image);        
    float*   affine_matrix_device = (float*)gpu_workspace;


    uint8_t* image_device         = size_matrix + gpu_workspace;

    // 5. data transfer from CPU to GPU
    memcpy(image_host, image.data, size_image);
    memcpy(affine_matrix_host, affine_M.d2i, sizeof(affine_M.d2i));
    checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_));
    checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine_M.d2i), cudaMemcpyHostToDevice, stream_));

     
    CUDAKernel::warp_affine_bilinear_and_normalize_plane(//! 潜在坑：插值不一样
        image_device,         image.cols * 3,       image.cols,       image.rows, 
        tensor->gpu<float>(), input_width_,         input_height_, 
        affine_matrix_device, 
        0, normalize_, stream_
    );
    int a = 0;

};



    


    
    




auto desigmoid = [](float y){
        return - log(1/y - 1);
    };

auto sigmoid = [](float x){
        return 1/(1+ exp(-x));
    };


namespace Centernet{

static __global__ void decode_kernel(float* predict, int num_bboxes, int num_classes, float conf_T, float* invert_affine_matrix, float* parray, int max_objects);

void decode_kernel_invoker(float* predict, int num_boxes, int num_classes, 
            float conf_T, float* invert_affine_matrix, 
            float* parray, int max_objects, cudaStream_t stream);


};