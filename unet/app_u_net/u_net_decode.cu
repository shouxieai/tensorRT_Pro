
#include <common/cuda_tools.hpp>
#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <memory>

namespace U_net{


    static __global__ void decode_kernel(
        float* predict, 
        int num_classes, 
        float* parray, int edge,     // parray 512，512，3     offset = origin + 512 * 3 * iy + i_y * 3 + 3;
        int in_width, int in_height,
        unsigned char* colors_ptr,
        int color_class_size
        ){  

        int max_index = 0;
        float temp = 0.0;
        int ix, iy;

        // 得到 当前的 position  // 512 * 512 
        int position = blockDim.x * blockIdx.x + threadIdx.x;
        // 如果越界返回
		if (position >= edge) return;

        ix = position % in_width;
        iy = position / in_width;

        int predict_area = in_height * in_width;       
        for(int iz=0; iz < color_class_size; ++iz){
            float* pitem = predict + predict_area * iz + iy * in_width + ix;
            if(*pitem > temp){
                temp = *pitem;
                max_index = iz;
                }
        }

        float* pdst_ptr = parray + in_width * 3 * iy + ix * 3;    
        

        for(int i=0; i < color_class_size; ++i){
            if (max_index == i){
                *(pdst_ptr + 0) = *(colors_ptr + 3 * i + 0);
                *(pdst_ptr + 1) = *(colors_ptr + 3 * i + 1);
                *(pdst_ptr + 2) = *(colors_ptr + 3 * i + 2);

            }
        }
        
    }

    

    void decode_kernel_u_net_invoker(
        float* predict,                          // 模型推理直接得到的输出
        int num_classes,                         
        float* parray,                           // 解码得到中间输出 和 加上warpAffine得到的最后输出
        cv::Size src_size, 
        int channels, 
        cudaStream_t stream){

        unsigned char colors[] = {0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 
            128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 
            64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 12};

        int color_class_size = (sizeof(colors) / sizeof(u_char) / 3) - 1;
        
        unsigned char* color_device = nullptr;

        size_t colors_bytes = sizeof(colors);

        cudaMalloc(&color_device, colors_bytes);

        cudaMemcpy(color_device, colors, colors_bytes, cudaMemcpyHostToDevice);

        int edge = src_size.area();

        int in_width = src_size.width;   // 网络输出的mask h 、 w
        int in_height = src_size.height;
        
        auto grid = CUDATools::grid_dims(edge);
        auto block = CUDATools::block_dims(edge);
        


        checkCudaKernel(decode_kernel<<<grid, block, 0, stream>>>(predict, num_classes, parray, edge, in_width, in_height, color_device, color_class_size));
        
        cudaFree(color_device);


    }

};