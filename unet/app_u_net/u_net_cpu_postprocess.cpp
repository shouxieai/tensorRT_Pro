#include <iostream>
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include <common/infer_controller.hpp>
#include <common/preprocess_kernel.cuh>
#include <common/monopoly_allocator.hpp>
#include <common/cuda_tools.hpp>
#include <vector>
#include <opencv2/opencv.hpp>

namespace U_net{

using namespace std;
using namespace cv;

vector<int> colors = {0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 
            128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 
            64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 12};

void decode_kernel_invoker(
        std::shared_ptr<TRT::Tensor> predict, int num_classes, float* parray
    ){
        int o_h = predict->shape(1);
        int o_w = predict->shape(2);
        int num = predict->shape(0);
        printf("o_h is %d , o_w is %d, batch is %d\n", o_h, o_w, num);
        int max_index = 0; 
        for(int i = 0; i < o_h; ++i){            // cols  512   511
            for(int j = 0; j < o_w; ++j){        // rows  512
                for(int k = 0; k < num; ++k){    // channels   

                    if (*(predict->cpu<float>(k, i, j)) > *(predict->cpu<float>(max_index, i, j)))
                        max_index = k;

                    if (k == num - 1){
                        *(parray + j + i * o_h) = max_index;
                        max_index = 0;
                        }
                }                 
            } 
        }

        printf("核函数处理完成！\n");
        
    }

// 老方案
Mat add_mask(float* parray){                  // parray  是 一个tensor 512 * 512 的指针，指向首地址 
    Mat output = Mat::zeros(512, 512, CV_8UC3);
    printf("vector's size is  %d \n", colors.size());  // 66

    for (int i = 0; i < 512; i++){      // heights
        for(int j = 0; j < 512; j++)    // widths  
            {   
            //   循环 512 * 512 次
            // printf("test %d \n", sizeof(colors)/sizeof(int)/3);
            for (int c=0; c < colors.size()/3 - 1; ++c)    // 循环 21 次
                {    // 取 512 * 512 的 tensor 的值 和 colors 的 index 进行比较， 相同则改写 output
                if(*(parray + j + i * 512) == c){
                    auto& pixel = output.at<Vec3i>(i,j);
                    pixel[0] = colors.at(3 * c);
                    pixel[1] = colors.at(3 * c + 1);
                    pixel[2] = colors.at(3 * c + 2);
                }
            }
        }
    }
    printf("%d, %d \n", output.cols, output.rows);
    return output;
}


Mat new_mask(shared_ptr<TRT::Tensor> mask_ptr, float* affine_matrix_ptr, Size size_ori){                  // parray  是 一个tensor 512 * 512 的指针，指向首地址 
    Mat output = Mat::zeros(512, 512, CV_8UC3);                  // 8u  是 unsigned char
    printf("%d  %d \n", mask_ptr -> shape(0), mask_ptr -> shape(1));  // 66

    for (int i = 0; i < mask_ptr -> shape(0); i++){      // heights
        for(int j = 0; j < mask_ptr -> shape(1); j++)    // widths  
            {   
            for (int c=0; c < colors.size()/3 - 1; ++c)    // 循环 21 次
                {    // 取 512 * 512 的 tensor 的值 和 colors 的 index 进行比较， 相同则改写 output
                if(int(*(mask_ptr -> cpu<float>(i, j))) == c){
                    auto& pixel = output.at<Vec3b>(i,j);  
                    pixel[0] = colors.at(3 * c);
                    pixel[1] = colors.at(3 * c + 1);
                    pixel[2] = colors.at(3 * c + 2);
                }
            }
        }
    }
    
    float affine_mat[] = {affine_matrix_ptr[0], affine_matrix_ptr[1], affine_matrix_ptr[2], 
                affine_matrix_ptr[3], affine_matrix_ptr[4], affine_matrix_ptr[5]};
    for(int i=0; i<6; ++i)
        printf("affine[%d] is %f\n", i, affine_matrix_ptr[i]);
    Mat affine_matrix(2, 3, CV_32F, affine_mat);
    Mat out_mat(1080, 800, 3); 
    warpAffine(output, out_mat, affine_matrix, size_ori);
    printf("%d, %d \n", output.cols, output.rows);
    
    return out_mat;
}
    
}


