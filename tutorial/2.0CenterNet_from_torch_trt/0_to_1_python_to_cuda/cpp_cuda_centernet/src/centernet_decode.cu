#include <common/cuda_tools.hpp>
#include <common/ilogger.hpp> // for using INFO()
#include <stdio.h>
#include <iostream>

using namespace std;

namespace Centernet{

    const int NUM_BOX_ELEMENT = 7; // left, top, right, bottom, confidence, class, keepflag


    static __device__ float sigmoid(float x){
        return 1 / (1 + exp(-x));
    }


    static __device__ void affine_project(float* matrix, float x, float y, float* ox, float* oy){
        *ox = matrix[0] * x + matrix[1] * y + matrix[2];
        *oy = matrix[3] * x + matrix[4] * y + matrix[5];
    }



    static __global__ void decode_kernel(float* predict, int num_bboxes, int num_channels, float conf_T, float* invert_affine_matrix, float* parray, int max_objects){
        int feat_h = 128;
        int feat_w = 128;
        float dp_ratio = 4.0; // downsampling ratio
        bool has_obj = false;
        int num_classes = 80;

        /* 
        On the right, we access a thread in GPU by only using blockIdx.x, blockDim.x and threadIdx.x for 
        simplicity, which indicates which block ,firstly, the gpu thread is in and within the given block, which thread the gpu thread
        is. In other words, we actually do an ndim-indexes-to-1-dim-index conversion for indexing a gpu thread.
         */
        int position = blockIdx.x * blockDim.x + threadIdx.x;
  
        if (position >= num_bboxes){
            printf("out of the edge\n");
            return;

        } 
        
        float* pitem_ptr = predict + num_channels * position;
        
        /*
        pitem_ptr is the addr of anyone of the pixels on the 128 x 128 2d feature map in python but
        in nature is a 16384 feature 1d array in c++.
        Along the pixel, we can access its relavant (2+2+80+80) infomation as follows:
            2: regxy
            2: wh
            80: hm
            80: pooled_hm
        
        Now (2+2+80+80) channels need to be anaylized to compute the bboxes.
        Steps can be summaried as the follows:
            - 1. for loop the hm and pool_hm channels to get the class with largest confidence (aka argmax).
                After that, a mask indicating the obj (y,x) location is produced, which is named as obj_kept_mask.
            - 2. During the for loop mentioned above, we add the x,y with regx and regy, which is followed using w and h to compute the x,y,r,b
                on featuremap. Finally, we do the inverse-downsampling(x4) to restore the real x,y,r,b.  
   
         */

        // printf("%f  %f\n",hm, pool_hm);

        float max_conf = 0.f;
        int label_with_max_conf = 0;
        

        auto hm_ptr = pitem_ptr + 4;
        auto pool_hm_ptr = pitem_ptr + 84;
        
        for (int i = 0; i<80; ++i, ++hm_ptr, ++pool_hm_ptr){ // for loop the class from 0 to class 79

            if ((*hm_ptr == *pool_hm_ptr) && (sigmoid(*hm_ptr) >= conf_T) ){
                has_obj = true;
                // printf("this pixel has an obj\n");

                /* 
                compute the bbox only when the pixel value is local maximum and its value
                is bigger than conf_T
                 */
                // predict class and the conf
                if (sigmoid(*hm_ptr) > max_conf) {
                    max_conf = sigmoid(*hm_ptr);
                    label_with_max_conf = i;
                };
            }


        }

        
        if (has_obj == true){
            // detect an object. parray should be added by 1.
            int index = atomicAdd(parray, 1);
            if (index >= max_objects) return;


            float x_ = position % feat_w + pitem_ptr[0]; // (y_,x_) is the center of an obj on the featuremap scale.
            float y_ = position / feat_w + pitem_ptr[1]; // the sybol here with the trailing underscore refers to featuremap scale.

            float w_ = pitem_ptr[2];
            float h_ = pitem_ptr[3];

            int cx = (int)(x_) * dp_ratio;
            int cy = (int)(y_) * dp_ratio;

            float left   = (float)((x_ - w_ * 0.5f) * dp_ratio);
            float right  = (float)((x_ + w_ * 0.5f) * dp_ratio);
            float top    = (float)((y_ - h_ * 0.5f) * dp_ratio);
            float bottom = (float)((y_ + h_ * 0.5f) * dp_ratio);  
            int label = label_with_max_conf;
            float confidence = max_conf;

           
        


            affine_project(invert_affine_matrix, left,  top,    &left,  &top); // modify the xyrb in place.
            affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

            

            printf("cxcy:%d, %d  ||  xyrb:%d, %d, %d, %d  ||  class:%d, conf:%f \n",
            cx, cy ,(int)left, (int)top, (int)right, (int)bottom, 
            label_with_max_conf, max_conf);

            
            float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
            *pout_item++ = left;
            *pout_item++ = top;
            *pout_item++ = right;
            *pout_item++ = bottom;
            *pout_item++ = confidence;
            *pout_item++ = label;
            *pout_item++ = 1; // keep = 1  ignore = 0

        } 
        else return;


    };


    void decode_kernel_invoker(float* predict, int num_bboxes, int num_channles, 
                float conf_T, float* invert_affine_matrix, 
                float* parray, int max_objects, cudaStream_t stream){
        
        int grid = 128;
        int block = 128;
        
        checkCudaKernel(decode_kernel<<<grid, block, 0, stream>>>(predict, num_bboxes, num_channles, conf_T, 
                                        invert_affine_matrix, parray, max_objects));

    };



} // namespace Centernet


