#include <common/cuda_tools.hpp>
#include <common/ilogger.hpp>
#include <stdio.h>
#include <iostream>

using namespace std;

namespace CenterNet{

    static const int NUM_BOX_ELEMENT = 7; // left, top, right, bottom, confidence, class, keepflag

    static __host__ float desigmoid(float y){
        return -log(1.0f / y - 1.0f);
    }

    static __device__ float sigmoid(float x){
        return 1.0f / (1.0f + exp(-x));
    }

    static __device__ void affine_project(float* matrix, float x, float y, float* ox, float* oy){
        *ox = matrix[0] * x + matrix[1] * y + matrix[2];
        *oy = matrix[3] * x + matrix[4] * y + matrix[5];
    }

    static __global__ void decode_kernel(
        float* predict, int num_bboxes, int num_channels, int num_classes, 
        int fm_width, int fm_height, int stride,
        float conf_T, float deconf_T, float* invert_affine_matrix, float* parray, int max_objects
    ){
        /* 
        On the right, we access a thread in GPU by only using blockIdx.x, blockDim.x and threadIdx.x for 
        simplicity, which indicates which block ,firstly, the gpu thread is in and within the given block, which thread the gpu thread
        is. In other words, we actually do an ndim-indexes-to-1-dim-index conversion for indexing a gpu thread.
         */
        int position = blockIdx.x * blockDim.x + threadIdx.x;
        if (position >= num_bboxes)
            return;
        
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

        auto hm_ptr = pitem_ptr + 4;
        auto pool_hm_ptr = pitem_ptr + (num_classes + 4);
        float max_conf = *hm_ptr;
        int label_with_max_conf = 0;
        bool has_obj = false;
        
        for (int i = 0; i < num_classes; ++i, ++hm_ptr, ++pool_hm_ptr){ // for loop the class from 0 to class 79

            float hm_conf   = *hm_ptr;
            float pool_conf = *pool_hm_ptr;

            if ((hm_conf == pool_conf) && (hm_conf >= deconf_T) ){
                has_obj = true;
                /* 
                compute the bbox only when the pixel value is local maximum and its value
                is bigger than conf_T
                 */
                // predict class and the conf
                if (hm_conf > max_conf) {
                    max_conf = hm_conf;
                    label_with_max_conf = i;
                };
            }
        }

        if(!has_obj)
            return;
        
        // detect an object. parray should be added by 1.
        int index = atomicAdd(parray, 1);
        if (index >= max_objects) return;

        float x_ = (position % fm_width) + pitem_ptr[0]; // (y_,x_) is the center of an obj on the featuremap scale.
        float y_ = (position / fm_width) + pitem_ptr[1]; // the sybol here with the trailing underscore refers to featuremap scale.
        float w_ = pitem_ptr[2];
        float h_ = pitem_ptr[3];

        float left   = (x_ - w_ * 0.5f) * stride;
        float right  = (x_ + w_ * 0.5f) * stride;
        float top    = (y_ - h_ * 0.5f) * stride;
        float bottom = (y_ + h_ * 0.5f) * stride;  
        int label    = label_with_max_conf;
        float confidence = sigmoid(max_conf);

        affine_project(invert_affine_matrix, left,  top,    &left,  &top); // modify the xyrb in place.
        affine_project(invert_affine_matrix, right, bottom, &right, &bottom);
        
        float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
        *pout_item++ = left;
        *pout_item++ = top;
        *pout_item++ = right;
        *pout_item++ = bottom;
        *pout_item++ = confidence;
        *pout_item++ = label;
        *pout_item++ = 1; // keep = 1  ignore = 0
    };

    static __device__ float box_iou(
        float aleft, float atop, float aright, float abottom, 
        float bleft, float btop, float bright, float bbottom
    ){

        float cleft 	= max(aleft, bleft);
        float ctop 		= max(atop, btop);
        float cright 	= min(aright, bright);
        float cbottom 	= min(abottom, bbottom);
        
        float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
        if(c_area == 0.0f)
            return 0.0f;
        
        float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
        float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
        return c_area / (a_area + b_area - c_area);
    }

    static __global__ void nms_kernel(float* bboxes, int max_objects, float threshold){
        // refer to tensorRT_cpp/tutorial/2.0CenterNet_from_torch_trt/nms_cuda.jpg and comments.jpg for understanding.
        int position = (blockDim.x * blockIdx.x + threadIdx.x);
        int count = min((int)*bboxes, max_objects);
        if (position >= count) 
            return;
        
        // left, top, right, bottom, confidence, class, keepflag
        float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
        for(int i = 0; i < count; ++i){
            float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
            if(i == position || pcurrent[5] != pitem[5]) continue;

            if(pitem[4] >= pcurrent[4]){
                if(pitem[4] == pcurrent[4] && i < position)
                    continue;

                float iou = box_iou(
                    pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                    pitem[0],    pitem[1],    pitem[2],    pitem[3]
                );

                if(iou > threshold){
                    pcurrent[6] = 0;  // 1=keep, 0=ignore
                    return;
                }
            }
        }
    } 

    void decode_kernel_invoker(float* predict, int num_bboxes, int num_channles, int num_classes,
                int fm_width, int fm_height, int stride,
                float conf_T, float nms_threshold, float* invert_affine_matrix, 
                float* parray, int max_objects, cudaStream_t stream){
        
        auto grid = CUDATools::grid_dims(num_bboxes);
        auto block = CUDATools::block_dims(num_bboxes);
                
        checkCudaKernel(decode_kernel<<<grid, block, 0, stream>>>(predict, 
            num_bboxes, num_channles, num_classes, fm_width, fm_height, stride, conf_T, desigmoid(conf_T),
            invert_affine_matrix, parray, max_objects));

        grid = CUDATools::grid_dims(max_objects);
        block = CUDATools::block_dims(max_objects);
        checkCudaKernel(nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold));
    };
} // namespace Centernet
