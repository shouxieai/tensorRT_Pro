

#include <common/cuda_tools.hpp>

namespace RetinaFace{

    static const int NUM_BOX_ELEMENT = 16;
    __constant__ float variances[] = {0.1f, 0.2f};
    
    static __device__ void affine_project(float* matrix, float x, float y, float* ox, float* oy){
        *ox = matrix[0] * x + matrix[1] * y + matrix[2];
        *oy = matrix[3] * x + matrix[4] * y + matrix[5];
    }

    static __device__ float sigmoid(float x){
        return 1.0f / (1.0f + exp(-x));
    }

    static __global__ void decode_kernel(
        float* predict, int num_bboxes, float deconfidence_threshold, float nms_threshold, 
        float* invert_affine_matrix, float* parray, int max_objects, float* prior_array
    ){  
        int position = blockDim.x * blockIdx.x + threadIdx.x;
		if (position >= num_bboxes) return;

        float* pitem     = predict     + 16 * position;

        // cx, cy, w, h, neg_conf, pos_conf, landmark0.x, landmark0.y, landmark1.x, landmark1.y, landmark2.x, landmark2.y
        float neg_deconfidence = pitem[4];
        float pos_deconfidence = pitem[5];
        float object_deconfidence = (pos_deconfidence - neg_deconfidence);
        if(object_deconfidence < deconfidence_threshold)
            return;

        int index = atomicAdd(parray, 1);
        if(index >= max_objects)
            return;

        float* prior     = prior_array + 4  * position;
        float cx         = prior[0] + pitem[0] * variances[0] * prior[2];
        float cy         = prior[1] + pitem[1] * variances[0] * prior[3];
        float width      = prior[2] * exp(pitem[2] * variances[1]);
        float height     = prior[3] * exp(pitem[3] * variances[1]);
        float left   = cx - width  * 0.5f;
        float top    = cy - height * 0.5f;
        float right  = cx + width  * 0.5f;
        float bottom = cy + height * 0.5f;
        affine_project(invert_affine_matrix, left,  top,    &left,  &top);
        affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

        float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
        *pout_item++ = left;
        *pout_item++ = top;
        *pout_item++ = right;
        *pout_item++ = bottom;
        *pout_item++ = sigmoid(object_deconfidence);
        *pout_item++ = 1;   // 1=keep, 0=ignore

        float* landmark_predict = pitem + 6;
        for(int i = 0; i < 5; ++i){
            float x = prior[0] + landmark_predict[0] * variances[0] * prior[2];
            float y = prior[1] + landmark_predict[1] * variances[0] * prior[3];
            affine_project(invert_affine_matrix, x, y, pout_item + 0, pout_item + 1);
            pout_item        += 2;
            landmark_predict += 2;
        }
    }

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

        int position = (blockDim.x * blockIdx.x + threadIdx.x);
        int count = min((int)*bboxes, max_objects);
        if (position >= count) 
            return;
        
        // left, top, right, bottom, confidence, keepflag
        float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
        for(int i = 0; i < count; ++i){
            float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
            if(i == position) continue;

            if(pitem[4] >= pcurrent[4]){
                if(pitem[4] == pcurrent[4] && i < position)
                    continue;

                float iou = box_iou(
                    pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                    pitem[0],    pitem[1],    pitem[2],    pitem[3]
                );

                if(iou > threshold){
                    pcurrent[5] = 0;  // 1=keep, 0=ignore
                    return;
                }
            }
        }
    } 

    static float desigmoid(float x){
        return -log(1.0f / x - 1.0f);
    }

    void decode_kernel_invoker(
        float* predict, int num_bboxes, float confidence_threshold, float nms_threshold, 
        float* invert_affine_matrix, float* parray, int max_objects, float* prior,
        cudaStream_t stream
    ){
        auto grid = CUDATools::grid_dims(num_bboxes);
        auto block = CUDATools::block_dims(num_bboxes);
        checkCudaKernel(decode_kernel<<<grid, block, 0, stream>>>(
            predict, num_bboxes, desigmoid(confidence_threshold), nms_threshold, 
            invert_affine_matrix, parray, max_objects, prior
        ));

        grid = CUDATools::grid_dims(max_objects);
        block = CUDATools::block_dims(max_objects);
        checkCudaKernel(nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold));
    }
};