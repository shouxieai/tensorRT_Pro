

#include <common/cuda_tools.hpp>

namespace YoloFast{

    static __host__ inline float desigmoid(float y){
        return -log(1.0f/y - 1.0f);
    }

    static __device__ inline float sigmoid(float x){
        return 1.0f / (1.0f + exp(-x));
    }

    static const int NUM_BOX_ELEMENT = 7;      // left, top, right, bottom, confidence, class, keepflag
    static __device__ inline void affine_project(float* matrix, float x, float y, float* ox, float* oy){
        *ox = matrix[0] * x + matrix[1];
        *oy = matrix[0] * y + matrix[2];
    }

    static __global__ void decode_kernel(
        float* predict, 
        int num_bboxes, 
        int fm_area,
        int num_classes, 
        float confidence_threshold,
        float deconfidence_threshold,  // desigmoid
        float* invert_affine_matrix, 
        float* parray, 
        const float* prior_box,
        int max_objects
    ){  
        int position = blockDim.x * blockIdx.x + threadIdx.x;
		if (position >= num_bboxes) return;

        // prior_box is 25200(axhxw) x 5
        // predict is 3 x 85 x 8400
        int anchor_index = position / fm_area;
        int fm_index     = position % fm_area;
        float* pitem     = predict + (anchor_index * (num_classes + 5) + 0) * fm_area + fm_index;
        float objectness = pitem[fm_area * 4];
        if(objectness < deconfidence_threshold)
            return;

        float confidence        = pitem[fm_area * 5];
        int label               = 0;
        for(int i = 1; i < num_classes; ++i){
            float class_confidence = pitem[fm_area * (i + 5)];
            if(class_confidence > confidence){
                confidence = class_confidence;
                label      = i;
            } 
        }

        confidence = sigmoid(confidence);
        objectness = sigmoid(objectness);
        confidence *= objectness;
        if(confidence < confidence_threshold)
            return;

        int index = atomicAdd(parray, 1);
        if(index >= max_objects)
            return;

        float predict_cx = sigmoid(pitem[fm_area * 0]);
        float predict_cy = sigmoid(pitem[fm_area * 1]);
        float predict_w  = sigmoid(pitem[fm_area * 2]);
        float predict_h  = sigmoid(pitem[fm_area * 3]);

        const float* prior_ptr = prior_box + position * 5;
        float stride     = prior_ptr[4];
        float cx         = (predict_cx * 2 - 0.5f + prior_ptr[0]) * stride;
        float cy         = (predict_cy * 2 - 0.5f + prior_ptr[1]) * stride;
        float width      = pow(predict_w * 2, 2.0f) * prior_ptr[2];
        float height     = pow(predict_h * 2, 2.0f) * prior_ptr[3];
        float left   = cx - width * 0.5f;
        float top    = cy - height * 0.5f;
        float right  = cx + width * 0.5f;
        float bottom = cy + height * 0.5f;
        affine_project(invert_affine_matrix, left,  top,    &left,  &top);
        affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

        float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
        *pout_item++ = left;
        *pout_item++ = top;
        *pout_item++ = right;
        *pout_item++ = bottom;
        *pout_item++ = confidence;
        *pout_item++ = label;
        *pout_item++ = 1; // 1 = keep, 0 = ignore
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

    void yolov5_decode_kernel_invoker(
        float* predict, 
        int num_bboxes, 
        int fm_area,
        int num_classes, 
        float confidence_threshold, 
        float nms_threshold, 
        float* invert_affine_matrix, 
        float* parray, 
        const float* prior_box,
        int max_objects, 
        cudaStream_t stream
    ){
        auto grid = CUDATools::grid_dims(num_bboxes);
        auto block = CUDATools::block_dims(num_bboxes);
        checkCudaKernel(decode_kernel<<<grid, block, 0, stream>>>(
            predict, 
            num_bboxes, 
            fm_area,
            num_classes, 
            confidence_threshold,
            desigmoid(confidence_threshold), 
            invert_affine_matrix, 
            parray, 
            prior_box,
            max_objects
        ));

        grid = CUDATools::grid_dims(max_objects);
        block = CUDATools::block_dims(max_objects);
        checkCudaKernel(nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold));
    }
};