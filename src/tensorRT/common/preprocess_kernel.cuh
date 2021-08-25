#ifndef PREPROCESS_KERNEL_CUH
#define PREPROCESS_KERNEL_CUH

#include <common/cuda_tools.hpp>

namespace CUDAKernel{

    //            CHANNEL_ORDER   TYPE
    // 0x00  00       FF           FF
    enum class NormType : unsigned int{
        None      = 0,
        MeanStd   = 1,
        AlphaBeta = 2,
        InvertChannel = 1 << 8,
        ToRGB     = InvertChannel
    };

    struct Norm{
        float mean[3];
        float std[3];
        float alpha, beta;
        NormType type = NormType::None;

        // out = (x * alpha - mean) / std
        static Norm mean_std(float mean[3], float std[3], float alpha = 1/255.0f);

        // out = x * alpha + beta
        static Norm alpha_beta(float alpha, float beta = 0);

        Norm operator + (NormType t){
            Norm out = *this;
            out.type = NormType((unsigned int)out.type | (unsigned int)t);
            return out;
        }
    };

    void warp_affine_bilinear_and_normalize(
        uint8_t* src, int src_line_size, int src_width, int src_height, 
        float* dst  , int dst_width, int dst_height,
        float* matrix_2_3, uint8_t const_value, const Norm& norm,
        cudaStream_t stream);

    void norm_feature(
        float* feature_array, int num_feature, int feature_length,
        cudaStream_t stream
    );

    void convert_nv12_to_bgr_invoke(
        const uint8_t* y, const uint8_t* uv, int width, int height, 
        int linesize, uint8_t* dst, 
        cudaStream_t stream);
};

#endif // PREPROCESS_KERNEL_CUH