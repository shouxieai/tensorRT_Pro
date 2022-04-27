#ifndef PREPROCESS_KERNEL_CUH
#define PREPROCESS_KERNEL_CUH

#include "cuda_tools.hpp"

namespace CUDAKernel{

    enum class NormType : int{
        None      = 0,
        MeanStd   = 1,
        AlphaBeta = 2
    };

    enum class ChannelType : int{
        None          = 0,
        Invert        = 1
    };

    struct Norm{
        float mean[3];
        float std[3];
        float alpha, beta;
        NormType type = NormType::None;
        ChannelType channel_type = ChannelType::None;

        // out = (x * alpha - mean) / std
        static Norm mean_std(const float mean[3], const float std[3], float alpha = 1/255.0f, ChannelType channel_type=ChannelType::None);

        // out = x * alpha + beta
        static Norm alpha_beta(float alpha, float beta = 0, ChannelType channel_type=ChannelType::None);

        // None
        static Norm None();
    };

    void resize_bilinear_and_normalize(
		uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height,
		const Norm& norm,
		cudaStream_t stream);

    void warp_affine_bilinear_and_normalize_plane(
        uint8_t* src, int src_line_size, int src_width, int src_height, 
        float* dst  , int dst_width, int dst_height,
        float* matrix_2_3, uint8_t const_value, const Norm& norm,
        cudaStream_t stream);

    void warp_affine_bilinear_and_normalize_focus(
        uint8_t* src, int src_line_size, int src_width, int src_height, 
        float* dst  , int dst_width, int dst_height,
        float* matrix_2_3, uint8_t const_value, const Norm& norm,
        cudaStream_t stream);

    // 可以用来图像校正、图像旋转等等 (测试比cpu快10倍以上)
    // 使用示范:
    // float* matrix_3_3 = nullptr;
    // size_t matrix_bytes = 3 * 3 * sizeof(f32);
    // checkCudaRuntime(cudaMalloc(&matrix_3_3, matrix_bytes));
    // checkCudaRuntime(cudaMemset(matrix_3_3, 0,  matrix_bytes));
    //
    // #左上、右上、右下、左下 原图像四个点的坐标
    //    cv::Point2f src_points[] = { 
    //    vctvctPoints[nImageIdx][0],
    //    vctvctPoints[nImageIdx][1],
    //    vctvctPoints[nImageIdx][2],
    //    vctvctPoints[nImageIdx][3]};
    // 
    // #左上、右上、左下、右下（Z 字形排列） 目标图像四个点的坐标
    //    cv::Point2f dst_points[] = {
    //        cv::Point2f(0, 0),
    //        cv::Point2f(nw-1, 0),
    //        cv::Point2f(0, nh-1),
    //        cv::Point2f(nw-1, nh-1) };
    // 利用opencv 得到变换矩阵  dst -> src 的 矩阵
    //    cv::Mat Perspect_Matrix = cv::getPerspectiveTransform(dst_points, src_points);
    //    Perspect_Matrix.convertTo(Perspect_Matrix,  CV_32FC1);
    // 拷贝到 gpu 
    //    checkCudaRuntime(cudaMemcpy(matrix_3_3, Perspect_Matrix.data, matrix_bytes, cudaMemcpyHostToDevice));
    void warp_perspective(
        uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height,
		float* matrix_3_3, uint8_t const_value, const Norm& norm, cudaStream_t stream
    );

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