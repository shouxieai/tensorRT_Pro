

#include <onnxplugin/onnxplugin.hpp>
#include <common/cuda_tools.hpp>
#include <cublas_v2.h>
#include <cuda_fp16.h>

using namespace ONNXPlugin;

#define cublasCheck(op)														 \
do {																	 \
    auto ret = (op);													 \
    if (ret != CUBLAS_STATUS_SUCCESS) {											 \
        INFOF("%s fail, %d != %d", #op, ret, CUBLAS_STATUS_SUCCESS);				 \
    }																	 \
} while (0);


__global__ void sigmoidKernel(float* input, float* output, int edge) {

    KernelPositionBlock;
    output[position] = 1 / (1 + exp(-input[position]));
}

// __global__ void sigmoidKernel(__half* input, __half* output, int edge) {

//     KernelPositionBlock;
//     __half one = 1.0f;
//     output[position] = one / (one + hexp(-input[position]));
// }

static __device__ float dmcnIm2colBilinear(const float *bottom_data, const int data_width,
    const int height, const int width, float h, float w)
{
    int h_low = floor(h);
    int w_low = floor(w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    float lh = h - h_low;
    float lw = w - w_low;
    float hh = 1 - lh, hw = 1 - lw;

    float v1 = 0;
    if (h_low >= 0 && w_low >= 0)
        v1 = bottom_data[h_low * data_width + w_low];
    float v2 = 0;
    if (h_low >= 0 && w_high <= width - 1)
        v2 = bottom_data[h_low * data_width + w_high];
    float v3 = 0;
    if (h_high <= height - 1 && w_low >= 0)
        v3 = bottom_data[h_high * data_width + w_low];
    float v4 = 0;
    if (h_high <= height - 1 && w_high <= width - 1)
        v4 = bottom_data[h_high * data_width + w_high];

    float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

    float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}

// static __device__ __half dmcnIm2colBilinear(const __half *bottom_data, const int data_width,
//     const int height, const int width, __half h, __half w)
// {
//     int h_low = hfloor(h);
//     int w_low = hfloor(w);
//     int h_high = h_low + 1;
//     int w_high = w_low + 1;

//     __half one = 1.0f;
//     __half h_low_hf = h_low;
//     __half w_low_hf = w_low;
//     __half lh = h - h_low_hf;
//     __half lw = w - w_low_hf;
//     __half hh = one - lh, hw = one - lw;

//     __half zero = 0.0f;
//     __half v1 = zero;
//     if (h_low >= 0 && w_low >= 0)
//         v1 = bottom_data[h_low * data_width + w_low];
//     __half v2 = zero;
//     if (h_low >= 0 && w_high <= width - 1)
//         v2 = bottom_data[h_low * data_width + w_high];
//     __half v3 = zero;
//     if (h_high <= height - 1 && w_low >= 0)
//         v3 = bottom_data[h_high * data_width + w_low];
//     __half v4 = zero;
//     if (h_high <= height - 1 && w_high <= width - 1)
//         v4 = bottom_data[h_high * data_width + w_high];

//     __half w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
//     return (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
// }

__global__ void DCNIm2colKernel(
    const float *data_input, const float *data_offset, const float *data_mask,
    const int height_input, const int width_input, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group,
    const int batch_size, const int num_channels, const int deformable_group,
    const int height_output, const int width_output,
    float *data_output, int edge)
{
    KernelPositionBlock;

    const int f_area_input = width_input * height_input;
    const int f_area_output = width_output * height_output;

    // index index of output matrix
    const int w_output = position % width_output;
    const int h_output = (position / width_output) % height_output;
    const int c_input = (position / width_output / height_output) % num_channels;

    const int c_output = c_input * kernel_h * kernel_w;
    const int deformable_group_index = c_input / channel_per_deformable_group;
    const int h_input = h_output * stride_h - pad_h;
    const int w_input = w_output * stride_w - pad_w;

    int data_output_offset = c_input * kernel_h * kernel_w * f_area_output + h_output * width_output + w_output;
    float *data_output_ptr = data_output + data_output_offset;
    const float *data_input_ptr = data_input + c_input * f_area_input;
    const float *data_offset_ptr = data_offset + deformable_group_index * 2 * kernel_h * kernel_w * f_area_output;
    const float *data_mask_ptr = data_mask + deformable_group_index * kernel_h * kernel_w * f_area_output;

    for (int i = 0; i < kernel_h; ++i)
    {
        for (int j = 0; j < kernel_w; ++j)
        {
            const int row = i + h_input;
            const int col = j + w_input;
            const int kernel_index = i * kernel_w + j;

            const int offset_h_offset = 2 * kernel_index * f_area_output + h_output * width_output + w_output;
            const int offset_w_offset = (2 * kernel_index + 1) * f_area_output + h_output * width_output + w_output;
            const int mask_offset = kernel_index * f_area_output + h_output * width_output + w_output;

            const float offset_h = data_offset_ptr[offset_h_offset];
            const float offset_w = data_offset_ptr[offset_w_offset];
            const float mask = data_mask_ptr[mask_offset];

            float val = 0;
            const float h_im = h_input + i * dilation_h + offset_h;
            const float w_im = w_input + j * dilation_w + offset_w;
            if (h_im > -1 && w_im > -1 && h_im < height_input && w_im < width_input)
            {
                val = dmcnIm2colBilinear(data_input_ptr, width_input, height_input, width_input, h_im, w_im);
            }
            *data_output_ptr = val * mask;
            data_output_ptr += f_area_output;
        }
    }
}

// __global__ void DCNIm2colKernel(
//     const __half *data_input, const __half *data_offset, const __half *data_mask,
//     const int height_input, const int width_input, const int kernel_h, const int kernel_w,
//     const int pad_h, const int pad_w,
//     const int stride_h, const int stride_w,
//     const int dilation_h, const int dilation_w,
//     const int channel_per_deformable_group,
//     const int batch_size, const int num_channels, const int deformable_group,
//     const int height_output, const int width_output,
//     __half *data_output, int edge)
// {
//     KernelPositionBlock;

//     const int f_area_input = width_input * height_input;
//     const int f_area_output = width_output * height_output;

//     // index index of output matrix
//     const int w_output = position % width_output;
//     const int h_output = (position / width_output) % height_output;
//     const int c_input = (position / width_output / height_output) % num_channels;

//     const int c_output = c_input * kernel_h * kernel_w;
//     const int deformable_group_index = c_input / channel_per_deformable_group;
//     const int h_input = h_output * stride_h - pad_h;
//     const int w_input = w_output * stride_w - pad_w;

//     __half width_input_hf = __float2half(width_input);
//     __half height_input_hf = __float2half(height_input);

//     __half h_input_hf = __float2half(h_input);
//     __half w_input_hf = __float2half(w_input);
//     __half dilation_h_hf = __float2half(dilation_h);
//     __half dilation_w_hf = __float2half(dilation_w);

//     int data_output_offset = c_input * kernel_h * kernel_w * f_area_output + h_output * width_output + w_output;
//     __half *data_output_ptr = data_output + data_output_offset;
//     const __half *data_input_ptr = data_input + c_input * f_area_input;
//     const __half *data_offset_ptr = data_offset + deformable_group_index * 2 * kernel_h * kernel_w * f_area_output;
//     const __half *data_mask_ptr = data_mask + deformable_group_index * kernel_h * kernel_w * f_area_output;

//     __half n_one = -1.0f;
//     __half zero = 0.0f;

//     for (int i = 0; i < kernel_h; ++i)
//     {
//         for (int j = 0; j < kernel_w; ++j)
//         {
//             __half i_hf = __float2half(i);
//             __half j_hf = __float2half(j);
//             const int row = i + h_input;
//             const int col = j + w_input;
//             const int kernel_index = i * kernel_w + j;

//             const int offset_h_offset = 2 * kernel_index * f_area_output + h_output * width_output + w_output;
//             const int offset_w_offset = (2 * kernel_index + 1) * f_area_output + h_output * width_output + w_output;
//             const int mask_offset = kernel_index * f_area_output + h_output * width_output + w_output;

//             const __half offset_h = data_offset_ptr[offset_h_offset];
//             const __half offset_w = data_offset_ptr[offset_w_offset];
//             const __half mask = data_mask_ptr[mask_offset];

//             __half val = zero;
//             __half h_im = h_input_hf + i_hf * dilation_h_hf + offset_h;
//             __half w_im = w_input_hf + j_hf * dilation_w_hf + offset_w;

//             if (h_im > n_one && w_im > n_one && h_im < height_input_hf && w_im < width_input_hf)
//             {
//                 val = dmcnIm2colBilinear(data_input_ptr, width_input_hf, height_input_hf, width_input_hf, h_im, w_im);
//             }
//             *data_output_ptr = val * mask;
//             data_output_ptr += f_area_output;
//         }
//     }
// }

template<typename DataType>
static __global__ void biasKernel(DataType* data_input, const DataType* bias, const int f_area, int edge) {

    KernelPositionBlock;
    int bias_index = position / f_area;
    data_input[position] += bias[bias_index];
}

inline void segemm_native(cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    float alpha, /* host or device pointer */
    const float *A,
    int lda,
    const float *B,
    int ldb,
    float beta, /* host or device pointer */
    float *C,
    int ldc) {
    cublasCheck(cublasSgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
    //cublasCheck(cublasGemmEx(handle, transa, transb, m, n, k, &alpha, A, CUDA_R_32F, lda, B, CUDA_R_32F, ldb, &beta, C, CUDA_R_32F, ldc, CUDA_R_32F, CUBLAS_GEMM_DFALT));
}

// inline void segemm_native(cublasHandle_t handle,
//     cublasOperation_t transa,
//     cublasOperation_t transb,
//     int m,
//     int n,
//     int k,
//     float alpha,
//     const __half *A,
//     int lda,
//     const __half *B,
//     int ldb,
//     float beta, 
//     __half *C,
//     int ldc) {

//     auto halpha = __float2half(alpha);
//     auto hbeta  = __float2half(beta);
//     //cublasCheck(cublasHgemm(handle, transa, transb, m, n, k, &halpha, A, lda, B, ldb, &hbeta, C, ldc));
//     cublasCheck(cublasGemmEx(handle, transa, transb, m, n, k, &halpha, A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb, &hbeta, C, CUDA_R_16F, ldc, CUDA_R_16F, CUBLAS_GEMM_DFALT));
// }

template<typename DataType>
static void enqueue_native(cublasHandle_t handle, const std::vector<GTensor>& inputs, std::vector<GTensor>& outputs, const std::vector<GTensor>& weights, void* workspace, cudaStream_t stream) {
    auto& data = inputs[0];
    auto& om = inputs[1];
    auto& out = outputs[0];

    int kernel_size = weights[0].width();
    int deformable_group = 1;
    size_t maskSize = (size_t)data.height() * data.width() * kernel_size * kernel_size * deformable_group;
    size_t im2colSize = (size_t)data.channel() * kernel_size * kernel_size * out.height() * out.width();

    const int m = out.channel();
    const int n = out.count(2);
    const int k = data.channel() * kernel_size * kernel_size;
    float alpha = 1.0;
    float beta = 0.0;

    //cublasCheck(cublasSetStream(handle, stream));
    for (int ibatch = 0; ibatch < data.batch(); ++ibatch) {
        DataType* maskWorkspacePtr = (DataType*)workspace + (maskSize + im2colSize) * ibatch;
        DataType* im2colWorkspacePtr = (DataType*)workspace + (maskSize + im2colSize) * ibatch + maskSize;

        DataType* inputMask = om.ptr<DataType>(ibatch, om.channel() / 3 * 2);
        checkCudaKernel(
            sigmoidKernel<<<CUDATools::grid_dims(maskSize), CUDATools::block_dims(maskSize), 0, stream>>>(inputMask, maskWorkspacePtr, maskSize);
        );

        DataType* datainput = data.ptr<DataType>(ibatch);
        DataType* offset = om.ptr<DataType>(ibatch);

        auto jobs = (size_t)data.channel() * out.height() * out.width();
        checkCudaKernel(
            DCNIm2colKernel<<<CUDATools::grid_dims(jobs), CUDATools::block_dims(jobs), 0, stream>>>(
                datainput, offset, maskWorkspacePtr, data.height(), data.width(), kernel_size, kernel_size, 1, 1, 1, 1, 1, 1, data.channel(), data.batch(), data.channel(), deformable_group,
                out.height(), out.width(), im2colWorkspacePtr, jobs
            );
        );

        DataType* weightKernel = weights[0].ptr<DataType>();
        segemm_native(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha, im2colWorkspacePtr, n, weightKernel, k, beta, out.ptr<DataType>(ibatch), n);

        if (weights.size() > 1) {
            DataType* weightBias = weights[1].ptr<DataType>();
            size_t edge = out.count(1);
            size_t area = out.count(2);

            checkCudaKernel(
                biasKernel<<<CUDATools::grid_dims(edge), CUDATools::block_dims(edge), 0, stream>>>(
                    out.ptr<DataType>(ibatch), weightBias, area, edge
                );
            );
        }
    }
}


class DCNv2 : public TRTPlugin {
public:
    cublasHandle_t cublasHandle_ = nullptr;
    SetupPlugin(DCNv2);

    virtual void attachToContext(cudnnContext* /*cudnn*/, cublasContext* cublas, nvinfer1::IGpuAllocator* /*allocator*/) noexcept override{
        cublasHandle_ = cublas;
    }

    virtual void detachFromContext() noexcept override{
        cublasHandle_ = nullptr;
    }

    std::shared_ptr<LayerConfig> new_config() {
        auto cfg = TRTPlugin::new_config();

        //cfg->supportDataType_ = {nvinfer1::DataType::kFLOAT};
        //cfg->supportDataType_ = {nvinfer1::DataType::kHALF, nvinfer1::DataType::kFLOAT};
        cfg->support_dtype_set_ = {nvinfer1::DataType::kFLOAT};
        return cfg;
    }

    virtual void config_finish() override{
        
        // INFO("weights = %d", config_->weights_.size());
        // for(int i = 0; i < config_->weights_.size(); ++i){
        // 	auto& w = config_->weights_[i];
        // 	if(w->type() == TRT::DataType::Float16){
        // 		INFO("Weight[%d] shape is %s, dtype = %s, value[0] = %f", i, w->shape_string(), data_type_string(w->type()), float(w->at<__half>(0)));
        // 	}else{
        // 		INFO("Weight[%d] shape is %s, dtype = %s, value[0] = %f", i, w->shape_string(), data_type_string(w->type()), w->at<float>(0));
        // 	}
        // }
    }

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs, const nvinfer1::PluginTensorDesc* outputs,
        int32_t nbOutputs) const noexcept{
            
        int kernel_size = 3;
        int deformable_group = 1;
        size_t im2colSize = (size_t)inputs[0].dims.d[1] * kernel_size * kernel_size * outputs[0].dims.d[2] * outputs[0].dims.d[3];
        size_t maskSize = (size_t)inputs[0].dims.d[2] * inputs[0].dims.d[3] * kernel_size * kernel_size * deformable_group;
        config_->workspace_size_ = (im2colSize + maskSize) * config_->max_batch_size_ * TRT::data_type_size(config_->usage_dtype_);
        return config_->workspace_size_;
    }

    nvinfer1::DimsExprs getOutputDimensions(
        int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept{

        nvinfer1::DimsExprs output_dims;
        output_dims.nbDims = 4;
        output_dims.d[0] = inputs[0].d[0];
        output_dims.d[1] = exprBuilder.constant(config_->weights_[0]->size(0));
        output_dims.d[2] = inputs[0].d[2];
        output_dims.d[3] = inputs[0].d[3];
        return output_dims;
    }

    int enqueue(const std::vector<GTensor>& inputs, std::vector<GTensor>& outputs, const std::vector<GTensor>& weights, void* workspace, cudaStream_t stream) {
        
        if (config_->usage_dtype_ == TRT::DataType::Float) {
            enqueue_native<float>(cublasHandle_, inputs, outputs, weights, workspace, stream);
        }
        else if (config_->usage_dtype_ == TRT::DataType::Float16) {
            // enqueue_native<__half>(cublasHandle_, inputs, outputs, weights, workspace, stream);
            INFOF("not implement function");
        }
        else{
            INFOF("not implement function");
        }
        return 0;
    }
};

RegisterPlugin(DCNv2);