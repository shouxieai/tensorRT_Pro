

#include <cuda_runtime.h>

typedef unsigned char uint8_t;
static __device__ uint8_t cast(float value){
    return value < 0 ? 0 : (value > 255 ? 255 : value);
}

static __global__ void convert_nv12_to_bgr_kernel(
    const uint8_t* y, const uint8_t* uv, int width, int height, int linesize, uint8_t* dst_bgr, int edge
){
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    int ox = position % width;
    int oy = position / width;
    const uint8_t& yvalue = y[oy * linesize + ox];
    int offset_uv = (oy >> 1) * linesize + (ox & 0xFFFFFFFE);
    const uint8_t& u = uv[offset_uv + 0];
    const uint8_t& v = uv[offset_uv + 1];
    dst_bgr[position * 3 + 0] = cast(1.164f * (yvalue - 16.0f) + 2.018f * (u - 128.0f));
    dst_bgr[position * 3 + 1] = cast(1.164f * (yvalue - 16.0f) - 0.813f * (v - 128.0f) - 0.391f * (u - 128.0f));
    dst_bgr[position * 3 + 2] = cast(1.164f * (yvalue - 16.0f) + 1.596f * (v - 128.0f));
}

void convert_nv12_to_bgr_invoker(
    const uint8_t* y, const uint8_t* uv, int width, int height, int linesize, uint8_t* dst_bgr, 
    cudaStream_t stream
){
    int total = width * height;
    int block = total < 512 ? total : 512;
    int grid = (total + block - 1) / block;

    convert_nv12_to_bgr_kernel<<<grid, block, 0, stream>>>(
        y, uv, width, height, linesize,
        dst_bgr, total
    );
}