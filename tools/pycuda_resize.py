# pip install pycuda==2019.1.2
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np

from pycuda.compiler import SourceModule
mod = SourceModule('''
#define INTER_RESIZE_COEF_BITS 11
#define INTER_RESIZE_COEF_SCALE (1 << INTER_RESIZE_COEF_BITS)
#define CAST_BITS (INTER_RESIZE_COEF_BITS << 1)

typedef unsigned char uint8_t;
static __inline__ __device__ int limit(int value, int low, int high){
    return value < low ? low : (value > high ? high : value);
}

static __inline__ __device__ int resize_cast(int value){
    return (value + (1 << (CAST_BITS - 1))) >> CAST_BITS;
}

__global__ void resize_bilinear_and_normalize_kernel(
    uint8_t* src, int src_line_size, int src_width, int src_height, 
    uint8_t* dst, int dst_line_size, int dst_width, int dst_height, 
    float sx, float sy, int edge
){
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    int dx      = position % dst_width;
    int dy      = position / dst_width;
    float src_x = (dx + 0.5f) * sx - 0.5f;
    float src_y = (dy + 0.5f) * sy - 0.5f;
    int y_low = floorf(src_y);
    int x_low = floorf(src_x);
    int y_high = limit(y_low + 1, 0, src_height - 1);
    int x_high = limit(x_low + 1, 0, src_width - 1);
    y_low = limit(y_low, 0, src_height - 1);
    x_low = limit(x_low, 0, src_width - 1);

    int ly    = rint((src_y - y_low) * INTER_RESIZE_COEF_SCALE);
    int lx    = rint((src_x - x_low) * INTER_RESIZE_COEF_SCALE);
    int hy    = INTER_RESIZE_COEF_SCALE - ly;
    int hx    = INTER_RESIZE_COEF_SCALE - lx;
    int w1    = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
    uint8_t* v1 = src + y_low * src_line_size + x_low * 3;
    uint8_t* v2 = src + y_low * src_line_size + x_high * 3;
    uint8_t* v3 = src + y_high * src_line_size + x_low * 3;
    uint8_t* v4 = src + y_high * src_line_size + x_high * 3;
    uint8_t* output_ptr = dst + dy * dst_line_size + dx * 3;
    output_ptr[0] = resize_cast(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0]);
    output_ptr[1] = resize_cast(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1]);
    output_ptr[2] = resize_cast(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2]);
}
''')
 
resize_bilinear_and_normalize_kernel = mod.get_function("resize_bilinear_and_normalize_kernel")

def resize(image, dsize):
        
    assert image.ndim == 3 and image.shape[2] == 3, "Image must be 3 channels"
    assert image.dtype == np.uint8, "Image.dtype must be np.uint8"
    
    sh, sw = image.shape[:2]
    dw, dh = dsize
    gpu_dst = np.empty((dh, dw, 3), np.uint8)

    jobs = dw * dh
    block = min(512, jobs)
    grid = (jobs + block - 1) // block
    resize_bilinear_and_normalize_kernel(
        drv.In(image), np.int32(image.strides[0]), np.int32(sw), np.int32(sh),
        drv.Out(gpu_dst), np.int32(gpu_dst.strides[0]), np.int32(dw), np.int32(dh),
        np.float32(sw / dw), np.float32(sh / dh), np.int32(jobs),
        block=(block, 1, 1), grid=(grid, 1))
    return gpu_dst