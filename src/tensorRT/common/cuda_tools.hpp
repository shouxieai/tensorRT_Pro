#ifndef CUDA_TOOLS_HPP
#define CUDA_TOOLS_HPP


/*
 *  系统关于CUDA的功能函数
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include "ilogger.hpp"

#define GPU_BLOCK_THREADS  512


#define KernelPositionBlock											\
	int position = (blockDim.x * blockIdx.x + threadIdx.x);		    \
    if (position >= (edge)) return;


#define checkCudaRuntime(call) cuda::check_runtime(call, #call, __LINE__, __FILE__)

#define checkCudaKernel(...)                                                                         \
    __VA_ARGS__;                                                                                     \
    do{cudaError_t cudaStatus = cudaPeekAtLastError();                                               \
    if (cudaStatus != cudaSuccess){                                                                  \
        INFOE("launch failed: %s", cudaGetErrorString(cudaStatus));                                  \
    }} while(0);


#define Assert(op)					 \
	do{                              \
		bool cond = !(!(op));        \
		if(!cond){                   \
			INFOF("Assert failed, " #op);  \
		}                                  \
	}while(false)


struct CUstream_st;
typedef CUstream_st* ICUStream;

namespace cuda{
    bool check_runtime(cudaError_t e, const char* call, int iLine, const char *szFile);

    dim3 grid_dims(int numJobs);
    dim3 block_dims(int numJobs);
}


#endif // CUDA_TOOLS_HPP