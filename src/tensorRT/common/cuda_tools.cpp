
/*
 *  系统关于CUDA的功能函数
 */


#include "cuda_tools.hpp"

namespace cuda{
    bool check_driver(CUresult e, const char* call, int line, const char *file) {
        if (e != CUDA_SUCCESS) {

            const char* message = nullptr;
            const char* name = nullptr;
            cuGetErrorString(e, &message);
            cuGetErrorName(e, &name);
            INFOE("CUDA Driver error %s # %s, code = %s [ %d ] in file %s:%d", call, message, name, e, file, line);
            return false;
        }
        return true;
    }

    bool check_runtime(cudaError_t e, const char* call, int line, const char *file){
        if (e != cudaSuccess) {
            INFOE("CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d", call, cudaGetErrorString(e), cudaGetErrorName(e), e, file, line);
            return false;
        }
        return true;
    }

    dim3 grid_dims(int numJobs) {
        int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
        return dim3(((numJobs + numBlockThreads - 1) / (float)numBlockThreads));
    }

    dim3 block_dims(int numJobs) {
        return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
    }
}