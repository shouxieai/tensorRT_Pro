
/*
 *  系统关于CUDA的功能函数
 */


#include "cuda_tools.hpp"

namespace CUDATools{
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

    bool check_device_id(int device_id){
        int device_count = -1;
        checkCudaRuntime(cudaGetDeviceCount(&device_count));
        if(device_id < 0 || device_id >= device_count){
            INFOE("Invalid device id: %d, count = %d", device_id, device_count);
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

    std::string device_capability(int device_id){
        cudaDeviceProp prop;
        checkCudaRuntime(cudaGetDeviceProperties(&prop, device_id));
        return iLogger::format("%d.%d", prop.major, prop.minor);
    }

    AutoDevice::AutoDevice(int device_id){

        cudaGetDevice(&old_);
        if(old_ != device_id && device_id != -1){
            checkCudaRuntime(cudaSetDevice(device_id));
            return;
        }

        CUcontext context = nullptr;
        cuCtxGetCurrent(&context);
        if(context == nullptr){
            checkCudaRuntime(cudaSetDevice(device_id));
            return;
        }
    }

    AutoDevice::~AutoDevice(){
        if(old_ != -1){
            checkCudaRuntime(cudaSetDevice(old_));
        }
    }
}