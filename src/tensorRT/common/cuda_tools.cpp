
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

    int current_device_id(){
        int device_id = 0;
        checkCudaRuntime(cudaGetDevice(&device_id));
        return device_id;
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

    std::string device_name(int device_id){
        cudaDeviceProp prop;
        checkCudaRuntime(cudaGetDeviceProperties(&prop, device_id));
        return prop.name;
    }

    std::string device_description(){

        cudaDeviceProp prop;
        size_t free_mem, total_mem;
        int device_id = 0;

        checkCudaRuntime(cudaGetDevice(&device_id));
        checkCudaRuntime(cudaGetDeviceProperties(&prop, device_id));
        checkCudaRuntime(cudaMemGetInfo(&free_mem, &total_mem));

        return iLogger::format(
            "[ID %d]<%s>[arch %d.%d][GMEM %.2f GB/%.2f GB]",
            device_id, prop.name, prop.major, prop.minor, 
            free_mem / 1024.0f / 1024.0f / 1024.0f,
            total_mem / 1024.0f / 1024.0f / 1024.0f
        );
    }

    AutoDevice::AutoDevice(int device_id){

        cudaGetDevice(&old_);
        checkCudaRuntime(cudaSetDevice(device_id));
    }

    AutoDevice::~AutoDevice(){
        checkCudaRuntime(cudaSetDevice(old_));
    }
}