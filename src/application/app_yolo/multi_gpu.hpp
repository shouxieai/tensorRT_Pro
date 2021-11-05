#ifndef YOLO_MULTI_GPU_HPP
#define YOLO_MULTI_GPU_HPP

#include "yolo.hpp"

namespace Yolo{

    enum class DispatchMethod{
        Balanced     = 0,  /* commit阻塞，均衡的为每一个GPU分配任务，顺序的进行 */
        ThreadIDHash = 1,  /* commit阻塞，按照请求者的线程id，进行hash后选择gpu，同一个线程所有请求都送往同一个gpu */
        ThreadPool   = 2   /* commit不阻塞，线程池，commit函数立刻返回，由线程池内进行阻塞，好处是单个线程可以调度多
                              gpu。请保证请求的image不会立马修改 */
    };

    class MultiGPUInfer : public Yolo::Infer{};

    shared_ptr<MultiGPUInfer> create_multi_gpu_infer(
        const string& engine_file, Type type, const vector<int> gpuids, 
        DispatchMethod dispatch_method = DispatchMethod::Balanced,
        float confidence_threshold=0.25f, float nms_threshold=0.5f,
        NMSMethod nms_method = NMSMethod::FastGPU, int max_objects = 1024
    );
};


#endif // YOLO_MULTI_GPU_HPP