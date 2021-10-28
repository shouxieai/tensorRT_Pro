#ifndef SIMPLE_YOLO_HPP
#define SIMPLE_YOLO_HPP

/*
  简单的yolo接口，容易集成但是高性能
*/

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>

namespace SimpleYolo{

    using namespace std;

    enum class Type : int{
        V5 = 0,
        X  = 1
    };

    enum class Mode : int {
        FP32,
        FP16,
        INT8
    };

    struct Box{
        float left, top, right, bottom, confidence;
        int class_label;

        Box() = default;

        Box(float left, float top, float right, float bottom, float confidence, int class_label)
        :left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label){}
    };

    typedef std::vector<Box> BoxArray;

    class Infer{
    public:
        virtual shared_future<BoxArray> commit(const cv::Mat& image) = 0;
        virtual vector<shared_future<BoxArray>> commits(const vector<cv::Mat>& images) = 0;
    };

    const char* trt_version();
    const char* type_name(Type type);
    const char* mode_string(Mode type);
    void set_device(int device_id);

    /* 
        模型编译
        max batch size：为最大可以允许的batch数量
        source_onnx：仅仅只支持onnx格式输入
        saveto：储存的tensorRT模型，用于后续的加载
        max workspace size：最大工作空间大小，一般给1GB，在嵌入式可以改为256MB，单位是byte
        int8 images folder：对于Mode为INT8时，需要提供图像数据进行标定，请提供文件夹，会自动检索下面的jpg/jpeg/tiff/png/bmp
        int8_entropy_calibrator_cache_file：对于int8模式下，熵文件可以缓存，避免二次加载数据，可以跨平台使用，是一个txt文件
     */
    // 1GB = 1<<30
    bool compile(
        Mode mode, Type type,
        unsigned int max_batch_size,
        const string& source_onnx,
        const string& saveto,
        size_t max_workspace_size = 1<<30,
        const std::string& int8_images_folder = "",
        const std::string& int8_entropy_calibrator_cache_file = ""
    );

    shared_ptr<Infer> create_infer(const string& engine_file, Type type, int gpuid, float confidence_threshold=0.25f, float nms_threshold=0.5f);

}; // namespace SimpleYolo

#endif // SIMPLE_YOLO_HPP