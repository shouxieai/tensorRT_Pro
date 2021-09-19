#ifndef YOLO_HPP
#define YOLO_HPP

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>
#include <common/trt_tensor.hpp>
#include <common/object_detector.hpp>

/**
 * @brief 发挥极致的性能体验
 * 支持YoloX和YoloV5
 */
namespace Yolo{

    using namespace std;
    using namespace ObjectDetector;

    enum class Type : int{
        V5 = 0,
        X  = 1
    };    

    void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, Type type, int ibatch);

    class Infer{
    public:
        virtual shared_future<BoxArray> commit(const cv::Mat& image) = 0;
        virtual vector<shared_future<BoxArray>> commits(const vector<cv::Mat>& images) = 0;
    };

    shared_ptr<Infer> create_infer(const string& engine_file, Type type, int gpuid, float confidence_threshold=0.25f, float nms_threshold=0.5f);
    const char* type_name(Type type);

}; // namespace Yolo

#endif // YOLO_HPP