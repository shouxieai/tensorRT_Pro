#ifndef YOLO_FAST_HPP
#define YOLO_FAST_HPP

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
namespace YoloFast{

    using namespace std;
    using namespace ObjectDetector;

    enum class Type : int{
        V5_P5 = 0,
        V5_P6 = 1,
        X  = 2
    };

    struct DecodeMeta{
        int num_anchor;
        int num_level;
        float w[16], h[16];
        int strides[16];

        static DecodeMeta v5_p5_default_meta();
        static DecodeMeta v5_p6_default_meta();
        static DecodeMeta x_default_meta();
    };

    class Infer{
    public:
        virtual shared_future<BoxArray> commit(const cv::Mat& image) = 0;
        virtual vector<shared_future<BoxArray>> commits(const vector<cv::Mat>& images) = 0;
    };

    void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, Type type, int ibatch);

    shared_ptr<Infer> create_infer(
        const string& engine_file, 
        Type type, 
        int gpuid, 
        float confidence_threshold=0.25f, 
        float nms_threshold=0.5f, 
        const DecodeMeta& meta = DecodeMeta::v5_p5_default_meta()
    );
    const char* type_name(Type type);

}; // namespace YoloFast

#endif // YOLO_FAST_HPP