#ifndef YOLO_HIGHPERF_HPP
#define YOLO_HIGHPERF_HPP

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>
#include "high_performance.hpp"

/**
 * @brief 发挥极致的性能体验
 * 支持YoloX和YoloV5
 */
namespace YoloHighPerf{

    using namespace std;
    using namespace HighPerformance;

    enum class Type : int{
        V5 = 0,
        X  = 1,
        V3 = 2,
        V7 = 3
    };    

    struct Box{
        float left, top, right, bottom, confidence;
        int class_label;

        Box() = default;

        Box(float left, float top, float right, float bottom, float confidence, int class_label)
        :left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label){}
    };

    class BoxArray : public Data, public vector<Box>{
    public:
        SetupData(BoxArray);
    };

    class Infer{
    public:
        virtual shared_future<DataPtr> commit(const cv::Mat& image) = 0;
        virtual vector<shared_future<DataPtr>> commits(const vector<cv::Mat>& images) = 0;
    };

    shared_ptr<Infer> create_infer(const string& engine_file, Type type, int gpuid, float confidence_threshold=0.25f, float nms_threshold=0.5f);
    const char* type_name(Type type);

}; // namespace Yolo

#endif // YOLO_HIGHPERF_HPP