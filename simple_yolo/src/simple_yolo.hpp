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
		FP16
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

    const char* type_name(Type type);
    const char* mode_string(Mode type);
    void set_device(int device_id);
    bool compile(Mode mode, unsigned int maxBatchSize, const string& source_onnx, const string& saveto);
    shared_ptr<Infer> create_infer(const string& engine_file, Type type, int gpuid, float confidence_threshold=0.25f, float nms_threshold=0.5f);

}; // namespace SimpleYolo

#endif // SIMPLE_YOLO_HPP