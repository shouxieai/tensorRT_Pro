#ifndef YOLO_HPP
#define YOLO_HPP

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>

/**
 * @brief 发挥极致的性能体验
 * 支持YoloX和YoloV5
 */
namespace Yolo{

    using namespace std;

    enum class Type : int{
        V5 = 0,
        X  = 1
    };    

    struct ObjectBox{
        float left, top, right, bottom, confidence;
        int class_label;

        ObjectBox() = default;

        // 这个值构造函数，是为了给emplace_back函数使用的
        ObjectBox(float left, float top, float right, float bottom, float confidence, int class_label)
        :left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label){}
    };

    typedef vector<ObjectBox> box_array;

    class Infer{
    public:
        virtual shared_future<box_array> commit(const cv::Mat& image) = 0;
        virtual vector<shared_future<box_array>> commits(const vector<cv::Mat>& images) = 0;
    };

    // RAII，如果创建失败，返回空指针
    shared_ptr<Infer> create_infer(const string& engine_file, Type type, int gpuid, float confidence_threshold=0.25f);
    const char* type_name(Type type);

}; // namespace Yolo

#endif // YOLO_HPP