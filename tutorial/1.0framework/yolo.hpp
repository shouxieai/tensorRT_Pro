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
        X  = 1,
        V3 = 2,
        V7 = 3
    };    

    struct ObjectBox{
        float left, top, right, bottom, confidence;
        int class_label;

        ObjectBox() = default;

        ObjectBox(float left, float top, float right, float bottom, float confidence, int class_label)
        :left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label){}

        float get_left()                {return left;}
        void set_left(float value)      {left = value;}
        float get_top()                 {return top;}
        void set_top(float value)       {top = value;}
        float get_right()               {return right;}
        void set_right(float value)     {right = value;}
        float get_bottom()              {return bottom;}
        void set_bottom(float value)    {bottom = value;}
        float get_confidence()          {return confidence;}
        void set_confidence(float value){confidence = value;}
        int get_class_label()           {return class_label;}
        void set_class_label(int value) {class_label = value;}
    };

    typedef vector<ObjectBox> ObjectBoxArray;

    class Infer{
    public:
        virtual shared_future<ObjectBoxArray> commit(const cv::Mat& image) = 0;
        virtual vector<shared_future<ObjectBoxArray>> commits(const vector<cv::Mat>& images) = 0;
    };

    shared_ptr<Infer> create_infer(const string& engine_file, Type type, int gpuid, float confidence_threshold=0.25f, float nms_threshold=0.5f);
    const char* type_name(Type type);

}; // namespace Yolo

#endif // YOLO_HPP