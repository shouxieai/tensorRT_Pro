#ifndef YOLOX_HPP
#define YOLOX_HPP

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>

namespace YoloX{

    using namespace std;
    using namespace cv;

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
        virtual shared_future<box_array> commit(const Mat& image) = 0;
        virtual vector<shared_future<box_array>> commits(const vector<Mat>& images) = 0;
    };

    // RAII，如果创建失败，返回空指针
    shared_ptr<Infer> create_infer(const string& engine_file, int gpuid);

}; // namespace YoloV5

#endif // YOLOX_HPP