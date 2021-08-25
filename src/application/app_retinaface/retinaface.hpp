#ifndef RETINAFACE_HPP
#define RETINAFACE_HPP

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>

namespace RetinaFace{

    using namespace std;

    struct FaceBox{
        float left, top, right, bottom, confidence;
        float landmark[10];

        cv::Rect cvbox() const{return cv::Rect(left, top, right-left, bottom-top);}
        float width()    const{return max(0.0f, right-left);}
        float height()   const{return max(0.0f, bottom-top);}
        float area()     const{return width() * height();}
    };

    typedef vector<FaceBox> box_array;

    class Infer{
    public:
        virtual shared_future<box_array> commit(const cv::Mat& image) = 0;
        virtual vector<shared_future<box_array>> commits(const vector<cv::Mat>& images) = 0;
    };

    // RAII，如果创建失败，返回空指针
    shared_ptr<Infer> create_infer(const string& engine_file, int gpuid, float confidence_threshold=0.5f);

}; // namespace RetinaFace

#endif // RETINAFACE_HPP