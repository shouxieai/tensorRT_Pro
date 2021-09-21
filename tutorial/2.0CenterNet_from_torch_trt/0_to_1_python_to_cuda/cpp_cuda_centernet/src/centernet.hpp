#ifndef CENTERNET_HPP
#define CENTERNET_HPP

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>

/** 
 * @brief high performance
 * support res18_dcn
 */

namespace Centernet {
    using namespace std;

    enum class Type: int{ // task type: DET: obj det  POSE: pose estimation  DDD:3d bbox estimation
        DET = 0,
        POSE = 1,
        DDD = 2
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


    


    struct AffineMatrix{
        float i2d[6];
        float d2i[6];

        void compute(const cv::Size& from, const cv::Size& to){
            float scale_x = to.width / (float)from.width;
            float scale_y = to.height / (float)from.height;

            float scale = std::min(scale_x, scale_y);
            
            // the array of the transformation M
            i2d[0] = scale; i2d[1] = 0;     i2d[2] = -scale * from.width * 0.5 + to.width * 0.5;
            i2d[3] = 0;     i2d[4] = scale; i2d[5] = -scale * from.height* 0.5 + to.height * 0.5;

            cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
            cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);

            cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
        }

        cv::Mat i2d_mat(){
            return cv::Mat(2, 3, CV_32F, i2d);
        };

    };


    class Infer{
    public:
        virtual shared_future<ObjectBoxArray> commit(const cv::Mat& image) = 0;
        virtual vector<shared_future<ObjectBoxArray>> commits(const vector<cv::Mat>& images) = 0;
    };

    shared_ptr<Infer> create_infer(const string& engine_file, Type type, int gpuid, float confidence_threshold=0.25f, float nms_threshold=0.5f);
    const char* type_name(Type type);

}; // namespace Centernet

#endif // CENTERNET_HPP