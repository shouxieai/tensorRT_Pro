#ifndef U_net_HPP
#define U_net_HPP

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>


/**
 * @brief 发挥极致的性能体验
 * 支持YoloX和YoloV5
 */
namespace U_net{

    using namespace std;

    struct SegmentResult{
        cv::Mat out_img;
        cv::Mat invertMatrix;
        SegmentResult(cv::Mat out_img, cv::Mat invertMatrix)
        :out_img(out_img), invertMatrix(invertMatrix){}
        SegmentResult();
        
    };

    class Infer{
    public:
        virtual shared_future<SegmentResult> commit(const cv::Mat& image) = 0;
        virtual vector<shared_future<SegmentResult>> commits(const vector<cv::Mat>& images) = 0;
    };

    // RAII，如果创建失败，返回空指针
    shared_ptr<Infer> create_infer(const string& engine_file, int gpuid, float i);

    
}; // namespace Yolo

#endif // YOLO_HPP