#ifndef CENTERNET_HPP
#define CENTERNET_HPP

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>
#include <common/trt_tensor.hpp>
#include <common/object_detector.hpp>

namespace CenterNet{

    using namespace std;
    using namespace ObjectDetector;

    void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, int ibatch);

    class Infer{
    public:
        virtual shared_future<BoxArray> commit(const cv::Mat& image) = 0;
        virtual vector<shared_future<BoxArray>> commits(const vector<cv::Mat>& images) = 0;
    };

    shared_ptr<Infer> create_infer(const string& engine_file, int gpuid, float confidence_threshold=0.25f, float nms_threshold=0.5f);

}; // namespace CenterNet


#endif // CENTERNET_HPP