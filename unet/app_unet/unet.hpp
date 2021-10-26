#ifndef UNET_HPP
#define UNET_HPP

#include <memory>
#include <vector>
#include <future>
#include <string>
#include <opencv2/opencv.hpp>

namespace UNet{

    using namespace std;

    class Infer{
    public:
        virtual shared_future<cv::Mat> commit(const cv::Mat& image) = 0;
        virtual vector<shared_future<cv::Mat>> commits(const vector<cv::Mat>& images) = 0;
    };

    // RAII，如果创建失败，返回空指针 *//
    shared_ptr<Infer> create_infer(const string& engine_file, int gpuid);

}; // namespace UNet


#endif // UNET_HPP