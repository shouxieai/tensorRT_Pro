

#ifndef ZMQ_REMOTE_SHOW_HPP
#define ZMQ_REMOTE_SHOW_HPP

#include <memory>
#include <opencv2/opencv.hpp>

class ZMQRemoteShow{
public:
    virtual void post(const void* data, int size) = 0;
    virtual void post(const cv::Mat& image) = 0;
};

std::shared_ptr<ZMQRemoteShow> create_zmq_remote_show(const char* listen="tcp://0.0.0.0:15556");

#endif // ZMQ_REMOTE_SHOW_HPP