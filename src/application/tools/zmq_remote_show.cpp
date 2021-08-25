
#include "zmq_remote_show.hpp"
#include "zmq_u.hpp"
#include <common/ilogger.hpp>

using namespace std;

class ZMQRemoteShowImpl : public ZMQRemoteShow{
public:
    bool listen(const char* url){
        try{
            context_.reset(new zmq::context_t());
            socket_.reset(new zmq::socket_t(*context_.get(), zmq::socket_type::rep));
            socket_->bind(url);
            return true;
        }catch(zmq::error_t err){
            INFOE("ZMQ exception: %s", err.what());
            socket_.reset();
            context_.reset();
        }
        return false;
    }

    virtual void post(const void* data, int size) override{

        if(size < 1 || data == nullptr){
            INFOE("Null data to post");
            return;
        }

        zmq::message_t msg;
        socket_->recv(msg);
        socket_->send(zmq::message_t(data, size));
    }

    virtual void post(const cv::Mat& image) override{

        vector<unsigned char> data;
        cv::imencode(".jpg", image, data);
        post(data.data(), data.size());
    }

private:
    shared_ptr<zmq::context_t> context_;
    shared_ptr<zmq::socket_t>  socket_;
};

std::shared_ptr<ZMQRemoteShow> create_zmq_remote_show(const char* listen){
    
    shared_ptr<ZMQRemoteShowImpl> instance(new ZMQRemoteShowImpl());
    if(!instance->listen(listen)){
        instance.reset();
    }
    return instance;
}
