
#include "app_high_performance/high_performance.hpp"
#include "app_high_performance/yolo_high_perf.hpp"
#include "app_high_performance/alpha_pose_high_perf.hpp"

using namespace std;
using namespace HighPerformance;

class ImageData : public Data, public cv::Mat{
public:
    SetupData(ImageData);

    ImageData() = default;
    ImageData(const cv::Mat& image):cv::Mat(image){}
};

class YoloNode : public Node{
public:
    virtual bool startup(const string& file, YoloHighPerf::Type type, int gpuid = 0){
        infer_ = YoloHighPerf::create_infer(file, type, gpuid);
        if(infer_ == nullptr)
            return false;

        Node::startup();
        return true;
    }

protected:
    virtual void forward(vector<shared_ptr<Data>>& inputs_data) override{

        auto image = dynamic_obj_cast(inputs_data[0], ImageData);
        auto output_future = infer_->commit(*image);
        outputs_[0]->commit(output_future);
    }
    
private:
    shared_ptr<YoloHighPerf::Infer> infer_;
};

class PoseNode : public Node{
public:
    virtual bool startup(const string& file, int gpuid = 0){
        infer_ = AlphaPoseHighPerf::create_infer(file, gpuid);
        if(infer_ == nullptr)
            return false;

        Node::startup();
        return true;
    }

protected:
    virtual void forward(vector<shared_ptr<Data>>& inputs_data) override{

        auto image    = dynamic_obj_cast(inputs_data[0], cv::Mat);
        auto boxarray = dynamic_obj_cast(inputs_data[1], YoloHighPerf::BoxArray);
        vector<shared_future<DataPtr>> keypoints(boxarray->size());
        for(int i = 0; i < boxarray->size(); ++i){
            auto& box = boxarray->at(i);
            keypoints[i] = infer_->commit(make_tuple(*image, cv::Rect(box.left, box.top, box.right-box.left, box.bottom-box.top)));
        }

        outputs_[0]->commit(
            make_data_future(
                make_shared<Container<tuple<shared_ptr<cv::Mat>, vector<shared_future<DataPtr>>>>>(image, keypoints)
            )
        );
    }
    
private:
    shared_ptr<AlphaPoseHighPerf::Infer> infer_;
};

int app_high_performance(){

    InputNode camera;
    OutputNode output;
    YoloNode yolo;
    PoseNode pose;

    connect(camera, yolo);
    connect(camera, pose);
    connect(yolo, pose);
    connect(pose, output);

    camera.startup([](vector<shared_ptr<Pipeline>>& output_pipe){

        cv::VideoCapture cap("exp/fall_video.mp4");
        cv::Mat image;
        int num_frame = 0;
        while(cap.read(image)){
            num_frame++;
            for(int i = 0; i < output_pipe.size(); ++i)
                output_pipe[i]->commit(make_data_future(make_shared<ImageData>(image.clone())));
        }
        INFO("num frame %d", num_frame);
    });

    output.startup([](vector<shared_ptr<Data>>& datas){

        using ThisData = Container<tuple<shared_ptr<cv::Mat>, vector<shared_future<DataPtr>>>>;
        auto data = dynamic_obj_cast(datas[0], ThisData);

        auto& image = *get<0>(data->value);
        auto& points = get<1>(data->value);
        for(auto& keys_fut : points){

            auto keys = keys_fut.get();
            auto keys_kpt = dynamic_obj_cast(keys, AlphaPoseHighPerf::PointArray);
        }
    });

    yolo.startup("yolox_s.FP32.trtmodel", YoloHighPerf::Type::X);
    pose.startup("sppe.fp32.trtmodel");
    getchar();
    camera.stop();
    return 0;
}