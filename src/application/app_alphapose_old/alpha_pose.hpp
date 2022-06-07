#ifndef ALPHA_POSE_HPP
#define ALPHA_POSE_HPP

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>

// based on https://github.com/MVIG-SJTU/AlphaPose  v0.3.0 version
namespace AlphaPoseOld{

    using namespace std;
    using namespace cv;

    typedef tuple<Mat, Rect> Input;

    class Infer{
    public:
        virtual shared_future<vector<Point3f>> commit(const Input& input) = 0;
        virtual vector<shared_future<vector<Point3f>>> commits(const vector<Input>& inputs) = 0;
    };

    shared_ptr<Infer> create_infer(const string& engine_file, int gpuid);

}; // namespace AlphaPose

#endif // ALPHA_POSE_HPP