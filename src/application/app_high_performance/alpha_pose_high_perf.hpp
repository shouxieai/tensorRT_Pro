#ifndef ALPHA_POSE_HIGH_PERF_HPP
#define ALPHA_POSE_HIGH_PERF_HPP

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>
#include "high_performance.hpp"

namespace AlphaPoseHighPerf{

    using namespace std;
    using namespace cv;
    using namespace HighPerformance;

    typedef tuple<Mat, Rect> Input;

    class PointArray : public Data, public vector<Point3f>{
    public:
        SetupData(PointArray);
    };

    class Infer{
    public:
        virtual shared_future<DataPtr> commit(const Input& input) = 0;
        virtual vector<shared_future<DataPtr>> commits(const vector<Input>& inputs) = 0;
    };

    shared_ptr<Infer> create_infer(const string& engine_file, int gpuid);

}; // namespace AlphaPose

#endif // ALPHA_POSE_HIGH_PERF_HPP