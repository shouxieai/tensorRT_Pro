
#ifndef DEEPSORT_HPP
#define DEEPSORT_HPP

#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>

namespace DeepSORT {

struct Box{
    float left, top, right, bottom;
    cv::Mat feature;

    Box() = default;
    Box(float left, float top, float right, float bottom):left(left), top(top), right(right), bottom(bottom){}
    const float width() const{return right - left;}
    const float height() const{return bottom - top;}
    const cv::Point2f center() const{return cv::Point2f((left+right)/2, (top+bottom)/2);}
};

template<typename _T>
inline Box convert_to_box(const _T& b){
    return Box(b.left, b.top, b.right, b.bottom);
}

template<typename _T>
inline cv::Rect convert_box_to_rect(const _T& b){
    return cv::Rect(b.left, b.top, b.right-b.left, b.bottom-b.top);
}

enum class State : int{
    Tentative = 1,
    Confirmed = 2,
    Deleted   = 3
};

typedef std::vector<Box> BBoxes;

class TrackObject{
public:
	virtual int id() const = 0;
    virtual State state() const = 0;
	virtual Box predict_box() const = 0;
    virtual Box last_position() const = 0;
	virtual bool is_confirmed() const = 0;
	virtual int time_since_update() const = 0;
    virtual std::vector<cv::Point> trace_line() const = 0;
    virtual int trace_size() const = 0;
    virtual Box& location(int time_since_update=0) = 0;
    virtual const cv::Mat& feature_bucket() const = 0;
};

class Tracker{
public:
    virtual std::vector<TrackObject *> get_objects() = 0;
    virtual void update(const BBoxes& boxes) = 0;
};

std::shared_ptr<Tracker> create_tracker(
    float feature_score_threshold = 0.1f,
    int nbuckets = 150,
    int max_age  = 150,
    int nhit     = 3
);

}

#endif // DEEPSORT_HPP