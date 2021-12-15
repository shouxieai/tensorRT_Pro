#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include "app_retinaface/retinaface.hpp"
#include "app_scrfd/scrfd.hpp"
#include "app_arcface/arcface.hpp"
#include "tools/deepsort.hpp"
#include "tools/zmq_remote_show.hpp"
#include <unordered_map>

using namespace std;
using namespace cv;

bool requires(const char* name);
bool compile_retinaface(int input_width, int input_height, string& out_model_file, TRT::Mode mode=TRT::Mode::FP32);
bool compile_scrfd(int input_width, int input_height, string& out_model_file, TRT::Mode mode = TRT::Mode::FP32);

static bool compile_models(){

    TRT::set_device(0);
    string model_file;

    if(!compile_retinaface(640, 480, model_file))
        return false;

    if(!compile_scrfd(640, 480, model_file))
        return false;

    const char* onnx_files[]{"arcface_iresnet50"};
    for(auto& name : onnx_files){
        if(not requires(name))
            return false;

        string onnx_file = iLogger::format("%s.onnx", name);
        string model_file = iLogger::format("%s.FP32.trtmodel", name);
        int test_batch_size = 1;
        
        if(not iLogger::exists(model_file)){
            bool ok = TRT::compile(
                TRT::Mode::FP32,            // FP32、FP16、INT8
                test_batch_size,            // max batch size
                onnx_file,                  // source
                model_file                  // saveto
            );

            if(!ok) return false;
        }
    }
    return true;
}

tuple<Mat, vector<string>> build_library(shared_ptr<Scrfd::Infer> detector, shared_ptr<Arcface::Infer> arcface){
    
    Mat_<float> features(0, 512);
    vector<string> names;
    auto libs = iLogger::find_files("face/library");
    INFO("Build library, %d images", libs.size());

    for(auto& file : libs){
        auto file_name = iLogger::file_name(file, false);
        Mat image      = imread(file);

        auto faces = detector->commit(image).get();
        if(faces.empty()){
            INFOW("%s no detect face.", file.c_str());
            continue;
        }

        RetinaFace::Box max_face = faces[0];
        if(faces.size() > 1){
            int max_face_index = std::max_element(faces.begin(), faces.end(), [](RetinaFace::Box& face1, RetinaFace::Box& face2){
                return face1.area() < face2.area();
            }) - faces.begin();
            max_face = faces[max_face_index];
        }

        auto face = max_face;
        if(face.width() < 80 or face.height() < 80)
            continue;

        Mat crop;
        tie(crop, face) = Scrfd::crop_face_and_landmark(image, face);
        Arcface::landmarks landmarks;
        memcpy(landmarks.points, face.landmark, sizeof(landmarks.points));

        auto feature     = arcface->commit(make_tuple(crop, landmarks)).get();
        string face_name = file_name;
        features.push_back(feature);
        names.push_back(face_name);

        INFO("New face [%s], %d feature, %.5f", face_name.c_str(), feature.cols, face.confidence);

        rectangle(image, cv::Point(max_face.left, max_face.top), cv::Point(max_face.right, max_face.bottom), Scalar(0, 255, 0), 2);
        for(int j = 0; j < 5; ++j)
            circle(image, Point(max_face.landmark[j*2+0], max_face.landmark[j*2+1]), 3, Scalar(0, 255, 0), -1, 16);
        putText(image, face_name, cv::Point(max_face.left, max_face.top), 0, 1, Scalar(0, 255, 0), 1, 16);

        string save_file = iLogger::format("face/library_draw/%s.jpg", file_name.c_str());
        imwrite(save_file, image);
    }
    return make_tuple(features, names);
}

int app_arcface(){

    TRT::set_device(0);
    INFO("===================== test arcface fp32 ==================================");

    if(!compile_models())
        return 0;

    iLogger::rmtree("face/library_draw");
    iLogger::rmtree("face/result");
    iLogger::mkdirs("face/library_draw");
    iLogger::mkdirs("face/result");

    auto detector = Scrfd::create_infer("scrfd_2.5g_bnkps.640x480.FP32.trtmodel", 0, 0.6f);
    //auto detector = RetinaFace::create_infer("mb_retinaface.640x480.FP32.trtmodel", 0, 0.5f);
    auto arcface  = Arcface::create_infer("arcface_iresnet50.FP32.trtmodel", 0);
    auto library  = build_library(detector, arcface);

    auto files    = iLogger::find_files("face/recognize");
    for(auto& file : files){

        auto image  = imread(file);
        auto faces  = detector->commit(image).get();
        vector<string> names(faces.size());
        for(int i = 0; i < faces.size(); ++i){
            Mat crop;
            auto face = faces[i];
            tie(crop, face) = Scrfd::crop_face_and_landmark(image, face);
            
            Arcface::landmarks landmarks;
            memcpy(landmarks.points, face.landmark, sizeof(landmarks.points));

            auto out          = arcface->commit(make_tuple(crop, landmarks)).get();
            auto scores       = Mat(get<0>(library) * out.t());
            float* pscore     = scores.ptr<float>(0);
            int label         = std::max_element(pscore, pscore + scores.rows) - pscore;
            float match_score = max(0.0f, pscore[label]);

            if(match_score > 0.3f){
                names[i] = iLogger::format("%s[%.3f]", get<1>(library)[label].c_str(), match_score);
            }
        }

        for(int i = 0; i < faces.size(); ++i){
            auto& face = faces[i];

            auto color = Scalar(0, 255, 0);
            if(names[i].empty()){
                color = Scalar(0, 0, 255);
                names[i] = "Unknow";
            }
            
            rectangle(image, cv::Point(face.left, face.top), cv::Point(face.right, face.bottom), color, 3);
            putText(image, names[i], cv::Point(face.left, face.top), 0, 1, color, 1, 16);
        }

        auto save_file = iLogger::format("face/result/%s.jpg", iLogger::file_name(file, false).c_str());
        INFO("Save to %s", save_file.c_str());
        imwrite(save_file, image);
    }
    INFO("Done");
    return 0;
}

int app_arcface_video(){

    TRT::set_device(0);
    INFO("===================== test arcface fp32 ==================================");

    if(!compile_models())
        return 0;

    iLogger::rmtree("face/library_draw");
    iLogger::rmtree("face/result");
    iLogger::mkdirs("face/library_draw");
    iLogger::mkdirs("face/result");

    auto detector = Scrfd::create_infer("scrfd_2.5g_bnkps.640x480.FP32.trtmodel", 0, 0.6f);
    //auto detector = RetinaFace::create_infer("mb_retinaface.640x480.FP32.trtmodel", 0, 0.5f);
    auto arcface  = Arcface::create_infer("arcface_iresnet50.FP32.trtmodel", 0);
    auto library  = build_library(detector, arcface);
    //auto remote_show = create_zmq_remote_show();
    INFO("Use tools/show.py to remote show");

    VideoCapture cap("exp/face_tracker.mp4");
    Mat image;
    while(cap.read(image)){
        auto faces  = detector->commit(image).get();
        vector<string> names(faces.size());
        for(int i = 0; i < faces.size(); ++i){
            Mat crop;
            auto face = faces[i];
            tie(crop, face) = Scrfd::crop_face_and_landmark(image, face);
            
            Arcface::landmarks landmarks;
            memcpy(landmarks.points, face.landmark, sizeof(landmarks.points));

            auto out          = arcface->commit(make_tuple(crop, landmarks)).get();
            auto scores       = Mat(get<0>(library) * out.t());
            float* pscore     = scores.ptr<float>(0);
            int label         = std::max_element(pscore, pscore + scores.rows) - pscore;
            float match_score = max(0.0f, pscore[label]);

            if(match_score > 0.3f){
                names[i] = iLogger::format("%s[%.3f]", get<1>(library)[label].c_str(), match_score);
            }
        }

        for(int i = 0; i < faces.size(); ++i){
            auto& face = faces[i];

            auto color = Scalar(0, 255, 0);
            if(names[i].empty()){
                color = Scalar(0, 0, 255);
                names[i] = "Unknow";
            }
            
            rectangle(image, cv::Point(face.left, face.top), cv::Point(face.right, face.bottom), color, 3);
            putText(image, names[i], cv::Point(face.left, face.top - 5), 0, 1, color, 1, 16);
        }
        //remote_show->post(image);
    }
    INFO("Done");
    return 0;
}

class MotionFilter{
public:
    MotionFilter(){
        location_.left = location_.top = location_.right = location_.bottom = 0;
    }

    void missed(){
        init_ = false;
    }

    void update(const DeepSORT::Box& box){

        const float a[] = {box.left, box.top, box.right, box.bottom};
        const float b[] = {location_.left, location_.top, location_.right, location_.bottom};

        if(!init_){
            init_ = true;
            location_ = box;
            return;
        }

        float v[4];
        for(int i = 0; i < 4; ++i)
            v[i] = a[i] * 0.6 + b[i] * 0.4;

        location_.left = v[0];
        location_.top = v[1];
        location_.right = v[2];
        location_.bottom = v[3];
    }

    DeepSORT::Box predict(){
        return location_;
    }

private:
    DeepSORT::Box location_;
    bool init_ = false;
};

int app_arcface_tracker(){

    TRT::set_device(0);
    INFO("===================== test arcface fp32 ==================================");

    if(!compile_models())
        return 0;

    auto detector = Scrfd::create_infer("scrfd_2.5g_bnkps.640x480.FP32.trtmodel", 0, 0.6f);
    //auto detector = RetinaFace::create_infer("mb_retinaface.640x480.FP32.trtmodel", 0, 0.6f);
    auto arcface  = Arcface::create_infer("arcface_iresnet50.FP32.trtmodel", 0);
    //auto library  = build_library(detector, arcface);

    //tools/show.py connect to remote show
    //auto remote_show = create_zmq_remote_show();
    INFO("Use tools/show.py to remote show");

    auto config = DeepSORT::TrackerConfig();
    config.has_feature = true;
    config.max_age     = 150;
    config.nbuckets    = 150;
    config.distance_threshold = 0.9f;

    config.set_per_frame_motion({
        0.05, 0.02, 0.1, 0.02,
        0.08, 0.02, 0.1, 0.02
    });
    
    auto tracker     = DeepSORT::create_tracker(config);
    VideoCapture cap("exp/face_tracker.mp4");
    Mat image;

    VideoWriter writer("tracker.result.mp4", cv::VideoWriter::fourcc('a', 'v', 'c', '1'), 
        cap.get(cv::CAP_PROP_FPS),
        Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT))
    );
    if(!writer.isOpened()){
        INFOE("Writer failed.");
        return 0;
    }

    unordered_map<int, MotionFilter> MotionFilter;
    while(cap.read(image)){
        auto faces  = detector->commit(image).get();
        vector<string> names(faces.size());
        vector<DeepSORT::Box> boxes;
        for(int i = 0; i < faces.size(); ++i){
            auto& face = faces[i];
            if(max(face.width(), face.height()) < 30) continue;

            auto crop  = Scrfd::crop_face_and_landmark(image, face);
            auto track_box = DeepSORT::convert_to_box(face);
            
            Arcface::landmarks landmarks;
            memcpy(landmarks.points, get<1>(crop).landmark, sizeof(landmarks.points));

            track_box.feature = arcface->commit(make_tuple(get<0>(crop), landmarks)).get();
            // Mat scores        = get<0>(library) * track_box.feature.t();
            // float* pscore     = scores.ptr<float>(0);
            // int label         = std::max_element(pscore, pscore + scores.rows) - pscore;
            // float match_score = max(0.0f, pscore[label]);
            boxes.emplace_back(std::move(track_box));

            // if(match_score > 0.3f){
            //     names[i] = iLogger::format("%s[%.3f]", get<1>(library)[label].c_str(), match_score);
            // }
        }
        tracker->update(boxes);

        auto final_objects = tracker->get_objects();
        for(int i = 0; i < final_objects.size(); ++i){
            auto& person = final_objects[i];
            auto& filter = MotionFilter[person->id()];

            if(person->time_since_update() == 0 && person->state() == DeepSORT::State::Confirmed){
                uint8_t r, g, b;
                std::tie(b, g, r) = iLogger::random_color(person->id());
                
                auto loaction = person->last_position();
                filter.update(loaction);
                loaction = filter.predict();

                const int shift = 4, shv = 1 << shift;
                rectangle(image, 
                    Point(loaction.left * shv, loaction.top * shv), 
                    Point(loaction.right * shv, loaction.bottom * shv), 
                    Scalar(b, g, r), 3, 16, shift
                );

                putText(image, iLogger::format("%d", person->id()), 
                    Point(loaction.left, loaction.top - 10), 
                    0, 2, Scalar(b, g, r), 3, 16
                );
           }else{
               filter.missed();
           }
        }

        // for(int i = 0; i < faces.size(); ++i){
        //     auto& face = faces[i];
        //     auto color = Scalar(0, 255, 0);
        //     if(names[i].empty()){
        //         color = Scalar(0, 0, 255);
        //         names[i] = "Unknow";
        //     }
        //     rectangle(image, cv::Point(face.left, face.top), cv::Point(face.right, face.bottom), color, 3);
        //     putText(image, names[i], cv::Point(face.left + 30, face.top - 10), 0, 1, color, 2, 16);
        // }
        //remote_show->post(image);
        writer.write(image);
    }
    INFO("Done");
    return 0;
}