
#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include <common/json.hpp>
#include "app_yolo/yolo.hpp"
#include <vector>
#include <string>

using namespace std;

static const char* cocolabels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

bool requires(const char* name);

struct BoxLabel{
    int label;
    float cx, cy, width, height;
    float confidence;
};

struct ImageItem{
    string image_file;
    Yolo::BoxArray detections;
};

vector<ImageItem> scan_dataset(const string& images_root){

    vector<ImageItem> output;
    auto image_files = iLogger::find_files(images_root, "*.jpg");

    for(int i = 0; i < image_files.size(); ++i){
        auto& image_file = image_files[i];

        if(!iLogger::exists(image_file)){
            INFOW("Not found: %s", image_file.c_str());
            continue;
        }

        ImageItem item;
        item.image_file = image_file;
        output.emplace_back(item);
    }
    return output;
}

static void inference(vector<ImageItem>& images, int deviceid, const string& engine_file, TRT::Mode mode, Yolo::Type type, const string& model_name){

    auto engine = Yolo::create_infer(
        engine_file, type, deviceid, 0.001f, 0.65f,
        Yolo::NMSMethod::CPU, 10000
    );
    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    int nimages = images.size();
    vector<shared_future<Yolo::BoxArray>> image_results(nimages);
    for(int i = 0; i < nimages; ++i){
        if(i % 100 == 0){
            INFO("Commit %d / %d", i+1, nimages);
        }
        image_results[i] = engine->commit(cv::imread(images[i].image_file));
    }
    
    for(int i = 0; i < nimages; ++i)
        images[i].detections = image_results[i].get();
}

void detect_images(vector<ImageItem>& images, Yolo::Type type, TRT::Mode mode, const string& model){

    int deviceid = 0;
    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(deviceid);

    auto int8process = [=](int current, int count, const vector<string>& files, shared_ptr<TRT::Tensor>& tensor){

        INFO("Int8 %d / %d", current, count);

        for(int i = 0; i < files.size(); ++i){
            auto image = cv::imread(files[i]);
            Yolo::image_to_tensor(image, tensor, type, i);
        }
    };

    const char* name = model.c_str();
    INFO("===================== test %s %s %s ==================================", Yolo::type_name(type), mode_name, name);

    if(not requires(name))
        return;

    string onnx_file = iLogger::format("%s.onnx", name);
    string model_file = iLogger::format("%s.%s.trtmodel", name, mode_name);
    int test_batch_size = 16;
    
    if(not iLogger::exists(model_file)){
        TRT::compile(
            mode,                       // FP32、FP16、INT8
            test_batch_size,            // max batch size
            onnx_file,                  // source 
            model_file,                 // save to
            {},
            int8process,
            "inference"
        );
    }
    inference(images, deviceid, model_file, mode, type, name);
}

bool save_to_json(const vector<ImageItem>& images, const string& file){

    int to_coco90_class_map[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
    };
    Json::Value predictions(Json::arrayValue);
    for(int i = 0; i < images.size(); ++i){
        auto& image = images[i];
        auto file_name = iLogger::file_name(image.image_file, false);
        int image_id = atoi(file_name.c_str());

        auto& boxes = image.detections;
        for(auto& box : boxes){
            Json::Value jitem;
            jitem["image_id"] = image_id;
            jitem["category_id"] = to_coco90_class_map[box.class_label];
            jitem["score"] = box.confidence;

            auto& bbox = jitem["bbox"];
            bbox.append(box.left);
            bbox.append(box.top);
            bbox.append(box.right - box.left);
            bbox.append(box.bottom - box.top);
            predictions.append(jitem);
        }
    }
    return iLogger::save_file(file, predictions.toStyledString());
}

int test_yolo_map(){
    
    /*
    结论：
    1. YoloV5在tensorRT下和pytorch下，只要输入一样，输出的差距最大值是1e-3
    2. YoloV5-6.0的mAP，官方代码跑下来是mAP@.5:.95 = 0.367, mAP@.5 = 0.554，与官方声称的有差距
    3. 这里的tensorRT版本测试的精度为：mAP@.5:.95 = 0.357, mAP@.5 = 0.539，与pytorch结果有差距
    4. cv2.imread与cv::imread，在操作jpeg图像时，在我这里测试读出的图像值不同，最大差距有19。而png图像不会有这个问题
        若想完全一致，请用png图像
    5. 预处理部分，若采用letterbox的方式做预处理，由于tensorRT这里是固定640x640大小，测试采用letterbox并把多余部分
        设置为0. 其推理结果与pytorch相近，但是依旧有差别
    6. 采用warpAffine和letterbox两种方式的预处理结果，在mAP上没有太大变化（小数点后三位差）
    7. mAP差一个点的原因可能在固定分辨率这件事上，还有是pytorch实现的所有细节并非完全加入进来。这些细节可能有没有
        找到的部分
    */

    auto images = scan_dataset("/data/sxai/dataset/coco/images/val2017");
    INFO("images.size = %d", images.size());

    string model = "yolov5s";
    detect_images(images, Yolo::Type::V5, TRT::Mode::FP32, model);
    save_to_json(images, model + ".prediction.json");
    return 0;
}