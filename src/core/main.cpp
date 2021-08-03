
/**
 * @file _main.cpp
 * @author 手写AI (zifuture.com:8090)
 * @date 2021-07-26
 * 
 *   实现了基于TensorRT对yolov5-5.0的推理工作 
 *   1. 基于FP32的模型编译、和推理执行
 *   2. 基于INT8的模型编译、和推理执行
 *   3. 自定义插件的实现，从pytorch导出到推理编译，并支持FP16
 * 
 *   预处理、后处理采用CPU实现（若想GPU可以自行实现）
 *   一次推理5张图获取结果
 * 
 *   我们是一群热血的个人组织者，力图发布免费高质量内容
 *   我们的博客地址：http://zifuture.com:8090
 *   我们的B站地址：https://space.bilibili.com/1413433465
 *   
 *   如果想要深入学习关于tensorRT的技术栈，请通过博客中的二维码联系我们（免费崔更即可）
 *   请关注B站，我们根据情况发布相关教程视频（免费）
 */

// 模型编译时使用的头文件
#include <builder/trt_builder.hpp>

// 模型推理时使用的头文件
#include <infer/trt_infer.hpp>

// 日志打印使用的文件
#include <common/ilogger.hpp>

// 封装的方式测试
#include "yolov5.hpp"

using namespace std;


// 展示效果时使用的coco标签名称 
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

// 因为图像需要进行预处理，这里采用仿射变换warpAffine进行处理，因此在这里计算仿射变换的矩阵
struct AffineMatrix{
    float i2d[6];       // image to dst(network), 2x3 matrix
    float d2i[6];       // dst to image, 2x3 matrix

    void compute(const cv::Size& from, const cv::Size& to){
        float scale_x = to.width / (float)from.width;
        float scale_y = to.height / (float)from.height;

        // 这里取min的理由是
        // 1. M矩阵是 from * M = to的方式进行映射，因此scale的分母一定是from
        // 2. 取最小，即根据宽高比，算出最小的比例，如果取最大，则势必有一部分超出图像范围而被裁剪掉，这不是我们要的
        float scale = min(scale_x, scale_y);

        /**
        这里的仿射变换矩阵实质上是2x3的矩阵，具体实现是
        scale, 0, -scale * from.width * 0.5 + to.width * 0.5
        0, scale, -scale * from.height * 0.5 + to.height * 0.5
        
        这里可以想象成，是经历过缩放、平移、平移三次变换后的组合，M = TPS
        例如第一个S矩阵，定义为把输入的from图像，等比缩放scale倍，到to尺度下
        S = [
          scale,     0,      0
          0,     scale,      0
          0,         0,      1
        ]
        
        P矩阵定义为第一次平移变换矩阵，将图像的原点，从左上角，移动到缩放(scale)后图像的中心上
        P = [
           1,        0,      -scale * from.width * 0.5
           0,        1,      -scale * from.height * 0.5
           0,        0,                1
        ]

        T矩阵定义为第二次平移变换矩阵，将图像从原点移动到目标（to）图的中心上
        T = [
           1,        0,      to.width * 0.5,
           0,        1,      to.height * 0.5,
           0,        0,            1
        ]

        通过将3个矩阵顺序乘起来，即可得到下面的表达式：
        M = [
           scale,    0,     -scale * from.width * 0.5 + to.width * 0.5
           0,     scale,    -scale * from.height * 0.5 + to.height * 0.5
           0,        0,                     1
        ]
        去掉第三行就得到opencv需要的输入2x3矩阵
        **/

        i2d[0] = scale;  i2d[1] = 0;  i2d[2] = -scale * from.width  * 0.5  + to.width * 0.5;
        i2d[3] = 0;  i2d[4] = scale;  i2d[5] = -scale * from.height * 0.5 + to.height * 0.5;

        // 有了i2d矩阵，我们求其逆矩阵，即可得到d2i（用以解码时还原到原始图像分辨率上）
        cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
        cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
        cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
    }

    cv::Mat i2d_mat(){
        return cv::Mat(2, 3, CV_32F, i2d);
    }
};

struct ObjectBox{
    float left, top, right, bottom, confidence;
    int class_label;

    ObjectBox() = default;

    // 这个值构造函数，是为了给emplace_back函数使用的
    ObjectBox(float left, float top, float right, float bottom, float confidence, int class_label)
    :left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label){}
};

float iou(const ObjectBox& a, const ObjectBox& b){
    float xmax = std::max(a.left, b.left);
    float ymax = std::max(a.top, b.top);
    float xmin = std::min(a.right, b.right);
    float ymin = std::min(a.bottom, b.bottom);
    float uw = (xmin - xmax > 0) ? (xmin - xmax) : 0;
    float uh = (ymin - ymax > 0) ? (ymin - ymax) : 0;
    float iou = uw * uh;
    return iou / ((a.right - a.left) * (a.bottom - a.top) + 
                  (b.right - b.left) * (b.bottom - b.top) - iou);
}

// 非极大值抑制，这个函数实现了类间的nms，即不同类之间是互不干扰的
// 思路是沿用yolov5代码里面的小技巧，即把每个框的坐标增加class * MAX_IMAGE_SIZE，再进行nms
// 执行完后还原即可，这样不同类别之间的iou会很小甚至为0
void nms(vector<ObjectBox>& objs, float threshold=0.5){

    // 对框排序，基于置信度
    std::sort(objs.begin(), objs.end(), [](ObjectBox& a, ObjectBox& b){
        return a.confidence > b.confidence;
    });

    vector<bool> removed_flags(objs.size());
    for(int i = 0; i < objs.size(); ++i){

        if(removed_flags[i])
            continue;

        for(int j = i + 1; j < objs.size(); ++j){
            if(objs[i].class_label == objs[j].class_label){
                if(iou(objs[i], objs[j]) >= threshold)
                    removed_flags[j] = true;
            }
        }
    }

    // 移除被删掉的框
    for(int i = (int)objs.size() - 1; i >= 0; --i){
        if(removed_flags[i])
            objs.erase(objs.begin() + i);
    }
}

// 给定x、y坐标，经过仿射变换矩阵，进行映射
tuple<float, float> affine_project(float x, float y, float* pmatrix){

    float newx = x * pmatrix[0] + y * pmatrix[1] + pmatrix[2];
    float newy = x * pmatrix[3] + y * pmatrix[4] + pmatrix[5];
    return make_tuple(newx, newy);
}

void forward_engine(const string& engine_file){

    iLogger::set_log_level(ILOGGER_VERBOSE);

    // 加载tensorRT编译好的模型，并打印模型的情况
    INFO("Load engine %s", engine_file.c_str());
    auto engine = TRTInfer::load_engine(engine_file);
    int test_batch_size = engine->get_max_batch_size();
    engine->print();

    // 获取模型的输入和输出tensor，也可以通过input(索引)和output(索引)函数
    // 用索引获取，自己要十分清楚获取的tensor是谁
    auto input = engine->tensor("images");
    auto output = engine->tensor("output");

    // 预处理部分，将图像输入到tensor
    vector<AffineMatrix> affine_matrixs(test_batch_size);
    int input_width = input->size(3);
    int input_height = input->size(2);
    cv::Size input_size(input_width, input_height);

    // 对batch个图像进行预处理，并放到tensor中
    for(int ibatch = 0; ibatch < test_batch_size; ++ibatch){

        auto& affine_matrix = affine_matrixs[ibatch];
        auto image = cv::imread(iLogger::format("%d.jpg", ibatch+1));
        affine_matrix.compute(image.size(), input_size);

        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        cv::warpAffine(image, image, affine_matrix.i2d_mat(), input_size, 1, cv::BORDER_CONSTANT, cv::Scalar::all(114));
        image.convertTo(image, CV_32F, 1 / 255.0f);
        input->set_mat(ibatch, image);
    }

    // 进行推理，这里的false是指，执行异步操作，即立马返回
    engine->forward(false);

    // 后处理部分，解码，还原
    vector<vector<ObjectBox>> batch_object_boxes(test_batch_size);
    int num_classes = output->size(2) - 5;
    float threshold = 0.25;

    for(int ibatch = 0; ibatch < test_batch_size; ++ibatch){

        float* image_based_output = output->cpu<float>(ibatch);
        auto& image_based_boxes = batch_object_boxes[ibatch];
        auto& affine_matrix = affine_matrixs[ibatch];

        for(int i = 0; i < output->size(1); ++i){
            int position = output->size(2) * i;
            float* pos_ptr = image_based_output + position;
            float objectness = pos_ptr[4];

            // yolov5的85个值排列如下：
            // cx, cy, width, height, objectness, class1, class2, ... class80
            if(objectness >= threshold){

                float* pbegin = pos_ptr + 5;
                float* pend = pbegin + num_classes;
                int class_label = std::max_element(pbegin, pend) - pbegin;

                // 根据yolov5的定义，置信度的定义是需要乘以类别的得分
                float confidence = objectness * pbegin[class_label];
                if(confidence >= threshold){
                    float left = pos_ptr[0] - pos_ptr[2] / 2;
                    float top = pos_ptr[1] - pos_ptr[3] / 2;
                    float right = pos_ptr[0] + pos_ptr[2] / 2;
                    float bottom = pos_ptr[1] + pos_ptr[3] / 2;

                    // 将获取到的框进行反映射回去
                    tie(left, top) = affine_project(left, top, affine_matrix.d2i);
                    tie(right, bottom) = affine_project(right, bottom, affine_matrix.d2i);
                    image_based_boxes.emplace_back(left, top, right, bottom, confidence, class_label);
                }
            }
        }

        // 对整个图做nms，不同类别之间互不干扰
        nms(image_based_boxes);
    }

    INFO("Forward done, draw and save result");
    for(int ibatch = 0; ibatch < test_batch_size; ++ibatch){

        auto image = cv::imread(iLogger::format("%d.jpg", ibatch + 1));
        auto& image_based_boxes = batch_object_boxes[ibatch];
        for(auto& obj : image_based_boxes){

            // 使用根据类别计算的随机颜色填充
            uint8_t b, g, r;
            tie(r, g, b) = iLogger::random_color(obj.class_label);
            cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

            // 绘制类别名字
            auto name = cocolabels[obj.class_label];
            int width = strlen(name) * 18 + 10;
            cv::rectangle(image, cv::Point(obj.left-3, obj.top-33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
            cv::putText(image, iLogger::format("%s", name), cv::Point(obj.left, obj.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
        }

        INFO("Save to %d.draw.jpg", ibatch + 1);
        cv::imwrite(iLogger::format("%d.draw.jpg", ibatch + 1), image);
    }
}

void test_plugin(){

    // 通过以下代码即可生成plugin.onnx
    // cd workspace
    // python test_plugin.py

    // plugin.onnx是通过test_plugin.py生成的
    TRTBuilder::compile(
        TRTBuilder::TRTMode_FP32, {}, 3, "plugin.onnx", "plugin.fp32.trtmodel", {}, false
    );
 
    auto engine = TRTInfer::load_engine("plugin.fp32.trtmodel");
    engine->print();

    auto input0 = engine->input(0);
    auto input1 = engine->input(1);
    auto output = engine->output(0);

    // 推理使用到了插件
    INFO("input0: %s", input0->shape_string());
    INFO("input1: %s", input1->shape_string());
    INFO("output: %s", output->shape_string());
    input0->set_to(0.80);
    engine->forward(true);
    INFO("output %f", output->at<float>(0, 0, 0, 0));
}

void test_int8(){

    INFO("===================== test int8 ==================================");
    auto int8process = [](int current, int count, vector<string>& images, shared_ptr<TRTInfer::Tensor>& tensor){

        INFO("Int8 %d / %d", current, count);

        // 按道理，这里的输入，应该按照推理的方式进行。这里我简单模拟了resize的方式输入
        for(int i = 0; i < images.size(); ++i){
            auto image = cv::imread(images[i]);
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
            cv::resize(image, image, cv::Size(tensor->size(3), tensor->size(2)));
            float mean[] = {0, 0, 0};
            float std[]  = {1, 1, 1};
            tensor->set_norm_mat(i, image, mean, std);
        }
    };

    iLogger::set_log_level(ILOGGER_VERBOSE);

    auto onnx_file = "yolov5s.onnx";
    auto model_file = "yolov5s.int8.trtmodel";
    int test_batch_size = 5;
    
    if(!iLogger::exists(model_file)){
        TRTBuilder::compile(
            TRTBuilder::TRTMode_INT8,   // 编译方式有，FP32、FP16、INT8
            {},                         // onnx时无效，caffe的输出节点标记
            test_batch_size,            // 指定编译的batch size
            onnx_file,                  // 需要编译的onnx文件
            model_file,                 // 储存的模型文件
            {},                         // 指定需要重定义的输入shape，这里可以对onnx的输入shape进行重定义
            false,                      // 是否采用动态batch维度，true采用，false不采用，使用静态固定的batch size
            int8process,                // int8标定时的数据输入处理函数
            "."                         // 图像数据的路径，在当前路径下找到用以标定的图像
        );
    }

    forward_engine(model_file);
}

void test_fp32(){

    iLogger::set_log_level(ILOGGER_VERBOSE);
    INFO("===================== test fp32 ==================================");

    auto onnx_file = "yolov5s.onnx";
    auto model_file = "yolov5s.fp32.trtmodel";
    int test_batch_size = 5;
    
    // 动态batch和静态batch，如果你想要弄清楚，请打开http://www.zifuture.com:8090/
    // 找到右边的二维码，扫码加好友后进群交流（免费哈，就是技术人员一起沟通）
    if(!iLogger::exists(model_file)){
        TRTBuilder::compile(
            TRTBuilder::TRTMode_FP32,   // 编译方式有，FP32、FP16、INT8
            {},                         // onnx时无效，caffe的输出节点标记
            test_batch_size,            // 指定编译的batch size
            onnx_file,                  // 需要编译的onnx文件
            model_file,                 // 储存的模型文件
            {},                         // 指定需要重定义的输入shape，这里可以对onnx的输入shape进行重定义
            false                       // 是否采用动态batch维度，true采用，false不采用，使用静态固定的batch size
        );
    }

    forward_engine(model_file);
}

void test_library_yolov5_hpp(){

    iLogger::set_log_level(ILOGGER_VERBOSE);
    INFO("===================== test library yolov5.hpp ==================================");

    auto onnx_file      = "yolov5s.onnx";
    auto model_file     = "yolov5s.fp32.trtmodel";
    int test_batch_size = 5;
    
    // 动态batch和静态batch，如果你想要弄清楚，请打开http://www.zifuture.com:8090/
    // 找到右边的二维码，扫码加好友后进群交流（免费哈，就是技术人员一起沟通）
    if(!iLogger::exists(model_file)){
        TRTBuilder::compile(
            TRTBuilder::TRTMode_FP32,   // 编译方式有，FP32、FP16、INT8
            {},                         // onnx时无效，caffe的输出节点标记
            test_batch_size,            // 指定编译的batch size
            onnx_file,                  // 需要编译的onnx文件
            model_file,                 // 储存的模型文件
            {},                         // 指定需要重定义的输入shape，这里可以对onnx的输入shape进行重定义
            false                       // 是否采用动态batch维度，true采用，false不采用，使用静态固定的batch size
        );
    }

    auto engine = YoloV5::create_infer(model_file, 0);
    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    vector<cv::Mat> images;
    for(int i = 0; i < 5; ++i){
        auto image = cv::imread(iLogger::format("%d.jpg", i + 1));

        // 使用eingine->commits一次推理一批，越多越好，性能最好
        auto box   = engine->commit(image).get();

        for(auto& obj : box){

            // 使用根据类别计算的随机颜色填充
            uint8_t b, g, r;
            tie(r, g, b) = iLogger::random_color(obj.class_label);
            cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

            // 绘制类别名字
            auto name = cocolabels[obj.class_label];
            int width = strlen(name) * 18 + 10;
            cv::rectangle(image, cv::Point(obj.left-3, obj.top-33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
            cv::putText(image, iLogger::format("%s", name), cv::Point(obj.left, obj.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
        }

        INFO("Save to %d.draw2.jpg", i + 1);
        cv::imwrite(iLogger::format("%d.draw2.jpg", i + 1), image);
    }
}

int main(){

    if(!iLogger::exists("yolov5s.onnx")){
        INFO("Auto download yolov5s.onnx");
        system("wget http://zifuture.com:1556/fs/25.shared/yolov5s.onnx");
    }

    test_library_yolov5_hpp();
    // test_fp32();
    // test_plugin();
    return 0;
}