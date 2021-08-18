#include "yolox.hpp"
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>

namespace YoloX{

    // 因为图像需要进行预处理，这里采用仿射变换warpAffine进行处理，因此在这里计算仿射变换的矩阵
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

    struct Job{
        Mat image;
        AffineMatrix affine_matrix;
        box_array boxes;
        shared_ptr<promise<box_array>> pro;
    };

    // 给定x、y坐标，经过仿射变换矩阵，进行映射
    static tuple<float, float> affine_project(float x, float y, float* pmatrix){

        float newx = x * pmatrix[0] + y * pmatrix[1] + pmatrix[2];
        float newy = x * pmatrix[3] + y * pmatrix[4] + pmatrix[5];
        return make_tuple(newx, newy);
    }

    static float iou(const ObjectBox& a, const ObjectBox& b){
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
    static void nms(vector<ObjectBox>& objs, float threshold=0.5){

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


    class InferImpl : public Infer{
    public:
        virtual ~InferImpl(){
            stop();
        }

        void stop(){
            run_ = false;
            cond_.notify_all();

            if(worker_){
                worker_->join();
                worker_.reset();
            }
        }

        bool startup(const string& file, int gpuid){
            run_ = true;

            promise<bool> pro;
            worker_ = make_shared<thread>(&InferImpl::worker, this, file, gpuid, std::ref(pro));
            return pro.get_future().get();
        }

        void worker(string file, int gpuid, promise<bool>& result){

            TRT::set_device(gpuid);
            auto engine = TRT::load_infer(file);
            if(engine == nullptr){
                INFOE("Engine %s load failed", file.c_str());
                result.set_value(false);
                return;
            }

            engine->print();

            int max_batch_size = engine->get_max_batch_size();
            auto input         = engine->tensor("images");
            auto output        = engine->tensor("output");
            int num_classes    = output->size(2) - 5;
            float threshold    = 0.25;
            float mean[]       = {0.485, 0.456, 0.406};
            float std[]        = {0.229, 0.224, 0.225};

            const int num_stride = 3;
            int fm_sizes[num_stride];
            int strides[num_stride] = {8, 16, 32};
            int stride_limit[num_stride];
            int stride_begin[num_stride] = {0};

            for(int i = 0; i < num_stride; ++i){
                int size = input->size(2) / strides[i];
                fm_sizes[i] = size;
                stride_limit[i] = size * size;

                if(i > 0)
                    stride_begin[i] = stride_limit[i - 1];

                if(i > 0)
                    stride_limit[i] += stride_limit[i - 1];
            }

            input_width_  = input->size(3);
            input_height_ = input->size(2);
            result.set_value(true);
            input->resize_single_dim(0, max_batch_size);

            vector<Job> fetch_jobs;
            while(run_){
                {
                    unique_lock<mutex> l(jobs_lock_);
                    cond_.wait(l, [&](){
                        return !run_ || !jobs_.empty();
                    });

                    if(!run_) continue;
                    
                    for(int i = 0; i < max_batch_size && !jobs_.empty(); ++i){
                        fetch_jobs.emplace_back(std::move(jobs_.front()));
                        jobs_.pop();
                    }
                };

                // 一次推理越多越好
                // 把图像批次丢引擎里边去
                int infer_batch_size = fetch_jobs.size();
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    input->set_norm_mat(ibatch, fetch_jobs[ibatch].image, mean, std);
                }
                // 模型推理
                engine->forward(false);

                // 收取结果，output->cpu里面存在一个同步操作
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    
                    auto& job                 = fetch_jobs[ibatch];
                    float* image_based_output = output->cpu<float>(ibatch);
                    auto& image_based_boxes   = job.boxes;
                    auto& affine_matrix       = job.affine_matrix;

                    for(int i = 0; i < output->size(1); ++i){
                        int position     = output->size(2) * i;
                        float* pos_ptr   = image_based_output + position;
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

                                int s = 0;
                                for(; s < num_stride; ++s){
                                    if(i < stride_limit[s])
                                        break;
                                }

                                if(s == num_stride){
                                    // 如果提示这个错误，说明没有修改head.py
                                    // 1. 打开：/data/sxai/YOLOX/yolox/models/yolo_head.py第208行
                                    // 2. 找到：outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
                                    // 3. 修改为如下：
                                    /*
                                        # outputs = torch.cat(
                                        #     [x.flatten(start_dim=2) for x in outputs], dim=2
                                        # ).permute(0, 2, 1)

                                        proc_view = lambda x: x.view(-1, int(x.size(1)), int(x.size(2) * x.size(3)))
                                        outputs = torch.cat(
                                            [proc_view(x) for x in outputs], dim=2
                                        ).permute(0, 2, 1)
                                    */
                                    // 具体原因可以群里问我。是因为你使用到了多batch，但是导出时的x.flatten确把batch维度给弄到size(1)上去了，这与预期不符
                                    // 正确做法是，把导出的reshape算子节点，从1, n, 85，改为-1, n, 85。这样多batch时，动态维度是batch
                                    INFOE("Invalid model, s must be less num_stride[%d]", num_stride);
                                    continue;
                                }

                                int real_i = i - stride_begin[s];
                                int grid_x = real_i % fm_sizes[s];
                                int grid_y = real_i / fm_sizes[s];
                                int stride = strides[s];
                                float cx   = (pos_ptr[0] + grid_x) * stride;
                                float cy   = (pos_ptr[1] + grid_y) * stride;
                                float bw   = exp(pos_ptr[2]) * stride;
                                float bh   = exp(pos_ptr[3]) * stride;
                                float left   = cx - bw / 2;
                                float top    = cy - bh / 2;
                                float right  = cx + bw / 2;
                                float bottom = cy + bh / 2;

                                // 将获取到的框进行反映射回去
                                tie(left, top)     = affine_project(left,  top,    affine_matrix.d2i);
                                tie(right, bottom) = affine_project(right, bottom, affine_matrix.d2i);
                                image_based_boxes.emplace_back(left, top, right, bottom, confidence, class_label);
                            }
                        }
                    }

                    // 对整个图做nms，不同类别之间互不干扰
                    nms(image_based_boxes);
                    job.pro->set_value(job.boxes);
                }
                fetch_jobs.clear();
            }
            INFOV("Engine destroy.");
        }

        virtual vector<shared_future<box_array>> commits(const vector<Mat>& images) override{
            
            vector<shared_future<box_array>> results(images.size());

            #pragma omp paralell for
            for(int i = 0; i < images.size(); ++i){

                auto& image  = images[i];
                auto& result = results[i];

                Job job;
                job.pro = make_shared<promise<box_array>>();

                Size input_size(input_width_, input_height_);
                job.affine_matrix.compute(image.size(), input_size);

                cv::cvtColor(image, job.image, cv::COLOR_BGR2RGB);
                cv::warpAffine(job.image, job.image, job.affine_matrix.i2d_mat(), input_size, 1, cv::BORDER_CONSTANT, cv::Scalar::all(114));
                job.image.convertTo(job.image, CV_32F, 1 / 255.0f);
                {
                    unique_lock<mutex> l(jobs_lock_);
                    jobs_.push(job);
                };
                result = job.pro->get_future();
            }
            cond_.notify_all();
            return results;
        }

        virtual shared_future<vector<ObjectBox>> commit(const Mat& image) override{

            Job job;
            job.pro = make_shared<promise<box_array>>();

            Size input_size(input_width_, input_height_);
            job.affine_matrix.compute(image.size(), input_size);

            cv::cvtColor(image, job.image, cv::COLOR_BGR2RGB);
            cv::warpAffine(job.image, job.image, job.affine_matrix.i2d_mat(), input_size, 1, cv::BORDER_CONSTANT, cv::Scalar::all(114));
            job.image.convertTo(job.image, CV_32F, 1 / 255.0f);
            {
                unique_lock<mutex> l(jobs_lock_);
                jobs_.push(job);
            };
            cond_.notify_one();
            return job.pro->get_future();
        }

    private:
        int input_width_ = 0;
        int input_height_ = 0;
        shared_ptr<thread> worker_;
        atomic<bool> run_;
        mutex jobs_lock_;
        condition_variable cond_;
        queue<Job> jobs_;
    };

    shared_ptr<Infer> create_infer(const string& engine_file, int gpuid){
        shared_ptr<InferImpl> instance(new InferImpl());
        if(!instance->startup(engine_file, gpuid)){
            instance.reset();
        }
        return instance;
    }
};