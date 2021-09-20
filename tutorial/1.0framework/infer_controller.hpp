#ifndef INFER_CONTROLLER_HPP
#define INFER_CONTROLLER_HPP

#include <string>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include <infer/trt_infer.hpp>
#include "monopoly_allocator.hpp"


/* 
中文注释请下滑
//*Overview
Just do a quick recap. 
create_infer ===> class InferImpl->startup ===> ControllerImpl->startup
                                                (InferController->startup)

Just keep a closer eye to startup in @infer_controller.hpp
We set the run_ status to true. Then get the start_param_ which are trt_file and gpuid.
At the same time, we start up a thread with a worker function (InferController::worker)

Now look at InferController::worker in this page (use ctrl + f if you are in vscode)
We might be surprised to find that virtual void worker(std::promise<bool>& result) = 0;
that means it's a virtual function, which implements not here but ususally in its child class(plz look at @yolo.cpp)

The following description requires you to jump back and forth between @infer_controller.hpp and @yolo.cpp


In the worker() of @yolo.cpp (keep in mind that one thread one worker in the tutorial)
We build up some basic settings in worker(), for example:
    - which gpu you gonna use
    - max num of bboxes that can be detected in an image
    - how many elements to describe a bbox
    - where is the affine_matrix and how much space they take
    - specify input and output of the network(which node in onnx matches with which node in trt)
    - h w of the image and h w of the net_input
    - stream
    - open up a 2 x max_batch_size space specifically as a buffer for the incomming batch images.
    - create a tensor to store the result of the inference.(e.g. to store the output bbox)
    - create a fetch_jobs vector for the current worker

After these all settings, the worker waits for the input commits from the app_yolo. If you feel difficult to 
undertand it, feel free to have a look at the vivid image at ./tutorial/pipeline.jpg. You will also learn about
some other important concepts that making up the high performance system.

It's highly recommeded to set the pipeline.jpg on the left of the comments.
    Two important parts are listed with ======>

    1. =============> get_jobs_and_wait part (@infer_controller.hpp)
    the worker will wait until the run_ is false or jobs_ queue contains sth.
    if there is job in the queue, cond_.wait(....) return true
    and the get_jobs_and_wait program continue.

    On the one hand, the current worker keeps picking job from jobs_ queue until the max_size(e.g. 16),
    on the other hand, the we pop the picked job out of the queue.

    Jump back to the while(get_jobs_and_wait...) in @yolo.cpp. Now the current thread has already
    got the 16 images in its fetch_jobs such that it can do the inference.
    


    2. =============> commits part  (@infer_controller.hpp)
    Let's say we have 56 images in the ./workspace/inference
    We pass them as a std::vector 
    inputs size: 56    
    jobs size: 56
    batchsize: 16*2 (monopoly tensor size)


    #################### A supplementary note ##########################
    !Note that the class Input and class Output has been declared in the following code by template
    using ControllerImpl = InferController
    <
        Mat,                    // input
        ObjectBoxArray,         // output
        tuple<string, int>,     // start param
        AffineMatrix            // additional
    >;  
    Therefore in @infer_controller.hpp, with the template<class Input, class Output, class StartParam=std::tuple<std::string, int>, class JobAdditional=int> on the top
    of class InferController, we can use something like <Input>  <Output> etc.
    ###################################################################
    

    We split the inputs by batchsize(32) such that we get 2 batches(32 + 24).
    
    For i in 2 epochs:(for 32 and 24)
        for j in [32 then 24 images]:
            pass only one job-image pair to preprocessing function

    Have a glance at what the job is in @infer_controller.hpp. A point worth noting is job.additional(default: int) is actually AffineMatrix. Therefore, job-image pair not only contains 
    the info of the image(input) but also the info of the transformation(i2d, d2i) and the result(output).


    Now go into virtual bool preprocess()
    We request a monopolic space for image to be preprocessed. Within the preprocessing func,
    THe main goal is to do the warpaffine transformation.

    An example can be the following: @(src/tensorRT/common/infer_controller.hpp)
    We process batch A on GPU, then push it into the queue jobs_. Then we preprocess the next batch(batch B) while infer the batch A on GPU. 

    Note that before cond_.notify_one(), which means notify one thread to process the jobs(e.g. 32 images). The jobs_ queue already has 32 jobs. If we conitue with notify_one(), then get_jobs_and_wait part will continue. 
    In a recap, once the jobs_ queue has 32 jobs, a thread will be notifed to fetch the new jobs from jobs_ queue.


*概述
简单回顾一下。
create_infer ===> class InferImpl->startup ===> ControllerImpl->startup .

请仔细观察 @infer_controller.hpp中的startup
我们将run_ status设置为true。然后获取start_param_，即trt_file和gpuid。
同时，我们用一个worker函数(InferController::worker)启动一个线程。
现在在这个页面中查看InferController::worker(如果你在vscode中建议使用ctrl + f)
我们可能会惊讶地发现virtual void worker(std::promise<bool>& result) = 0;
这意味着它是一个虚函数，它不是在这里实现的，而是在它的子类中实现的(请看@yolo.cpp)
下面的描述建议您在@infer_controller.hpp和@yolo.cpp之间来回跳转


在@yolo.cpp的worker()中(请记住，在本教程中一个线程就是一个worker)
我们在worker()中构建了一些基本设置，例如:
- 你要用哪个gpu
- 图像中可检测到的最多框数
- 一个bbox需要多少元素来描述
- 仿射矩阵放在哪里和它们占用多少空间
- 指定网络的输入和输出(onnx中的哪个节点与trt中的哪个节点匹配)
- 图像的h w和net_input的h w
- stream
- 开辟一个2 x max_batch_size的空间，专门作为接收批处理图像的缓冲区。
- 创建一个张量来存储推断的结果。存储输出框
- 为当前worker创建一个fetch_jobs向量

在所有这些设置之后，worker将等待app_yolo提交的输入。如果你觉得很难
理解它，可以在./tutorial/pipeline.jpg 查看示意图。你会学到其他的一些组成高性能系统的重要概念。

强烈建议将pipeline.jpg放在旁边对比阅读，使用效果更加。

======>两个重要部分

1. =============> get_jobs_and_wait部分(@infer_controller.hpp)

worker将一直等待，直到run_为false或jobs_队列中得到了jobs。
如果队列中有jobs且notify_one了，则cond_.wait(....)返回true，get_jobs_and_wait程序继续执行。
一方面，当前的worker一直从jobs_队列中拿job，直到max_size(例如.16), 另一方面，我们将job从队列中取出。
跳转回while(get_jobs_and_wait…) @yolo.cpp。现在，当前线程已经拿到了16个图像，并放在fetch_jobs中，我们可以推理了。


2. =============> 提交部分(@infer_controller.hpp)
假设我们有56张图片在 ./workspace/inference
我们将它们作为std::vector传入进去。
    inputs size: 56    
    jobs size: 56
    batchsize: 16*2 (monopoly tensor size)

#################### 补充说明 ##########################
注意，类Input和类Output已经由模板声明了
using ControllerImpl = InferController
    using ControllerImpl = InferController
    <
        Mat,                    // input
        ObjectBoxArray,         // output
        tuple<string, int>,     // start param
        AffineMatrix            // additional
    >;  

因此，在@infer_controller.hpp中，template<class Input, class Output, class StartParam=std::tuple<std::string, int>， class JobAdditional=int>
在类InferController中，我们可以使用<Input> <Output>之类的东西。

########################################################

我们将输入按批次大小(32)分割，这样我们得到2批(32 + 24)。

    For i in 2 epochs:(for 32 and 24)
        for j in [32 then 24 images]:
            pass only one job-image pair to preprocessing function


现在你可以瞥一眼@infer_controller.hpp中的struct job是什么。值得注意的一点是job.additional(默认:int)实际上是affinmatrix。因此，job-image不仅包含
图像的信息(输入)，还有变换矩阵的信息(i2d, d2i)和结果(输出)。

现在去看看virtual bool preprocess()
我们请求对图像进行预处理的独占空间。在预处理函数中，我们的主要目标是进行warpaffine转换。

例:
我们在GPU上处理批处理A，然后将其推入队列jobs_。然后我们预处理下一批(批B)，同时在GPU上推断批A。
注意，在cond_.notify_one()之前，(即通知一个线程来处理jobs 例如:32张图), jobs_队列已经有32个jobs。如果notify_one()了，则get_jobs_and_wait部分会返回true，然后进入循环。

总结一下就是，一旦jobs_队列有32个jobs，线程将被通知从jobs_队列中获取新的jobs。


*/



//* kp template
template<class Input, class Output, class StartParam=std::tuple<std::string, int>, class JobAdditional=int>
class InferController{
public:
    struct Job{
        Input input;
        Output output;
        JobAdditional additional;
        MonopolyAllocator<TRT::Tensor>::MonopolyDataPointer mono_tensor;
        std::shared_ptr<std::promise<Output>> pro;
    };

    virtual ~InferController(){
        stop();
    }

    void stop(){
        run_ = false;
        cond_.notify_all();

        ////////////////////////////////////////// cleanup jobs
        {
            std::unique_lock<std::mutex> l(jobs_lock_);
            while(!jobs_.empty()){
                auto& item = jobs_.front();
                if(item.pro)
                    item.pro->set_value(Output());
                jobs_.pop();
            }
        };

        if(worker_){
            worker_->join();
            worker_.reset();
        }
    }

    bool startup(const StartParam& param){
        run_ = true;

        std::promise<bool> pro; //* kp: promise and future: https://www.youtube.com/watch?v=XDZkyQVsbDY
        start_param_ = param;   // trtmodel file name and gpuid
        worker_      = std::make_shared<std::thread>(&InferController::worker, this, std::ref(pro)); //* kp: shared_ptr  (highly recommended)ref: https://www.geeksforgeeks.org/auto_ptr-unique_ptr-shared_ptr-weak_ptr-2/#:~:text=unique_ptr%20to%20another.-,shared_ptr,-A%20shared_ptr%20is
                                                                                                     //* kp: std::thread   Here we spawn a new thread that calls    aref: https://www.cplusplus.com/reference/thread/thread/?kw=thread
                                                                                                     //  how to construct a thread with args  //*ref: https://www.cplusplus.com/reference/thread/thread/thread/#:~:text=A%20pointer%20to%20function
        return pro.get_future().get(); // the line 'result.set_value(true);' in yolo.cpp will be returned here.                
    }

    virtual std::shared_future<Output> commit(const Input& input){

        Job job;
        job.pro = std::make_shared<std::promise<Output>>();
        if(!preprocess(job, input)){
            job.pro->set_value(Output());
            return job.pro->get_future();
        }
        
        ///////////////////////////////////////////////////////////
        {
            std::unique_lock<std::mutex> l(jobs_lock_);  //* kp: unique_lock   It unblocks when it goes out of the scope (curly bracket).
            jobs_.push(job);
        };
        cond_.notify_one();
        return job.pro->get_future();
    }

    virtual std::vector<std::shared_future<Output>> commits(const std::vector<Input>& inputs){

        int batch_size = std::min((int)inputs.size(), this->tensor_allocator_->capacity());
        std::vector<Job> jobs(inputs.size());
        std::vector<std::shared_future<Output>> results(inputs.size());   // intputs.size: 56   batch_size = monopoly_size:16*2 = 32

        int nepoch = (inputs.size() + batch_size - 1) / batch_size;       // if the num of inputs are bigger than batch size, several epoch are needed.
        for(int epoch = 0; epoch < nepoch; ++epoch){
            int begin = epoch * batch_size;
            int end   = std::min((int)inputs.size(), begin + batch_size); // after we set the begin and end point for this batch

            for(int i = begin; i < end; ++i){                             // we for loop this batch in one epoch
                Job& job = jobs[i];
                job.pro = std::make_shared<std::promise<Output>>();
                if(!preprocess(job, inputs[i])){
                    job.pro->set_value(Output());
                }
                results[i] = job.pro->get_future();
            }
            
            ///////////////////////////////////////////////////////////
            {
                std::unique_lock<std::mutex> l(jobs_lock_);
                for(int i = begin; i < end; ++i){
                    jobs_.emplace(std::move(jobs[i]));
                };
            }
            cond_.notify_one();  //* kp: notify_one and wait is a pair. 
        }
        return results;
    }

protected:
    virtual void worker(std::promise<bool>& result) = 0; //* kp: virtual function
    virtual bool preprocess(Job& job, const Input& input) = 0;
    
    virtual bool get_jobs_and_wait(std::vector<Job>& fetch_jobs, int max_size){

        std::unique_lock<std::mutex> l(jobs_lock_);
        cond_.wait(l, [&](){
            return !run_ || !jobs_.empty();
        }); //* kp : condition variable and lambda expression.  

        if(!run_) return false;
        
        fetch_jobs.clear();
        for(int i = 0; i < max_size && !jobs_.empty(); ++i){
            fetch_jobs.emplace_back(std::move(jobs_.front()));
            jobs_.pop();
        }
        return true;
    }

    virtual bool get_job_and_wait(Job& fetch_job){

        std::unique_lock<std::mutex> l(jobs_lock_);
        cond_.wait(l, [&](){                 // Wait causes the current thread to block until the condition variable is notified.
            return !run_ || !jobs_.empty();  // When the pred func returns trueref: https://en.cppreference.com/w/cpp/thread/condition_variable/wait 
        });                                  // Waiting will be over if return true.

        if(!run_) return false;
        
        fetch_job = std::move(jobs_.front());
        jobs_.pop();
        return true;
    }

protected:
    StartParam start_param_;
    std::atomic<bool> run_;
    std::mutex jobs_lock_;
    std::queue<Job> jobs_;
    std::shared_ptr<std::thread> worker_;
    std::condition_variable cond_;
    std::shared_ptr<MonopolyAllocator<TRT::Tensor>> tensor_allocator_;
};

#endif // INFER_CONTROLLER_HPP