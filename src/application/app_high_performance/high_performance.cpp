
#include "high_performance.hpp"

namespace HighPerformance{

    Pipeline::Pipeline(int max_cache){
        run_       = true;
        max_cache_ = max_cache;
    }

    Pipeline::~Pipeline(){
        stop();
    }

    void Pipeline::stop(){
        run_ = false;
        wait_cond_.notify_all();
        cond_.notify_all();
    }

    void Pipeline::commit(const shared_future<shared_ptr<Data>>& input){

        if(!wait_if_full()) return;
        ///////////////////////////////////////////////////////////
        {
            unique_lock<mutex> l(jobs_lock_);
            jobs_.push(input);
        };
        cond_.notify_one();
    }

    bool Pipeline::get_job_and_wait(Job& fetch_job){

        unique_lock<mutex> l(jobs_lock_);
        cond_.wait(l, [&](){
            return !run_ || !jobs_.empty();
        });

        if(!run_) return false;
        
        fetch_job = move(jobs_.front());
        jobs_.pop();
        wait_cond_.notify_one();
        return true;
    }

    bool Pipeline::wait_if_full() {

        if(jobs_.size() >= max_cache_){
            unique_lock<mutex> l(wait_lock_);
            wait_cond_.wait(l, [&](){return !run_ || jobs_.size() < max_cache_;});
        }
        return run_;
    }

    ////////////////////////////////////////////////////////////
    Node::Node(){
    }

    Node::~Node(){
        stop();
    }

    void Node::stop(){
        run_ = false;
        for(auto& o : outputs_)
            o->stop();

        if(worker_.joinable()){
            worker_.join();
        }
        outputs_.clear();
    }

    void Node::add_input(shared_ptr<Pipeline> input){
        inputs_.push_back(input);
    }

    void Node::add_output(shared_ptr<Pipeline> output){
        outputs_.push_back(output);
    }

    void Node::startup(){
        run_    = true;
        worker_ = thread(&Node::worker, this);
    }

    void Node::worker(){
        vector<shared_future<shared_ptr<Data>>> inputs_future;
        vector<shared_ptr<Data>> inputs_data;
        while(get_inputs_future(inputs_future)){
            
            if(!get_inputs_data(inputs_future, inputs_data))
                break;

            forward(inputs_data);
        }
    }

    bool Node::get_inputs_data(vector<shared_future<shared_ptr<Data>>>& inputs_future, vector<shared_ptr<Data>>& inputs_data){

        if(!run_) return false;

        inputs_data.resize(inputs_future.size());
        for(int i = 0; i < inputs_future.size(); ++i){
            auto& fut = inputs_future[i];
            inputs_data[i] = fut.get();
            if(inputs_data[i] == nullptr)
                return false;
        }
        return true;
    }

    bool Node::get_inputs_future(vector<shared_future<shared_ptr<Data>>>& inputs_future){

        if(!run_) return false;

        inputs_future.resize(inputs_.size());
        for(int i = 0; i < inputs_.size(); ++i){
            auto& input = inputs_[i];
            if(!input->get_job_and_wait(inputs_future[i])){
                return false;
            }
        }
        return true;
    }

    void Node::forward(vector<shared_ptr<Data>>& inputs_data){};
    ////////////////////////////////////////////////////////////////

    void InputNode::startup(const function<void(vector<shared_ptr<Pipeline>>& output_pipe)>& worker){
        run_    = true;
        worker_ = thread(worker, ref(outputs_));
    }

    void InputNode::startup(){}
    void InputNode::worker(){}
    void InputNode::add_input(shared_ptr<Pipeline> input){}

    void OutputNode::startup(const function<void(vector<shared_ptr<Data>>& datas)>& callback){
        callback_ = callback;
        Node::startup();
    }

    void OutputNode::forward(vector<shared_ptr<Data>>& inputs_data){
        if(callback_) callback_(inputs_data);
    };
    
    void OutputNode::add_output(shared_ptr<Pipeline> input){}


    void connect(Node& output, Node& input, int max_cache){

        auto line = make_shared<Pipeline>(max_cache);
        output.add_output(line);
        input.add_input(line);
    }

    void connect(shared_ptr<Node>& output, shared_ptr<Node>& input, int max_cache){
        connect(*output, *input, max_cache);
    }

    shared_future<DataPtr> make_data_future(DataPtr ptr){
        promise<DataPtr> pro;
        pro.set_value(ptr);
        return pro.get_future();
    }

}; // namespace HihgPerformance