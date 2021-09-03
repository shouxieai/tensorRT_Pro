#ifndef HIGH_PERFORMANCE_HPP
#define HIGH_PERFORMANCE_HPP

#include <thread>
#include <mutex>
#include <future>
#include <memory>
#include <common/ilogger.hpp>
#include <queue>

namespace HighPerformance{

    using namespace std;

    class Data{
    public:
        virtual ~Data() = default;
        virtual const char* name(){return "Data";}
    };

    #define SetupData(classname)  virtual const char* name() override{return #classname;}

    #define DefData(classname, type)                    \
    class classname : public Data{                      \
    public:                                             \
        classname() = default;                          \
        classname(const type& data){                    \
            data_ = data;                               \
        }                                               \
                                                        \
        type& data(){return data_;}                     \
        const type& data() const{return data_;}         \
        SetupData(classname)                            \
                                                        \
    private:                                            \
        type data_;                                     \
    };

    DefData(StringData, string);
    DefData(IntData, int);
    DefData(FloatData, float);
    DefData(DoubleData, double);
    DefData(Int64Data, int64_t);

    typedef shared_ptr<Data> DataPtr;


    template<typename _T>
    class Container : public Data{
    public:
        SetupData(Container<_T>);

        template<typename ..._Args>
        Container(_Args&& ... args):value(std::forward<_Args>(args)...){}

        _T value;
    };


    template<typename _Tp, typename _Tp1>
    inline shared_ptr<_Tp>
    dynamic_obj_cast__(const shared_ptr<_Tp1>& __r, const char* file, int line) noexcept
    {
        if (_Tp* __p = dynamic_cast<_Tp*>(__r.get()))
            return shared_ptr<_Tp>(__r, __p);

        iLogger::__log_func(file, line, iLogger::LogLevel::Error, "dynamic_obj_cast failed ptr[%p], %s to %s is nullptr", __r.get(), typeid(_Tp1).name(), typeid(_Tp).name());
        return shared_ptr<_Tp>();
    }

    #define dynamic_obj_cast(p, ...)  dynamic_obj_cast__<__VA_ARGS__>(p, __FILE__, __LINE__)

    class Pipeline{
    public:
        typedef shared_future<shared_ptr<Data>> Job;

        Pipeline(int max_cache=30);
        virtual ~Pipeline();
        void stop();
        virtual void commit(const shared_future<shared_ptr<Data>>& input);
        virtual bool get_job_and_wait(Job& fetch_job);

    protected:
        virtual bool wait_if_full();

    protected:
        atomic<bool> run_;
        mutex jobs_lock_, wait_lock_;
        queue<Job> jobs_;
        condition_variable cond_, wait_cond_;
        int max_cache_ = 30;
    };

    class Node{
    public:
        Node();
        virtual ~Node();
        void stop();
        virtual void add_input(shared_ptr<Pipeline> input);
        virtual void add_output(shared_ptr<Pipeline> output);

    protected:
        virtual void startup();
        virtual void worker();
        bool get_inputs_data(vector<shared_future<shared_ptr<Data>>>& inputs_future, vector<shared_ptr<Data>>& inputs_data);
        bool get_inputs_future(vector<shared_future<shared_ptr<Data>>>& inputs_future);
        virtual void forward(vector<shared_ptr<Data>>& inputs_data);

    protected:
        atomic<bool> run_;
        thread worker_;
        vector<shared_ptr<Pipeline>> inputs_, outputs_;
    };

    class InputNode : public Node{
    public:
        virtual void startup(const function<void(vector<shared_ptr<Pipeline>>& output_pipe)>& worker);

    protected:
        virtual void startup() override;
        virtual void worker() override;

    private:
        virtual void add_input(shared_ptr<Pipeline> input) override;
    };

    class OutputNode : public Node{
    public:
        virtual void startup(const function<void(vector<shared_ptr<Data>>& datas)>& callback);

    protected:
        virtual void forward(vector<shared_ptr<Data>>& inputs_data)override;
        
    private:
        virtual void add_output(shared_ptr<Pipeline> input) override;

    private:
        function<void(vector<shared_ptr<Data>>& datas)> callback_;
    };

    void connect(Node& output, Node& input, int max_cache=30);
    void connect(shared_ptr<Node>& output, shared_ptr<Node>& input, int max_cache=30);
    shared_future<DataPtr> make_data_future(DataPtr ptr);
    
}; // namespace HihgPerformance


#endif // HIGH_PERFORMANCE_HPP