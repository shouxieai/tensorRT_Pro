

#include "trt_infer.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <NvInfer.h>
#include <NvCaffeParser.h>
#include <NvInferPlugin.h>
#include <cuda_fp16.h>
#include <common/cuda_tools.hpp>

using namespace nvinfer1;
using namespace std;

class Logger : public ILogger {
public:
	virtual void log(Severity severity, const char* msg) noexcept override {

		if (severity == Severity::kINTERNAL_ERROR) {
			INFOE("NVInfer INTERNAL_ERROR: %s", msg);
			abort();
		}else if (severity == Severity::kERROR) {
			INFOE("NVInfer: %s", msg);
		}
		else  if (severity == Severity::kWARNING) {
			INFOW("NVInfer: %s", msg);
		}
		else  if (severity == Severity::kINFO) {
			INFOD("NVInfer: %s", msg);
		}
		else {
			INFOD("%s", msg);
		}
	}
};
static Logger gLogger;

namespace TRT {

	////////////////////////////////////////////////////////////////////////////////
	template<typename _T>
	static void destroy_nvidia_pointer(_T* ptr) {
		if (ptr) ptr->destroy();
	}

	class EngineContext {
	public:
		virtual ~EngineContext() { destroy(); }

		void set_stream(CUStream stream){

			if(owner_stream_){
				if (stream_) {cudaStreamDestroy(stream_);}
				owner_stream_ = false;
			}
			stream_ = stream;
		}

		bool build_model(const void* pdata, size_t size) {
			destroy();

			if(pdata == nullptr || size == 0)
				return false;

			owner_stream_ = true;
			checkCudaRuntime(cudaStreamCreate(&stream_));
			if(stream_ == nullptr)
				return false;

			runtime_ = shared_ptr<IRuntime>(createInferRuntime(gLogger), destroy_nvidia_pointer<IRuntime>);
			if (runtime_ == nullptr)
				return false;

			engine_ = shared_ptr<ICudaEngine>(runtime_->deserializeCudaEngine(pdata, size, nullptr), destroy_nvidia_pointer<ICudaEngine>);
			if (engine_ == nullptr)
				return false;

			//runtime_->setDLACore(0);
			context_ = shared_ptr<IExecutionContext>(engine_->createExecutionContext(), destroy_nvidia_pointer<IExecutionContext>);
			return context_ != nullptr;
		}

	private:
		void destroy() {
			context_.reset();
			engine_.reset();
			runtime_.reset();

			if(owner_stream_){
				if (stream_) {cudaStreamDestroy(stream_);}
			}
			stream_ = nullptr;
		}

	public:
		cudaStream_t stream_ = nullptr;
		bool owner_stream_ = false;
		shared_ptr<IExecutionContext> context_;
		shared_ptr<ICudaEngine> engine_;
		shared_ptr<IRuntime> runtime_ = nullptr;
	};

	class InferImpl : public Infer {

	public:
		virtual ~InferImpl();
		virtual bool load(const std::string& file);
		virtual bool load_from_memory(const void* pdata, size_t size);
		virtual void destroy();
		virtual void forward(bool sync) override;
		virtual int get_max_batch_size() override;
		virtual CUStream get_stream() override;
		virtual void set_stream(CUStream stream) override;
		virtual void synchronize() override;
		virtual size_t get_device_memory_size() override;
		virtual std::shared_ptr<MixMemory> get_workspace() override;
		virtual std::shared_ptr<Tensor> input(int index = 0) override;
		virtual std::string get_input_name(int index = 0) override;
		virtual std::shared_ptr<Tensor> output(int index = 0) override;
		virtual std::string get_output_name(int index = 0) override;
		virtual std::shared_ptr<Tensor> tensor(const std::string& name) override;
		virtual bool is_output_name(const std::string& name) override;
		virtual bool is_input_name(const std::string& name) override;
		virtual void set_input (int index, std::shared_ptr<Tensor> tensor) override;
		virtual void set_output(int index, std::shared_ptr<Tensor> tensor) override;
		virtual std::shared_ptr<std::vector<uint8_t>> serial_engine() override;

		virtual void print() override;

		virtual int num_output();
		virtual int num_input();
		virtual int device() override;

	private:
		void build_engine_input_and_outputs_mapper();

	private:
		std::vector<std::shared_ptr<Tensor>> inputs_;
		std::vector<std::shared_ptr<Tensor>> outputs_;
		std::vector<int> inputs_map_to_ordered_index_;
		std::vector<int> outputs_map_to_ordered_index_;
		std::vector<std::string> inputs_name_;
		std::vector<std::string> outputs_name_;
		std::vector<std::shared_ptr<Tensor>> orderdBlobs_;
		std::map<std::string, int> blobsNameMapper_;
		std::shared_ptr<EngineContext> context_;
		std::vector<void*> bindingsPtr_;
		std::shared_ptr<MixMemory> workspace_;
		int device_ = 0;
	};

	////////////////////////////////////////////////////////////////////////////////////
	InferImpl::~InferImpl(){
		destroy();
	}

	void InferImpl::destroy() {

		int old_device = 0;
		checkCudaRuntime(cudaGetDevice(&old_device));
		checkCudaRuntime(cudaSetDevice(device_));
		this->context_.reset();
		this->blobsNameMapper_.clear();
		this->outputs_.clear();
		this->inputs_.clear();
		this->inputs_name_.clear();
		this->outputs_name_.clear();
		checkCudaRuntime(cudaSetDevice(old_device));
	}

	void InferImpl::print(){
		if(!context_){
			INFOW("Infer print, nullptr.");
			return;
		}

		INFO("Infer %p detail", this);
		INFO("\tBase device: %s", CUDATools::device_description().c_str());
		INFO("\tMax Batch Size: %d", this->get_max_batch_size());
		INFO("\tInputs: %d", inputs_.size());
		for(int i = 0; i < inputs_.size(); ++i){
			auto& tensor = inputs_[i];
			auto& name = inputs_name_[i];
			INFO("\t\t%d.%s : shape {%s}, %s", i, name.c_str(), tensor->shape_string(), data_type_string(tensor->type()));
		}

		INFO("\tOutputs: %d", outputs_.size());
		for(int i = 0; i < outputs_.size(); ++i){
			auto& tensor = outputs_[i];
			auto& name = outputs_name_[i];
			INFO("\t\t%d.%s : shape {%s}, %s", i, name.c_str(), tensor->shape_string(), data_type_string(tensor->type()));
		} 
	}

	std::shared_ptr<std::vector<uint8_t>> InferImpl::serial_engine() {
		auto memory = this->context_->engine_->serialize();
		auto output = make_shared<std::vector<uint8_t>>((uint8_t*)memory->data(), (uint8_t*)memory->data()+memory->size());
		memory->destroy();
		return output;
	}

	bool InferImpl::load_from_memory(const void* pdata, size_t size) {

		if (pdata == nullptr || size == 0)
			return false;

		context_.reset(new EngineContext());

		//build model
		if (!context_->build_model(pdata, size)) {
			context_.reset();
			return false;
		}

		workspace_.reset(new MixMemory());
		cudaGetDevice(&device_);
		build_engine_input_and_outputs_mapper();
		return true;
	}

	bool InferImpl::load(const std::string& file) {

		auto data = iLogger::load_file(file);
		if (data.empty())
			return false;

		context_.reset(new EngineContext());

		//build model
		if (!context_->build_model(data.data(), data.size())) {
			context_.reset();
			return false;
		}

		workspace_.reset(new MixMemory());
		cudaGetDevice(&device_);
		build_engine_input_and_outputs_mapper();
		return true;
	}

	size_t InferImpl::get_device_memory_size() {
		EngineContext* context = (EngineContext*)this->context_.get();
		return context->context_->getEngine().getDeviceMemorySize();
	}

	static TRT::DataType convert_trt_datatype(nvinfer1::DataType dt){
		switch(dt){
			case nvinfer1::DataType::kFLOAT: return TRT::DataType::Float;
			case nvinfer1::DataType::kHALF: return TRT::DataType::Float16;
			case nvinfer1::DataType::kINT32: return TRT::DataType::Int32;
			default:
				INFOE("Unsupport data type %d", dt);
				return TRT::DataType::Float;
		}
	}

	void InferImpl::build_engine_input_and_outputs_mapper() {
		
		EngineContext* context = (EngineContext*)this->context_.get();
		int nbBindings = context->engine_->getNbBindings();
		int max_batchsize = context->engine_->getMaxBatchSize();

		inputs_.clear();
		inputs_name_.clear();
		outputs_.clear();
		outputs_name_.clear();
		orderdBlobs_.clear();
		bindingsPtr_.clear();
		blobsNameMapper_.clear();
		for (int i = 0; i < nbBindings; ++i) {

			auto dims = context->engine_->getBindingDimensions(i);
			auto type = context->engine_->getBindingDataType(i);
			const char* bindingName = context->engine_->getBindingName(i);
			dims.d[0] = max_batchsize;
			auto newTensor = make_shared<Tensor>(dims.nbDims, dims.d, convert_trt_datatype(type));
			newTensor->set_stream(this->context_->stream_);
			newTensor->set_workspace(this->workspace_);
			if (context->engine_->bindingIsInput(i)) {
				//if is input
				inputs_.push_back(newTensor);
				inputs_name_.push_back(bindingName);
				inputs_map_to_ordered_index_.push_back(orderdBlobs_.size());
			}
			else {
				//if is output
				outputs_.push_back(newTensor);
				outputs_name_.push_back(bindingName);
				outputs_map_to_ordered_index_.push_back(orderdBlobs_.size());
			}
			blobsNameMapper_[bindingName] = i;
			orderdBlobs_.push_back(newTensor);
		}
		bindingsPtr_.resize(orderdBlobs_.size());
	}

	void InferImpl::set_stream(CUStream stream){
		this->context_->set_stream(stream);

		for(auto& t : orderdBlobs_)
			t->set_stream(stream);
	}

	CUStream InferImpl::get_stream() {
		return this->context_->stream_;
	}

	int InferImpl::device() {
		return device_;
	}

	void InferImpl::synchronize() {
		checkCudaRuntime(cudaStreamSynchronize(context_->stream_));
	}

	bool InferImpl::is_output_name(const std::string& name){
		return std::find(outputs_name_.begin(), outputs_name_.end(), name) != outputs_name_.end();
	}

	bool InferImpl::is_input_name(const std::string& name){
		return std::find(inputs_name_.begin(), inputs_name_.end(), name) != inputs_name_.end();
	}

	void InferImpl::forward(bool sync) {

		EngineContext* context = (EngineContext*)context_.get();
		int inputBatchSize = inputs_[0]->size(0);
		for(int i = 0; i < context->engine_->getNbBindings(); ++i){
			auto dims = context->engine_->getBindingDimensions(i);
			auto type = context->engine_->getBindingDataType(i);
			dims.d[0] = inputBatchSize;
			if(context->engine_->bindingIsInput(i)){
				context->context_->setBindingDimensions(i, dims);
			}
		}

		for (int i = 0; i < outputs_.size(); ++i) {
			outputs_[i]->resize_single_dim(0, inputBatchSize);
			outputs_[i]->to_gpu(false);
		}

		for (int i = 0; i < orderdBlobs_.size(); ++i)
			bindingsPtr_[i] = orderdBlobs_[i]->gpu();

		void** bindingsptr = bindingsPtr_.data();
		//bool execute_result = context->context_->enqueue(inputBatchSize, bindingsptr, context->stream_, nullptr);
		bool execute_result = context->context_->enqueueV2(bindingsptr, context->stream_, nullptr);
		if(!execute_result){
			auto code = cudaGetLastError();
			INFOF("execute fail, code %d[%s], message %s", code, cudaGetErrorName(code), cudaGetErrorString(code));
		}

		if (sync) {
			synchronize();
		}
	}

	std::shared_ptr<MixMemory> InferImpl::get_workspace() {
		return workspace_;
	}

	int InferImpl::num_input() {
		return static_cast<int>(this->inputs_.size());
	}

	int InferImpl::num_output() {
		return static_cast<int>(this->outputs_.size());
	}

	void InferImpl::set_input (int index, std::shared_ptr<Tensor> tensor){
		
		if(index < 0 || index >= inputs_.size()){
			INFOF("Input index[%d] out of range [size=%d]", index, inputs_.size());
		}

		this->inputs_[index] = tensor;
		int order_index = inputs_map_to_ordered_index_[index];
		this->orderdBlobs_[order_index] = tensor;
	}

	void InferImpl::set_output(int index, std::shared_ptr<Tensor> tensor){

		if(index < 0 || index >= outputs_.size()){
			INFOF("Output index[%d] out of range [size=%d]", index, outputs_.size());
		}

		this->outputs_[index] = tensor;
		int order_index = outputs_map_to_ordered_index_[index];
		this->orderdBlobs_[order_index] = tensor;
	}

	std::shared_ptr<Tensor> InferImpl::input(int index) {
		if(index < 0 || index >= inputs_.size()){
			INFOF("Input index[%d] out of range [size=%d]", index, inputs_.size());
		}
		return this->inputs_[index];
	}

	std::string InferImpl::get_input_name(int index){
		if(index < 0 || index >= inputs_name_.size()){
			INFOF("Input index[%d] out of range [size=%d]", index, inputs_name_.size());
		}
		return inputs_name_[index];
	}

	std::shared_ptr<Tensor> InferImpl::output(int index) {
		if(index < 0 || index >= outputs_.size()){
			INFOF("Output index[%d] out of range [size=%d]", index, outputs_.size());
		}
		return outputs_[index];
	}

	std::string InferImpl::get_output_name(int index){
		if(index < 0 || index >= outputs_name_.size()){
			INFOF("Output index[%d] out of range [size=%d]", index, outputs_name_.size());
		}
		return outputs_name_[index];
	}

	int InferImpl::get_max_batch_size() {
		Assert(this->context_ != nullptr);
		return this->context_->engine_->getMaxBatchSize();
	}

	std::shared_ptr<Tensor> InferImpl::tensor(const std::string& name) {

		auto node = this->blobsNameMapper_.find(name);
		if(node == this->blobsNameMapper_.end()){
			INFOF("Could not found the input/output node '%s', please makesure your model", name.c_str());
		}
		return orderdBlobs_[node->second];
	}

	std::shared_ptr<Infer> load_infer_from_memory(const void* pdata, size_t size){

		std::shared_ptr<InferImpl> Infer(new InferImpl());
		if (!Infer->load_from_memory(pdata, size))
			Infer.reset();
		return Infer;
	}

	std::shared_ptr<Infer> load_infer(const string& file) {
		
		std::shared_ptr<InferImpl> Infer(new InferImpl());
		if (!Infer->load(file))
			Infer.reset();
		return Infer;
	}

	DeviceMemorySummary get_current_device_summary() {
		DeviceMemorySummary info;
		checkCudaRuntime(cudaMemGetInfo(&info.available, &info.total));
		return info;
	}

	int get_device_count() {
		int count = 0;
		checkCudaRuntime(cudaGetDeviceCount(&count));
		return count;
	}

	int get_device() {
		int device = 0;
		checkCudaRuntime(cudaGetDevice(&device));
		return device;
	}

	void set_device(int device_id) {
		if (device_id == -1)
			return;

		checkCudaRuntime(cudaSetDevice(device_id));
	}

	bool init_nv_plugins() {

		bool ok = initLibNvInferPlugins(&gLogger, "");
		if (!ok) {
			INFOE("init lib nvinfer plugins failed.");
		}
		return ok;
	}
};