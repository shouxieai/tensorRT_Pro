

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
		}
		else if (severity == Severity::kERROR) {
			INFOE("NVInfer ERROR: %s", msg);
		}
		else  if (severity == Severity::kWARNING) {
			INFOW("NVInfer WARNING: %s", msg);
		}else{
			//INFO("NVInfer INFOV: %s", msg);
		}
	}
};
static Logger gLogger;

namespace TRTInfer {

	MemoryInfo get_current_device_memory_info() {
		MemoryInfo info;
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

	int data_type_size(DataType dt){
		switch (dt) {
		case DataType::dtFloat: return sizeof(float);

		#ifdef HAS_CUDA_HALF
		case DataType::dtHalfloat: return sizeof(halfloat);
		#endif

		default: {
			INFOE("Not support dtype: %d", dt);
			return -1;
		}
		}
	}

	MemoryManager::~MemoryManager() {
		release_all();
	}

	void* MemoryManager::gpu(size_t size) {

		if (gpu_size_ < size) {
			release_gpu();

			gpu_size_ = size;
			checkCudaRuntime(cudaMallocManaged(&gpu_, size));
			checkCudaRuntime(cudaMemset(gpu_, 0, size));
		}
		return gpu_;
	}

	void* MemoryManager::cpu(size_t size) {

		if (cpu_size_ < size) {
			release_cpu();

			cpu_size_ = size;
			checkCudaRuntime(cudaMallocHost(&cpu_, size));
			Assert(cpu_ != nullptr);
			memset(cpu_, 0, size);
		}
		return cpu_;
	}

	void MemoryManager::release_cpu() {
		if (cpu_) {
			checkCudaRuntime(cudaFreeHost(cpu_));
			cpu_ = nullptr;
		}
		cpu_size_ = 0;
	}

	void MemoryManager::release_gpu() {
		if (gpu_) {
			checkCudaRuntime(cudaFree(gpu_));
			gpu_ = nullptr;
		}
		gpu_size_ = 0;
	}

	void MemoryManager::release_all() {
		release_cpu();
		release_gpu();
	}

	void Tensor::compute_shape_string(){
		char* buffer = shape_string_;
		size_t buffer_size = sizeof(shape_string_);
		for(int i = 0; i < shape_.size(); ++i){

			int size = 0;
			if(i < shape_.size() - 1)
				size = snprintf(buffer, buffer_size, "%d x ", shape_[i]);
			else
				size = snprintf(buffer, buffer_size, "%d", shape_[i]);

			buffer += size;
			buffer_size -= size;
		}
	}

	Tensor::Tensor(int n, int c, int h, int w, DataType dtType) {
		this->dtType_ = dtType;
		resize(n, c, h, w);
	}

	Tensor::~Tensor() {
		release();
	}

	Tensor::Tensor(const std::vector<int>& dims, DataType dtType):Tensor(dims.size(), dims.data(), dtType){}

	Tensor::Tensor(int ndims, const int* dims, DataType dtType) {

		this->dtType_ = dtType;
		resize(ndims, dims);
	}

	Tensor::Tensor(){}

	void Tensor::release() {
		memory_.release_all();
		shape_.clear();
		capacity_ = 0;
		bytes_ = 0;
		head_ = DataHead_Init;
	}

	bool Tensor::empty() {
		return memory_.cpu() == nullptr && memory_.gpu() == nullptr;
	}

	int Tensor::count(int start_axis) const {

		if(start_axis >= 0 && start_axis < shape_.size()){
			int size = 1;
			for (int i = start_axis; i < shape_.size(); ++i) 
				size *= shape_[i];
			return size;
		}else{
			return 0;
		}
	}

	void Tensor::resize(const std::vector<int>& dims) {
		resize(dims.size(), dims.data());
	}

	int Tensor::numel(){
		int value = shape_.empty() ? 0 : 1;
		for(int i = 0; i < shape_.size(); ++i){
			value *= shape_[i];
		}
		return value;
	}

	void Tensor::resize_dim(int idim, int size){

		Assert(idim >= 0 && idim < shape_.size());

		auto new_shape = shape_;
		new_shape[idim] = size;
		resize(new_shape);
	}

	void Tensor::resize(int ndims, const int* dims) {

		vector<int> setup_dims(ndims);
		int numel = ndims == 0 ? 0 : 1;
		for(int i = 0; i < ndims; ++i){
			int dim = dims[i];
			if(dim == -1){
				// 如果维度不同，则-1没有意义
				Assert(ndims == shape_.size());
				dim = shape_[i];
			}
			setup_dims[i] = dim;
			numel *= setup_dims[i];
		}

		int needed_size = numel * element_size();
		if (needed_size > capacity_) {
			release();

			this->bytes_ = needed_size;
			this->capacity_ = needed_size;
		}

		this->shape_ = setup_dims;
		this->bytes_ = needed_size;
		this->compute_shape_string();
	}

	void Tensor::synchronize(){ 
		checkCudaRuntime(cudaStreamSynchronize(stream_));
	}

	void Tensor::to_gpu(bool copyedIfCPU) {

		if (head_ == DataHead_InGPU)
			return;

		head_ = DataHead_InGPU;
		memory_.gpu(this->bytes_);

		if (copyedIfCPU && memory_.cpu() != nullptr) {
			checkCudaRuntime(cudaMemcpyAsync(memory_.gpu(), memory_.cpu(), bytes_, cudaMemcpyHostToDevice, stream_));
		}
	}
	
	void Tensor::to_cpu(bool copyedIfGPU) {

		if (head_ == DataHead_InCPU)
			return;

		head_ = DataHead_InCPU;
		memory_.cpu(bytes_);

		if (copyedIfGPU && memory_.gpu() != nullptr) {
			checkCudaRuntime(cudaMemcpyAsync(memory_.cpu(), memory_.gpu(), bytes_, cudaMemcpyDeviceToHost, stream_));
			checkCudaRuntime(cudaStreamSynchronize(stream_));
		}
	}

	void Tensor::to_float() {

		if (type() == DataType::dtFloat)
			return;

		#ifdef HAS_CUDA_HALF

			if (type() != DataType::dtHalfloat) {
				INFOF("not implement function");
			}

			auto c = count();
			float* convert_memory = (float*)malloc(c * data_type_size(DataType::dtFloat));
			float* dst = convert_memory;
			halfloat* src = cpu<halfloat>();

			for (int i = 0; i < c; ++i)
				*dst++ = *src++;

			this->dtType_ = DataType::dtFloat;

			resize(-1);
			memcpy(cpu(), convert_memory, bytes_);
			free(convert_memory);

		#else
			LOG(LFATAL) << "not implement function";
		#endif
	}

	#ifdef HAS_CUDA_HALF
	void Tensor::to_half() {

		if (type() == DataType::dtHalfloat)
			return;

		if (type() != DataType::dtFloat) {
			INFOF("not implement function");
		}

		auto c = count();
		halfloat* convert_memory = (halfloat*)malloc(c * data_type_size(DataType::dtHalfloat));
		halfloat* dst = convert_memory;
		float* src = cpu<float>();

		for (int i = 0; i < c; ++i) 
			*dst++ = *src++;

		this->dtType_ = DataType::dtHalfloat;
		resize(-1);
		memcpy(cpu(), convert_memory, bytes_);
		free(convert_memory);
	}
	#endif

	void Tensor::set_to(float value) {
		int c = count();
		if (dtType_ == DataType::dtFloat) {
			float* ptr = cpu<float>();
			for (int i = 0; i < c; ++i)
				*ptr++ = value;
		}
		else {
			#ifdef HAS_CUDA_HALF
				halfloat* ptr = cpu<halfloat>();
				for (int i = 0; i < c; ++i)
					*ptr++ = value;
			#else
				LOG(LFATAL) << "not implement function";
			#endif
		}
	}

	int Tensor::offset(const std::vector<int>& index){
		
		// 对于维度超出的，直接报错
		// 对于维度比shape_少的，则后面全为0
		Assert(index.size() <= shape_.size());
		int value = 0;
		for(int i = 0; i < shape_.size(); ++i){

			if(i < index.size())
				value += index[i];

			if(i + 1 < shape_.size())
				value *= shape_[i+1];
		}
		return value;
	}

	#ifdef USE_OPENCV
	void Tensor::set_norm_mat(int n, const cv::Mat& image, float mean[3], float std[3]) {

		Assert(image.channels() == 3 && !image.empty() && type() == DataType::dtFloat);
		Assert(shape_.size() == 4 && n < shape_[0]);
		to_cpu(false);

		int width = shape_[3];
		int height = shape_[2];
		float scale = 1 / 255.0;
		cv::Mat inputframe = image;
		if(inputframe.size() != cv::Size(width, height))
			cv::resize(inputframe, inputframe, cv::Size(width, height));

		inputframe.convertTo(inputframe, CV_32F, scale);

		cv::Mat ms[3];
		for (int c = 0; c < 3; ++c)
			ms[c] = cv::Mat(height, width, CV_32F, cpu<float>(n, c));

		split(inputframe, ms);
		Assert((void*)ms[0].data == (void*)cpu<float>(n));

		for (int c = 0; c < 3; ++c)
			ms[c] = (ms[c] - mean[c]) / std[c];
	}

	void Tensor::set_mat(int n, const cv::Mat& _image) {

		cv::Mat image = _image;
		Assert(!image.empty() && CV_MAT_DEPTH(image.type()) == CV_32F && type() == DataType::dtFloat);
		Assert(shape_.size() == 4 && n < shape_[0] && image.channels() == shape_[1]);
		to_cpu(false);

		int width = shape_[3];
		int height = shape_[2];
		if (image.size() != cv::Size(width, height))
			cv::resize(image, image, cv::Size(width, height));

		if (image.channels() == 1) {
			memcpy(cpu<float>(n), image.data, width * height * sizeof(float));
			return;
		}

		vector<cv::Mat> ms(image.channels());
		for (int i = 0; i < ms.size(); ++i) 
			ms[i] = cv::Mat(height, width, CV_32F, cpu<float>(n, i));

		cv::split(image, &ms[0]);
		Assert((void*)ms[0].data == (void*)cpu<float>(n));
	}
	#endif // USE_OPENCV

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

		bool build_model(const vector<uint8_t>& data) {
			destroy();

			owner_stream_ = true;
			checkCudaRuntime(cudaStreamCreate(&stream_));
			if(stream_ == nullptr)
				return false;

			runtime_ = shared_ptr<IRuntime>(createInferRuntime(gLogger), destroy_nvidia_pointer<IRuntime>);
			if (runtime_ == nullptr)
				return false;

			engine_ = shared_ptr<ICudaEngine>(runtime_->deserializeCudaEngine(data.data(), data.size(), nullptr), destroy_nvidia_pointer<ICudaEngine>);
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
			pluginFactory_.reset();

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
		shared_ptr<nvinfer1::IPluginFactory> pluginFactory_;
		shared_ptr<IRuntime> runtime_ = nullptr;
	};

	class EngineImpl : public Engine {

	public:
		virtual bool load(const std::string& file);
		virtual void destroy();
		virtual void forward(bool sync = true) override;
		virtual int get_max_batch_size() override;
		virtual CUStream get_stream() override;
		virtual void set_stream(CUStream stream) override;
		virtual void synchronize() override;
		virtual size_t get_device_memory_size() override;
		virtual std::shared_ptr<MemoryManager> get_workspace() override;
		virtual bool is_dynamic_batch_dimension() override;
		virtual std::shared_ptr<Tensor> input(int index = 0) override;
		virtual std::string get_input_name(int index = 0) override;
		virtual std::shared_ptr<Tensor> output(int index = 0) override;
		virtual std::string get_output_name(int index = 0) override;
		virtual std::shared_ptr<Tensor> tensor(const std::string& name) override;
		virtual bool is_output_name(const std::string& name) override;
		virtual bool is_input_name(const std::string& name) override;

		virtual void print() override;

		virtual int num_output();
		virtual int num_input();
		virtual int device() override;

	private:
		void build_engine_input_and_outputs_mapper();

	private:
		std::vector<std::shared_ptr<Tensor>> inputs_;
		std::vector<std::shared_ptr<Tensor>> outputs_;
		std::vector<std::string> inputs_name_;
		std::vector<std::string> outputs_name_;
		std::vector<std::shared_ptr<Tensor>> orderdBlobs_;
		std::map<std::string, int> blobsNameMapper_;
		std::shared_ptr<EngineContext> context_;
		std::vector<void*> bindingsPtr_;
		std::shared_ptr<MemoryManager> workspace_;
		int device_ = -1;
	};

	////////////////////////////////////////////////////////////////////////////////////
	void EngineImpl::destroy() {
		this->context_.reset();
		this->blobsNameMapper_.clear();
		this->outputs_.clear();
		this->inputs_.clear();
		this->inputs_name_.clear();
		this->outputs_name_.clear();
	}

	bool EngineImpl::is_dynamic_batch_dimension(){
		return context_->engine_->hasImplicitBatchDimension();
	}

	void EngineImpl::print(){
		if(!context_){
			INFO("Engine print, nullptr.");
			return;
		}

		printf("Engine %p detail\n", this);
		printf("\tMax Batch Size: %d\n", this->get_max_batch_size());
		printf("\tDynamic Batch Dimension: %s\n", this->is_dynamic_batch_dimension() ? "true" : "false");
		printf("\tInputs: %d\n", inputs_.size());
		for(int i = 0; i < inputs_.size(); ++i){
			auto& tensor = inputs_[i];
			auto& name = inputs_name_[i];
			printf("\t\t%d.%s : shape {%s}\n", i, name.c_str(), tensor->shape_string());
		}

		printf("\tOutputs: %d\n", outputs_.size());
		for(int i = 0; i < outputs_.size(); ++i){
			auto& tensor = outputs_[i];
			auto& name = outputs_name_[i];
			printf("\t\t%d.%s : shape {%s}\n", i, name.c_str(), tensor->shape_string());
		}
	}

	bool EngineImpl::load(const std::string& file) {

		destroy();
		auto data = iLogger::load_file(file);
		if (data.empty())
			return false;

		this->context_.reset(new EngineContext());

		//build model
		EngineContext* context = (EngineContext*)this->context_.get();
		if (!context->build_model(data)) {
			this->context_.reset();
			return false;
		}

		workspace_.reset(new MemoryManager());
		cudaGetDevice(&device_);
		build_engine_input_and_outputs_mapper();
		return true;
	}

	size_t EngineImpl::get_device_memory_size() {
		EngineContext* context = (EngineContext*)this->context_.get();
		return context->context_->getEngine().getDeviceMemorySize();
	}

	void EngineImpl::build_engine_input_and_outputs_mapper() {
		
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
			const char* bindingName = context->engine_->getBindingName(i);
			auto mapperTensor = new Tensor(dims.nbDims, dims.d, TRTInfer::DataType::dtFloat);
			auto newTensor = shared_ptr<Tensor>(mapperTensor);
			newTensor->set_stream(this->context_->stream_);
			newTensor->set_workspace(this->workspace_);
			if (context->engine_->bindingIsInput(i)) {
				//if is input
				inputs_.push_back(newTensor);
				inputs_name_.push_back(bindingName);
			}
			else {
				//if is output
				outputs_.push_back(newTensor);
				outputs_name_.push_back(bindingName);
			}
			blobsNameMapper_[bindingName] = i;
			orderdBlobs_.push_back(newTensor);
		}
		bindingsPtr_.resize(orderdBlobs_.size());
	}

	void EngineImpl::set_stream(CUStream stream){
		this->context_->set_stream(stream);
	}

	CUStream EngineImpl::get_stream() {
		return this->context_->stream_;
	}

	int EngineImpl::device() {
		return device_;
	}

	void EngineImpl::synchronize() {
		checkCudaRuntime(cudaStreamSynchronize(context_->stream_));
	}

	bool EngineImpl::is_output_name(const std::string& name){
		return std::find(outputs_name_.begin(), outputs_name_.end(), name) != outputs_name_.end();
	}

	bool EngineImpl::is_input_name(const std::string& name){
		return std::find(inputs_name_.begin(), inputs_name_.end(), name) != inputs_name_.end();
	}

	void EngineImpl::forward(bool sync) {

		EngineContext* context = (EngineContext*)context_.get();
		int inputBatchSize = inputs_[0]->size(0);
		if(this->is_dynamic_batch_dimension())
			Assert(inputBatchSize <= context->engine_->getMaxBatchSize());
		else
			Assert(inputBatchSize == context->engine_->getMaxBatchSize());

		for (int i = 0; i < outputs_.size(); ++i) {
			outputs_[i]->resize_dim(0, inputBatchSize);
			outputs_[i]->to_gpu(false);
		}

		for (int i = 0; i < orderdBlobs_.size(); ++i)
			bindingsPtr_[i] = orderdBlobs_[i]->gpu();

		void** bindingsptr = bindingsPtr_.data();
		bool execute_result = context->context_->enqueue(inputBatchSize, bindingsptr, context->stream_, nullptr);
		//bool execute_result = context->context_->enqueueV2(bindingsptr, context->stream_, nullptr);
		if(!execute_result){
			auto code = cudaGetLastError();
			INFOF("execute fail, code %d[%s], message %s", code, cudaGetErrorName(code), cudaGetErrorString(code));
		}

		if (sync) {
			synchronize();
		}
	}

	std::shared_ptr<MemoryManager> EngineImpl::get_workspace() {
		return workspace_;
	}

	int EngineImpl::num_input() {
		return this->inputs_.size();
	}

	int EngineImpl::num_output() {
		return this->outputs_.size();
	}

	std::shared_ptr<Tensor> EngineImpl::input(int index) {
		return this->inputs_[index];
	}

	std::string EngineImpl::get_input_name(int index){
		Assert(index >= 0 && index < inputs_name_.size());
		return inputs_name_[index];
	}

	std::shared_ptr<Tensor> EngineImpl::output(int index) {
		Assert(index >= 0 && index < outputs_.size());
		return outputs_[index];
	}

	std::string EngineImpl::get_output_name(int index){
		Assert(index >= 0 && index < outputs_name_.size());
		return outputs_name_[index];
	}

	int EngineImpl::get_max_batch_size() {
		Assert(this->context_ != nullptr);
		return this->context_->engine_->getMaxBatchSize();
	}

	std::shared_ptr<Tensor> EngineImpl::tensor(const std::string& name) {
		Assert(this->blobsNameMapper_.find(name) != this->blobsNameMapper_.end());
		return orderdBlobs_[blobsNameMapper_[name]];
	}

	std::shared_ptr<Engine> load_engine(const string& file) {
		
		std::shared_ptr<Engine> engine(new EngineImpl());
		if (!engine->load(file))
			engine.reset();
		return engine;
	}
};