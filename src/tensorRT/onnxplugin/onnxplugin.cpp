
#include "onnxplugin.hpp"
#include <string>

using namespace nvinfer1;
using namespace std;

namespace ONNXPlugin {

	GTensor::GTensor(float* ptr, int ndims, int* dims) {
		this->ptr_ = ptr;
		this->shape_.insert(shape_.end(), dims, dims + ndims);
		this->dtType_ = TRT::DataType::dtFloat;
	}

	int GTensor::offset(const std::vector<int>& index){

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

	#ifdef HAS_CUDA_HALF
	GTensor::GTensor(TRT::halfloat* ptr, int ndims, int* dims) {
		this->ptr_ = ptr;
		this->shape_.insert(shape_.end(), dims, dims + ndims);
		this->dtType_ = TRT::DataType::dtHalfloat;
	}
	#endif

	GTensor::GTensor(const TRT::Tensor& tensor) {
		this->ptr_ = (float*)tensor.gpu();
		this->shape_ = tensor.dims();
		this->dtType_ = TRT::DataType::dtFloat;
	}

	int GTensor::count(int start_axis) const {
		if(start_axis >= 0 && start_axis < shape_.size()){
			int size = 1;
			for (int i = start_axis; i < shape_.size(); ++i) 
				size *= shape_[i];
			return size;
		}else{
			return 0;
		}
	}

	///////////////////////////////////
	LayerConfig::LayerConfig() {
		supportDataType_ = {nvinfer1::DataType::kFLOAT};
		supportPluginFormat_ = {nvinfer1::PluginFormat::kLINEAR};
		configDataType_ = TRT::DataType::dtFloat;
		configPluginFormat_ = nvinfer1::PluginFormat::kLINEAR;
	}

	void LayerConfig::serialCopyTo(void* buffer) {
		if (!serializeData_.empty())
			memcpy(buffer, &serializeData_[0], serializeData_.size());
	}

	int LayerConfig::serialize() {

		Plugin::BinIO out;
		out << input;
		out << output;
		out << workspaceSize_;
		out << configDataType_;
		out << configPluginFormat_;
		out << configMaxbatchSize_;
		out << info_;

		out << (int)weights_.size();
		for (int i = 0; i < weights_.size(); ++i) {

			if (configDataType_ == TRT::DataType::dtFloat) {
				weights_[i]->to_float();
			}
			
			#ifdef HAS_CUDA_HALF
			else if (configDataType_ == TRT::DataType::dtHalfloat) {
				weights_[i]->to_half();
			}
			#endif

			else{
				INFOE("unsupport datatype: %d", (int)configDataType_);
			}

			out << weights_[i]->dims();
			out << weights_[i]->type();
			out.write((char*)weights_[i]->cpu(), weights_[i]->bytes());
		}

		seril(out);
		serializeData_ = out.writedMemory();
		return serializeData_.size();
	}

	void LayerConfig::deserialize(const void* ptr, size_t length) {

		Plugin::BinIO in(ptr, length);
		in >> input;
		in >> output;
		in >> workspaceSize_;
		in >> configDataType_;
		in >> configPluginFormat_;
		in >> configMaxbatchSize_;
		in >> info_;

		int nbWeights = 0;
		in >> nbWeights;

		weights_.resize(nbWeights);
		for (int i = 0; i < nbWeights; ++i) {
			std::vector<int> dims;
			in >> dims;

			TRT::DataType dt;
			in >> dt;

			weights_[i].reset(new TRT::Tensor(dims, dt));
			in.read(weights_[i]->cpu(), weights_[i]->bytes());
			weights_[i]->gpu();
		}
		deseril(in);
	}

	void LayerConfig::setup(const std::string& info, const std::vector<std::shared_ptr<TRT::Tensor>>& weights) {

		this->info_ = info;
		this->weights_ = weights;
	}

	///////////////////////////////////////////////////////////////////////////////////

	TRTPlugin::~TRTPlugin() {
	}

	void TRTPlugin::pluginInit(const std::string& name, const std::string& info, const std::vector<std::shared_ptr<TRT::Tensor>>& weights) {
		phase_ = CompilePhase;
		layerName_ = name;
		config_ = this->config(name);
		Assert(config_ != nullptr);
		config_->output.resize(config_->nbOutput_);
		config_->setup(info, weights);
		config_->init();
		this->pluginConfigFinish();
	}

	void TRTPlugin::pluginInit(const std::string& name, const void* serialData, size_t serialLength) {
		phase_ = InferencePhase;
		layerName_ = name;
		config_ = this->config(name);
		Assert(config_ != nullptr);
		config_->deserialize(serialData, serialLength);
		config_->init();
		this->pluginConfigFinish();
	}

	std::shared_ptr<LayerConfig> TRTPlugin::config(const std::string& layerName) {
		return std::shared_ptr<LayerConfig>(new LayerConfig());
	}

	bool TRTPlugin::supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const noexcept{
		bool match = config_->supportDataType_.find(type) != config_->supportDataType_.end() &&
			config_->supportPluginFormat_.find(format) != config_->supportPluginFormat_.end();

		//INFO("supportsFormat %d, %d, match = %s", type, format, match ? "true" : "false");
		return match;
	}

	void TRTPlugin::configureWithFormat(
		const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims,
		int nbOutputs, nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize)noexcept {

		//INFO("configureWithFormat: type: %d, format: %d", type, format);
		if (type == nvinfer1::DataType::kFLOAT) {
			this->config_->configDataType_ = TRT::DataType::dtFloat;
		}

		#ifdef HAS_CUDA_HALF
		else if (type == nvinfer1::DataType::kHALF) {
			this->config_->configDataType_ = TRT::DataType::dtHalfloat;
		}
		#endif
		
		else {
			INFOE("unsuport datatype: %d", (int)type);
		}
		this->config_->configPluginFormat_ = format;
		this->config_->configMaxbatchSize_ = maxBatchSize;
	}

	int TRTPlugin::getNbOutputs() const noexcept{
		return config_->nbOutput_;
	}

	nvinfer1::Dims TRTPlugin::getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) noexcept{

		if (config_->input.empty()) {
			for (int i = 0; i < nbInputDims; ++i)
				config_->input.push_back(inputs[i]);
		}

		auto dims = outputDims(index, inputs, nbInputDims);
		config_->output[index] = dims;
		return dims;
	}

	int TRTPlugin::initialize() noexcept{
		return 0;
	}

	void TRTPlugin::terminate() noexcept{
	}

	size_t TRTPlugin::getWorkspaceSize(int maxBatchSize) const noexcept{
		return config_->workspaceSize_;
	}

	void TRTPlugin::mappingToGTensor() {
		if (inputTensors_.empty()) {
			inputTensors_.resize(config_->input.size());
			outputTensors_.resize(config_->output.size());
			weightTensors_.resize(config_->weights_.size());
			for (int i = 0; i < inputTensors_.size(); ++i) {
				auto& dims = config_->input[i];
				inputTensors_[i].shape_ = std::vector<int>(dims.d, dims.d + dims.nbDims);
			}

			for (int i = 0; i < outputTensors_.size(); ++i) {
				auto& dims = config_->output[i];
				outputTensors_[i].shape_ = std::vector<int>(dims.d, dims.d + dims.nbDims);
			}

			for (int i = 0; i < weightTensors_.size(); ++i) {
				auto& w = config_->weights_[i];
				weightTensors_[i].shape_ = w->dims();
				weightTensors_[i].ptr_ = w->gpu();
				weightTensors_[i].dtType_ = w->type();
			}
		}
	}

	int32_t TRTPlugin::enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
		mappingToGTensor();

		for (int i = 0; i < inputTensors_.size(); ++i) {
			inputTensors_[i].shape_[0] = batchSize;
			inputTensors_[i].ptr_ = (void*)inputs[i];
			inputTensors_[i].dtType_ = config_->configDataType_;
		}

		for (int i = 0; i < outputTensors_.size(); ++i) {
			outputTensors_[i].shape_[0] = batchSize;
			outputTensors_[i].ptr_ = outputs[i];
			inputTensors_[i].dtType_ = config_->configDataType_;
		}
		return enqueue(inputTensors_, outputTensors_, weightTensors_, workspace, stream);
	} 

	size_t TRTPlugin::getSerializationSize() const noexcept{
		return config_->serialize();
	}

	void TRTPlugin::serialize(void* buffer) const noexcept{
		config_->serialCopyTo(buffer);
	}
};// namespace Plugin