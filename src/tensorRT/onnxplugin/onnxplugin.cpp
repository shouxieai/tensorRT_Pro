
#include "onnxplugin.hpp"
#include <string>

using namespace nvinfer1;
using namespace std;

namespace ONNXPlugin {

	GTensor::GTensor(float* ptr, int ndims, int* dims) {
		this->ptr_ = ptr;
		this->shape_.insert(shape_.end(), dims, dims + ndims);
		this->dtype_ = TRT::DataType::Float;
	}

	int GTensor::offset_array(size_t size, const int* index_array) const{

		Assert(size <= shape_.size());
		int value = 0;
		for(int i = 0; i < shape_.size(); ++i){

			if(i < size)
				value += index_array[i];

			if(i + 1 < shape_.size())
				value *= shape_[i+1];
		}
		return value;
	}

	int GTensor::offset_array(const std::vector<int>& index) const{
		return offset_array(index.size(), index.data());
	}

	GTensor::GTensor(TRT::float16* ptr, int ndims, int* dims) {
		this->ptr_ = ptr;
		this->shape_.insert(shape_.end(), dims, dims + ndims);
		this->dtype_ = TRT::DataType::Float16;
	}

	GTensor::GTensor(const TRT::Tensor& tensor) {
		this->ptr_ = (float*)tensor.gpu();
		this->shape_ = tensor.dims();
		this->dtype_ = TRT::DataType::Float;
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
		support_dtype_set_ = {nvinfer1::DataType::kFLOAT};
		support_plugin_format_set_ = {nvinfer1::PluginFormat::kLINEAR};
		usage_dtype_ = TRT::DataType::Float;
		usage_plugin_format_ = nvinfer1::PluginFormat::kLINEAR;
	}

	void LayerConfig::serialize_data_copy_to(void* buffer) {
		if (!serialize_data_.empty())
			memcpy(buffer, &serialize_data_[0], serialize_data_.size());
	}

	int LayerConfig::serialize() {

		Plugin::BinIO out;
		out << workspace_size_;
		out << usage_dtype_;
		out << max_batch_size_;
		out << usage_plugin_format_;
		out << info_;

		out << (int)weights_.size();
		for (int i = 0; i < weights_.size(); ++i) {

			if (usage_dtype_ == TRT::DataType::Float) {
				weights_[i]->to_float();
			}
			else if (usage_dtype_ == TRT::DataType::Float16) {
				weights_[i]->to_half();
			}
			else{
				INFOE("unsupport datatype: %d", (int)usage_dtype_);
			}

			out << weights_[i]->dims();
			out << weights_[i]->type();
			out.write((char*)weights_[i]->cpu(), weights_[i]->bytes());
		}

		seril(out);
		serialize_data_ = out.writedMemory();
		return serialize_data_.size();
	}

	void LayerConfig::deserialize(const void* ptr, size_t length) {

		Plugin::BinIO in(ptr, length);
		in >> workspace_size_;
		in >> usage_dtype_;
		in >> max_batch_size_;
		in >> usage_plugin_format_;
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

	static TRT::DataType convert_trt_datatype(nvinfer1::DataType dt){
		switch(dt){
			case nvinfer1::DataType::kFLOAT: return TRT::DataType::Float;
			case nvinfer1::DataType::kHALF: return TRT::DataType::Float16;
			default:
				INFOE("Unsupport data type %d", dt);
				return TRT::DataType::Float;
		}
	}

	TRTPlugin::~TRTPlugin() {
	}

	void TRTPlugin::pluginInit(const std::string& name, const std::string& info, const std::vector<std::shared_ptr<TRT::Tensor>>& weights) {
		phase_ = CompilePhase;
		layerName_ = name;
		config_ = this->new_config();
		Assert(config_ != nullptr);
		config_->setup(info, weights);
		config_->init();
	}

	void TRTPlugin::pluginInit(const std::string& name, const void* serialData, size_t serialLength) {
		phase_ = InferencePhase;
		layerName_ = name;
		config_ = this->new_config();
		Assert(config_ != nullptr);
		config_->deserialize(serialData, serialLength);
		config_->init();
	}

	std::shared_ptr<LayerConfig> TRTPlugin::new_config() {
		return std::shared_ptr<LayerConfig>(new LayerConfig());
	}

	int TRTPlugin::getNbOutputs() const noexcept{
		return config_->num_output_;
	}

	void TRTPlugin::configurePlugin(
		const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs, 
		const nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept{

		auto type = in->desc.type;
		auto format = in->desc.format;
		this->config_->usage_dtype_     = convert_trt_datatype(type);
		this->config_->usage_plugin_format_ = format;
		this->config_->num_input_ = nbInputs;
		this->config_->max_batch_size_ = in->max.d[0];
		this->config_finish();
	}

	int TRTPlugin::initialize() noexcept{
		return 0;
	}

	void TRTPlugin::terminate() noexcept{
	}

	bool TRTPlugin::supportsFormatCombination(
		int32_t pos, const nvinfer1::PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept{
		
		bool match = config_->support_dtype_set_.find(inOut[pos].type) != config_->support_dtype_set_.end() &&
		config_->support_plugin_format_set_.find(inOut[pos].format) != config_->support_plugin_format_set_.end();
		return match;
	}

	size_t TRTPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs, const nvinfer1::PluginTensorDesc* outputs,
		int32_t nbOutputs) const noexcept{
		return config_->workspace_size_;
	}

	int32_t TRTPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept{

		if (inputTensors_.empty()) {
			inputTensors_.resize(config_->num_input_);
			outputTensors_.resize(config_->num_output_);
			weightTensors_.resize(config_->weights_.size());

			for (int i = 0; i < weightTensors_.size(); ++i) {
				auto& w = config_->weights_[i];
				weightTensors_[i].shape_ = w->dims();
				weightTensors_[i].ptr_ = w->gpu();
				weightTensors_[i].dtype_ = w->type();
			}
		}

		for (int i = 0; i < inputTensors_.size(); ++i) {
			inputTensors_[i].shape_ = std::vector<int>(inputDesc[i].dims.d, inputDesc[i].dims.d+inputDesc[i].dims.nbDims);
			inputTensors_[i].ptr_ = (void*)inputs[i];
			inputTensors_[i].dtype_ = convert_trt_datatype(inputDesc[i].type);
		}

		for (int i = 0; i < outputTensors_.size(); ++i) {
			outputTensors_[i].shape_ = std::vector<int>(outputDesc[i].dims.d, outputDesc[i].dims.d+outputDesc[i].dims.nbDims);
			outputTensors_[i].ptr_ = outputs[i];
			outputTensors_[i].dtype_ = convert_trt_datatype(outputDesc[i].type);
		}
		return enqueue(inputTensors_, outputTensors_, weightTensors_, workspace, stream);
	}

	size_t TRTPlugin::getSerializationSize() const noexcept{
		return config_->serialize();
	}

	void TRTPlugin::serialize(void* buffer) const noexcept{
		config_->serialize_data_copy_to(buffer);
	}
};// namespace Plugin