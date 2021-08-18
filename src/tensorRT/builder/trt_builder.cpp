
#include "trt_builder.hpp"

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvCaffeParser.h>
#include <onnx_parser/NvOnnxParser.h>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <assert.h>
#include <stdarg.h>
#include <common/cuda_tools.hpp>

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace std;

class Logger : public ILogger {
public:
	virtual void log(Severity severity, const char* msg) noexcept override {

		if (severity == Severity::kINTERNAL_ERROR) {
			INFOE("NVInfer INTERNAL_ERROR: %s", msg);
			abort();
		}else if (severity == Severity::kERROR) {
			INFOE("NVInfer ERROR: %s", msg);
		}
		else  if (severity == Severity::kWARNING) {
			INFOW("NVInfer WARNING: %s", msg);
		}
		else {
			//INFOV("%s", msg);
		}
	}
};

static Logger gLogger;

namespace TRT {

	static string join_dims(const vector<int>& dims){
		stringstream output;
		char buf[64];
		const char* fmts[] = {"%d", " x %d"};
		for(int i = 0; i < dims.size(); ++i){
			snprintf(buf, sizeof(buf), fmts[i != 0], dims[i]);
			output << buf;
		}
		return output.str();
	}

	static string format(const char* fmt, ...) {
		va_list vl;
		va_start(vl, fmt);
		char buffer[10000];
		vsprintf(buffer, fmt, vl);
		return buffer;
	}

	string dims_str(const nvinfer1::Dims& dims){
		return join_dims(vector<int>(dims.d, dims.d + dims.nbDims));
	}

	const char* padding_mode_name(nvinfer1::PaddingMode mode){
		switch(mode){
			case nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN: return "explicit round down";
			case nvinfer1::PaddingMode::kEXPLICIT_ROUND_UP: return "explicit round up";
			case nvinfer1::PaddingMode::kSAME_UPPER: return "same supper";
			case nvinfer1::PaddingMode::kSAME_LOWER: return "same lower";
			case nvinfer1::PaddingMode::kCAFFE_ROUND_DOWN: return "caffe round down";
			case nvinfer1::PaddingMode::kCAFFE_ROUND_UP: return "caffe round up";
		}
		return "Unknow padding mode";
	}

	const char* pooling_type_name(nvinfer1::PoolingType type){
		switch(type){
			case nvinfer1::PoolingType::kMAX: return "MaxPooling";
			case nvinfer1::PoolingType::kAVERAGE: return "AveragePooling";
			case nvinfer1::PoolingType::kMAX_AVERAGE_BLEND: return "MaxAverageBlendPooling";
		}
		return "Unknow pooling type";
	}

	const char* activation_type_name(nvinfer1::ActivationType activation_type){
		switch(activation_type){
			case nvinfer1::ActivationType::kRELU: return "ReLU";
			case nvinfer1::ActivationType::kSIGMOID: return "Sigmoid";
			case nvinfer1::ActivationType::kTANH: return "TanH";
			case nvinfer1::ActivationType::kLEAKY_RELU: return "LeakyRelu";
			case nvinfer1::ActivationType::kELU: return "Elu";
			case nvinfer1::ActivationType::kSELU: return "Selu";
			case nvinfer1::ActivationType::kSOFTSIGN: return "Softsign";
			case nvinfer1::ActivationType::kSOFTPLUS: return "Parametric softplus";
			case nvinfer1::ActivationType::kCLIP: return "Clip";
			case nvinfer1::ActivationType::kHARD_SIGMOID: return "Hard sigmoid";
			case nvinfer1::ActivationType::kSCALED_TANH: return "Scaled tanh";
			case nvinfer1::ActivationType::kTHRESHOLDED_RELU: return "Thresholded ReLU";
		}
		return "Unknow activation type";
	}

	string layer_type_name(nvinfer1::ILayer* layer){
		switch(layer->getType()){
			case nvinfer1::LayerType::kCONVOLUTION: return "Convolution";
			case nvinfer1::LayerType::kFULLY_CONNECTED: return "Fully connected";
			case nvinfer1::LayerType::kACTIVATION: {
				nvinfer1::IActivationLayer* act = (nvinfer1::IActivationLayer*)layer;
				auto type = act->getActivationType();
				return activation_type_name(type);
			}
			case nvinfer1::LayerType::kPOOLING: {
				nvinfer1::IPoolingLayer* pool = (nvinfer1::IPoolingLayer*)layer;
				return pooling_type_name(pool->getPoolingType());
			}
			case nvinfer1::LayerType::kLRN: return "LRN";
			case nvinfer1::LayerType::kSCALE: return "Scale";
			case nvinfer1::LayerType::kSOFTMAX: return "SoftMax";
			case nvinfer1::LayerType::kDECONVOLUTION: return "Deconvolution";
			case nvinfer1::LayerType::kCONCATENATION: return "Concatenation";
			case nvinfer1::LayerType::kELEMENTWISE: return "Elementwise";
			case nvinfer1::LayerType::kPLUGIN: return "Plugin";
			case nvinfer1::LayerType::kUNARY: return "UnaryOp operation";
			case nvinfer1::LayerType::kPADDING: return "Padding";
			case nvinfer1::LayerType::kSHUFFLE: return "Shuffle";
			case nvinfer1::LayerType::kREDUCE: return "Reduce";
			case nvinfer1::LayerType::kTOPK: return "TopK";
			case nvinfer1::LayerType::kGATHER: return "Gather";
			case nvinfer1::LayerType::kMATRIX_MULTIPLY: return "Matrix multiply";
			case nvinfer1::LayerType::kRAGGED_SOFTMAX: return "Ragged softmax";
			case nvinfer1::LayerType::kCONSTANT: return "Constant";
			case nvinfer1::LayerType::kRNN_V2: return "RNNv2";
			case nvinfer1::LayerType::kIDENTITY: return "Identity";
			case nvinfer1::LayerType::kPLUGIN_V2: return "PluginV2";
			case nvinfer1::LayerType::kSLICE: return "Slice";
			case nvinfer1::LayerType::kSHAPE: return "Shape";
			case nvinfer1::LayerType::kPARAMETRIC_RELU: return "Parametric ReLU";
			case nvinfer1::LayerType::kRESIZE: return "Resize";
		}
		return "Unknow layer type";
	}

	string layer_descript(nvinfer1::ILayer* layer){
		switch(layer->getType()){
			case nvinfer1::LayerType::kCONVOLUTION: {
				nvinfer1::IConvolutionLayer* conv = (nvinfer1::IConvolutionLayer*)layer;
				return format("channel: %d, kernel: %s, padding: %s, stride: %s, dilation: %s, group: %d", 
					conv->getNbOutputMaps(),
					dims_str(conv->getKernelSizeNd()).c_str(),
					dims_str(conv->getPaddingNd()).c_str(),
					dims_str(conv->getStrideNd()).c_str(),
					dims_str(conv->getDilationNd()).c_str(),
					conv->getNbGroups()
				);
			}
			case nvinfer1::LayerType::kFULLY_CONNECTED:{
				nvinfer1::IFullyConnectedLayer* fully = (nvinfer1::IFullyConnectedLayer*)layer;
				return format("output channels: %d", fully->getNbOutputChannels());
			}
			case nvinfer1::LayerType::kPOOLING: {
				nvinfer1::IPoolingLayer* pool = (nvinfer1::IPoolingLayer*)layer;
				return format(
					"window: %s, padding: %s",
					dims_str(pool->getWindowSizeNd()).c_str(),
					dims_str(pool->getPaddingNd()).c_str()
				);
			}
			case nvinfer1::LayerType::kDECONVOLUTION:{
				nvinfer1::IDeconvolutionLayer* conv = (nvinfer1::IDeconvolutionLayer*)layer;
				return format("channel: %d, kernel: %s, padding: %s, stride: %s, group: %d", 
					conv->getNbOutputMaps(),
					dims_str(conv->getKernelSizeNd()).c_str(),
					dims_str(conv->getPaddingNd()).c_str(),
					dims_str(conv->getStrideNd()).c_str(),
					conv->getNbGroups()
				);
			}
			case nvinfer1::LayerType::kACTIVATION:
			case nvinfer1::LayerType::kPLUGIN:
			case nvinfer1::LayerType::kLRN:
			case nvinfer1::LayerType::kSCALE:
			case nvinfer1::LayerType::kSOFTMAX:
			case nvinfer1::LayerType::kCONCATENATION:
			case nvinfer1::LayerType::kELEMENTWISE:
			case nvinfer1::LayerType::kUNARY:
			case nvinfer1::LayerType::kPADDING:
			case nvinfer1::LayerType::kSHUFFLE:
			case nvinfer1::LayerType::kREDUCE:
			case nvinfer1::LayerType::kTOPK:
			case nvinfer1::LayerType::kGATHER:
			case nvinfer1::LayerType::kMATRIX_MULTIPLY:
			case nvinfer1::LayerType::kRAGGED_SOFTMAX:
			case nvinfer1::LayerType::kCONSTANT:
			case nvinfer1::LayerType::kRNN_V2:
			case nvinfer1::LayerType::kIDENTITY:
			case nvinfer1::LayerType::kPLUGIN_V2:
			case nvinfer1::LayerType::kSLICE:
			case nvinfer1::LayerType::kSHAPE:
			case nvinfer1::LayerType::kPARAMETRIC_RELU:
			case nvinfer1::LayerType::kRESIZE:
				return "";
		}
		return "Unknow layer type";
	}

	bool layer_has_input_tensor(nvinfer1::ILayer* layer){
		int num_input = layer->getNbInputs();
		for(int i = 0; i < num_input; ++i){
			if(layer->getInput(i)->isNetworkInput())
				return true;
		}
		return false;
	}

	bool layer_has_output_tensor(nvinfer1::ILayer* layer){
		int num_output = layer->getNbOutputs();
		for(int i = 0; i < num_output; ++i){
			if(layer->getOutput(i)->isNetworkOutput())
				return true;
		}
		return false;
	}

	template<typename _T>
	static void destroy_nvidia_pointer(_T* ptr) {
		if (ptr) ptr->destroy();
	}

	const char* mode_string(TRTMode type) {
		switch (type) {
		case TRTMode_FP32:
			return "FP32";
		case TRTMode_FP16:
			return "FP16";
		case TRTMode_INT8:
			return "INT8";
		default:
			return "UnknowTRTMode";
		}
	}

	static nvinfer1::Dims convert_to_trt_dims(const std::vector<int>& dims){

		nvinfer1::Dims output{0};
		if(dims.size() > nvinfer1::Dims::MAX_DIMS){
			INFOE("convert failed, dims.size[%d] > MAX_DIMS[%d]", dims.size(), nvinfer1::Dims::MAX_DIMS);
			return output;
		}

		if(!dims.empty()){
			output.nbDims = dims.size();
			memcpy(output.d, dims.data(), dims.size() * sizeof(int));
		}
		return output;
	}

	const std::vector<int>& InputDims::dims() const{
		return dims_;
	}

	InputDims::InputDims(const std::initializer_list<int>& dims)
		:dims_(dims){
	}

	InputDims::InputDims(const std::vector<int>& dims)
		:dims_(dims){
	}

	ModelSource::ModelSource(const std::string& prototxt, const std::string& caffemodel) {
		this->type_ = ModelSourceType_FromCaffe;
		this->prototxt_ = prototxt;
		this->caffemodel_ = caffemodel;
	}

	ModelSource::ModelSource(const char* onnxmodel){
		this->type_ = ModelSourceType_FromONNX;
		this->onnxmodel_ = onnxmodel;
	}

	ModelSource::ModelSource(const std::string& onnxmodel) {
		this->type_ = ModelSourceType_FromONNX;
		this->onnxmodel_ = onnxmodel;
	}

	std::string ModelSource::prototxt() const { return this->prototxt_; }
	std::string ModelSource::caffemodel() const { return this->caffemodel_; }
	std::string ModelSource::onnxmodel() const { return this->onnxmodel_; }
	ModelSourceType ModelSource::type() const { return this->type_; }
	std::string ModelSource::descript() const{
		if(this->type_ == ModelSourceType_FromONNX)
			return format("Onnx Model '%s'", onnxmodel_.c_str());
		else
			return format("Caffe Model \nPrototxt: '%s'\nCaffemodel: '%s'", prototxt_.c_str(), caffemodel_.c_str());
	}

	/////////////////////////////////////////////////////////////////////////////////////////
	class Int8EntropyCalibrator : public IInt8EntropyCalibrator2
	{
	public:
		Int8EntropyCalibrator(const vector<string>& imagefiles, nvinfer1::Dims dims, const Int8Process& preprocess) {

			Assert(preprocess != nullptr);
			this->dims_ = dims;
			this->allimgs_ = imagefiles;
			this->preprocess_ = preprocess;
			this->fromCalibratorData_ = false;
			images_.resize(dims.d[0]);
		}

		Int8EntropyCalibrator(const vector<uint8_t>& entropyCalibratorData, nvinfer1::Dims dims, const Int8Process& preprocess) {
			Assert(preprocess != nullptr);

			this->dims_ = dims;
			this->entropyCalibratorData_ = entropyCalibratorData;
			this->preprocess_ = preprocess;
			this->fromCalibratorData_ = true;
			images_.resize(dims.d[0]);
		}

		int getBatchSize() const noexcept {
			return dims_.d[0];
		}

		bool next() {
			int batch_size = dims_.d[0];
			if (cursor_ + batch_size > allimgs_.size())
				return false;

			for(int i = 0; i < batch_size; ++i)
				images_[i] = allimgs_[cursor_++];

			if (!tensor_) 
				tensor_.reset(new Tensor(dims_.nbDims, dims_.d));

			preprocess_(cursor_, allimgs_.size(), images_, tensor_);
			return true;
		}

		bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept {
			if (!next()) return false;
			bindings[0] = tensor_->gpu();
			return true;
		}

		const vector<uint8_t>& getEntropyCalibratorData() {
			return entropyCalibratorData_;
		}

		const void* readCalibrationCache(size_t& length) noexcept {
			if (fromCalibratorData_) {
				length = this->entropyCalibratorData_.size();
				return this->entropyCalibratorData_.data();
			}

			length = 0;
			return nullptr;
		}

		virtual void writeCalibrationCache(const void* cache, size_t length) noexcept {
			entropyCalibratorData_.assign((uint8_t*)cache, (uint8_t*)cache + length);
		}

	private:
		Int8Process preprocess_;
		vector<string> allimgs_;
		size_t batchCudaSize_ = 0;
		int cursor_ = 0;
		nvinfer1::Dims dims_;
		vector<string> images_;
		shared_ptr<Tensor> tensor_;
		vector<uint8_t> entropyCalibratorData_;
		bool fromCalibratorData_ = false;
	};

	bool compile(
		TRTMode mode,
		const std::vector<std::string>& outputs,
		unsigned int maxBatchSize,
		const ModelSource& source,
		const std::string& savepath,
		std::vector<InputDims> inputsDimsSetup, bool dynamicBatch,
		Int8Process int8process,
		const std::string& int8ImageDirectory,
		const std::string& int8EntropyCalibratorFile) {

		if (mode == TRTMode::TRTMode_INT8 && int8process == nullptr) {
			INFOE("int8process must not nullptr, when in int8 mode.");
			return false;
		}

		bool hasEntropyCalibrator = false;
		vector<uint8_t> entropyCalibratorData;
		vector<string> entropyCalibratorFiles;
		if (mode == TRTMode_INT8) {
			if (!int8EntropyCalibratorFile.empty()) {
				if (iLogger::exists(int8EntropyCalibratorFile)) {
					entropyCalibratorData = iLogger::load_file(int8EntropyCalibratorFile);
					if (entropyCalibratorData.empty()) {
						INFO("entropyCalibratorFile is set as: %s, but we read is empty.", int8EntropyCalibratorFile.c_str());
						return false;
					}
					hasEntropyCalibrator = true;
				}
			}
			
			if (hasEntropyCalibrator) {
				if (!int8ImageDirectory.empty()) {
					INFO("imageDirectory is ignore, when entropyCalibratorFile is set");
				}
			}
			else {
				if (int8process == nullptr) {
					INFO("int8process must be set. when Mode is '%s'", mode_string(mode));
					return false;
				}

				entropyCalibratorFiles = iLogger::find_files(int8ImageDirectory, "*.jpg;*.png;*.bmp;*.jpeg;*.tiff");
				if (entropyCalibratorFiles.empty()) {
					INFO("Can not find any images(jpg/png/bmp/jpeg/tiff) from directory: %s", int8ImageDirectory.c_str());
					return false;
				}
			}
		}
		else {
			if (hasEntropyCalibrator) {
				INFO("int8EntropyCalibratorFile is ignore, when Mode is '%s'", mode_string(mode));
			}
		}

		INFO("Compile %s %s.", mode_string(mode), source.descript().c_str());
		shared_ptr<IBuilder> builder(createInferBuilder(gLogger), destroy_nvidia_pointer<IBuilder>);
		if (builder == nullptr) {
			INFOE("Can not create builder.");
			return false;
		}

		shared_ptr<IBuilderConfig> config(builder->createBuilderConfig(), destroy_nvidia_pointer<IBuilderConfig>);
		if (mode == TRTMode_FP16) {
			if (!builder->platformHasFastFp16()) {
				INFOW("Platform not have fast fp16 support");
			}
			config->setFlag(BuilderFlag::kFP16);
		}
		else if (mode == TRTMode_INT8) {
			if (!builder->platformHasFastInt8()) {
				INFOW("Platform not have fast int8 support");
			}
			config->setFlag(BuilderFlag::kINT8);
		}

		shared_ptr<INetworkDefinition> network;
		shared_ptr<ICaffeParser> caffeParser;
		shared_ptr<nvonnxparser::IParser> onnxParser;
		if (source.type() == ModelSourceType_FromCaffe) {
			
			const auto explicitBatch = 0; //1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
			network = shared_ptr<INetworkDefinition>(builder->createNetworkV2(explicitBatch), destroy_nvidia_pointer<INetworkDefinition>);
			caffeParser.reset(createCaffeParser(), destroy_nvidia_pointer<ICaffeParser>);
			if (!caffeParser) {
				INFOW("Can not create caffe parser.");
				return false;
			}

			auto blobNameToTensor = caffeParser->parse(source.prototxt().c_str(), source.caffemodel().c_str(), *network, nvinfer1::DataType::kFLOAT);
			if (blobNameToTensor == nullptr) {
				INFO("parse network fail, prototxt: %s, caffemodel: %s", source.prototxt().c_str(), source.caffemodel().c_str());
				return false;
			}

			for (auto& output : outputs) {
				auto blobMarked = blobNameToTensor->find(output.c_str());
				if (blobMarked == nullptr) {
					INFO("Can not found marked output '%s' in network.", output.c_str());
					return false;
				}

				INFO("Marked output blob '%s'.", output.c_str());
				network->markOutput(*blobMarked);
			}

			if (network->getNbInputs() > 1) {
				INFO("Warning: network has %d input, maybe have errors", network->getNbInputs());
			}
		}
		else if(source.type() == ModelSourceType_FromONNX){
			
			int explicit_batch_size = maxBatchSize;
			const auto explicitBatch = dynamicBatch ? 0U : 1U;   //1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
			//network = shared_ptr<INetworkDefinition>(builder->createNetworkV2(explicitBatch), destroy_nvidia_pointer<INetworkDefinition>);
			network = shared_ptr<INetworkDefinition>(builder->createNetworkV2(explicitBatch), destroy_nvidia_pointer<INetworkDefinition>);

			vector<nvinfer1::Dims> dims_setup(inputsDimsSetup.size());
			for(int i = 0; i < inputsDimsSetup.size(); ++i){
				auto s = inputsDimsSetup[i];
				dims_setup[i] = convert_to_trt_dims(s.dims());

				if(dynamicBatch){
					if(dims_setup[i].d[0] != 1){
						INFOW("The dynamic batch size is set, the setup[%d] dimension batch[%d] is not 1, will change it to 1", i, dims_setup[i].d[0]);
						dims_setup[i].d[0] = 1;
					}
				}else{
					if(dims_setup[i].d[0] != explicit_batch_size){
						INFOW("The dynamic batch size is set, the setup[%d] dimension batch[%d] is not %d, will change it to %d", i, dims_setup[i].d[0], explicit_batch_size, explicit_batch_size);
						dims_setup[i].d[0] = explicit_batch_size;
					}
				}
			}

			//from onnx is not markOutput
			onnxParser.reset(nvonnxparser::createParser(*network, gLogger, dims_setup, explicit_batch_size), destroy_nvidia_pointer<nvonnxparser::IParser>);
			if (onnxParser == nullptr) {
				INFO("Can not create parser.");
				return false;
			}

			if (!onnxParser->parseFromFile(source.onnxmodel().c_str(), 1)) {
				INFO("Can not parse OnnX file: %s", source.onnxmodel().c_str());
				return false;
			}
		}
		else {
			INFO("not implementation source type: %d", source.type());
			Assert(false);
		}

		auto inputTensor = network->getInput(0);
		auto inputDims = inputTensor->getDimensions();

		shared_ptr<Int8EntropyCalibrator> int8Calibrator;
		if (mode == TRTMode_INT8) {
			if (hasEntropyCalibrator) {
				INFO("Using exist entropy calibrator data[%d bytes]: %s", entropyCalibratorData.size(), int8EntropyCalibratorFile.c_str());
				int8Calibrator.reset(new Int8EntropyCalibrator(
					entropyCalibratorData, inputDims, int8process
				));
			}
			else {
				INFO("Using image list[%d files]: %s", entropyCalibratorFiles.size(), int8ImageDirectory.c_str());
				int8Calibrator.reset(new Int8EntropyCalibrator(
					entropyCalibratorFiles, inputDims, int8process
				));
			}
			config->setInt8Calibrator(int8Calibrator.get());
		}

		size_t _1_GB = 1 << 30;
		INFO("Input shape is %s", join_dims(vector<int>(inputDims.d, inputDims.d + inputDims.nbDims)).c_str());
		INFO("Set max batch size = %d", maxBatchSize);
		INFO("Set max workspace size = %.2f MB", _1_GB / 1024.0f / 1024.0f);
		INFO("Dynamic batch dimension is %s", dynamicBatch ? "true" : "false");

		int net_num_input = network->getNbInputs();
		INFO("Network has %d inputs:", net_num_input);
		vector<string> input_names(net_num_input);
		for(int i = 0; i < net_num_input; ++i){
			auto tensor = network->getInput(i);
			auto dims = tensor->getDimensions();
			auto dims_str = join_dims(vector<int>(dims.d, dims.d+dims.nbDims));
			INFO("      %d.[%s] shape is %s", i, tensor->getName(), dims_str.c_str());

			input_names[i] = tensor->getName();
		}

		int net_num_output = network->getNbOutputs();
		INFO("Network has %d outputs:", net_num_output);
		for(int i = 0; i < net_num_output; ++i){
			auto tensor = network->getOutput(i);
			auto dims = tensor->getDimensions();
			auto dims_str = join_dims(vector<int>(dims.d, dims.d+dims.nbDims));
			INFO("      %d.[%s] shape is %s", i, tensor->getName(), dims_str.c_str());
		}

		int net_num_layers = network->getNbLayers();
		INFOV("Network has %d layers:", net_num_layers);
		for(int i = 0; i < net_num_layers; ++i){
			auto layer = network->getLayer(i);
			auto name = layer->getName();
			auto type_str = layer_type_name(layer);
			auto input0 = layer->getInput(0);
			if(input0 == nullptr) continue;
			
			auto output0 = layer->getOutput(0);
			auto input_dims = input0->getDimensions();
			auto output_dims = output0->getDimensions();
			bool has_input = layer_has_input_tensor(layer);
			bool has_output = layer_has_output_tensor(layer);
			auto descript = layer_descript(layer);
			type_str = iLogger::align_blank(type_str, 18);
			auto input_dims_str = iLogger::align_blank(dims_str(input_dims), 18);
			auto output_dims_str = iLogger::align_blank(dims_str(output_dims), 18);
			auto number_str = iLogger::align_blank(format("%d.", i), 4);

			const char* token = "      ";
			if(has_input)
				token = "  >>> ";
			else if(has_output)
				token = "  *** ";

			INFOV("%s%s%s %s-> %s%s", token, 
				number_str.c_str(), 
				type_str.c_str(),
				input_dims_str.c_str(),
				output_dims_str.c_str(),
				descript.c_str()
			);
		}
		
		builder->setMaxBatchSize(maxBatchSize);
		config->setMaxWorkspaceSize(_1_GB);
		// config->setFlag(BuilderFlag::kGPU_FALLBACK);
		// config->setDefaultDeviceType(DeviceType::kDLA);
		// config->setDLACore(0);

		INFO("Building engine...");
		auto time_start = iLogger::timestamp_now();
		shared_ptr<ICudaEngine> engine(builder->buildEngineWithConfig(*network, *config), destroy_nvidia_pointer<ICudaEngine>);
		if (engine == nullptr) {
			INFOE("engine is nullptr");
			return false;
		}

		if (mode == TRTMode_INT8) {
			if (!hasEntropyCalibrator) {
				if (!int8EntropyCalibratorFile.empty()) {
					INFO("Save calibrator to: %s", int8EntropyCalibratorFile.c_str());
					iLogger::save_file(int8EntropyCalibratorFile, int8Calibrator->getEntropyCalibratorData());
				}
				else {
					INFO("No set entropyCalibratorFile, and entropyCalibrator will not save.");
				}
			}
		}

		INFO("Build done %lld ms !", iLogger::timestamp_now() - time_start);
		
		// serialize the engine, then close everything down
		shared_ptr<IHostMemory> seridata(engine->serialize(), destroy_nvidia_pointer<IHostMemory>);
		return iLogger::save_file(savepath, seridata->data(), seridata->size());
	}
}; //namespace TRTBuilder