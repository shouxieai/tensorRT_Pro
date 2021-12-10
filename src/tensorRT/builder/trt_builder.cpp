
#include "trt_builder.hpp"

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>
//#include <NvCaffeParser.h>
#include <onnx_parser/NvOnnxParser.h>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <assert.h>
#include <stdarg.h>
#include <common/cuda_tools.hpp>

using namespace nvinfer1;   
using namespace std;   
//using namespace nvcaffeparser1  ;

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

	static string dims_str(const nvinfer1::Dims& dims){
		return join_dims(vector<int>(dims.d, dims.d + dims.nbDims));
	}

	static const char* padding_mode_name(nvinfer1::PaddingMode mode){
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

	static const char* pooling_type_name(nvinfer1::PoolingType type){
		switch(type){
			case nvinfer1::PoolingType::kMAX: return "MaxPooling";
			case nvinfer1::PoolingType::kAVERAGE: return "AveragePooling";
			case nvinfer1::PoolingType::kMAX_AVERAGE_BLEND: return "MaxAverageBlendPooling";
		}
		return "Unknow pooling type";
	}

	static const char* activation_type_name(nvinfer1::ActivationType activation_type){
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

	static string layer_type_name(nvinfer1::ILayer* layer){
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

	static string layer_descript(nvinfer1::ILayer* layer){
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

	static bool layer_has_input_tensor(nvinfer1::ILayer* layer){
		int num_input = layer->getNbInputs();
		for(int i = 0; i < num_input; ++i){
			auto input = layer->getInput(i);
			if(input == nullptr)
				continue;

			if(input->isNetworkInput())
				return true;
		}
		return false;
	}

	static bool layer_has_output_tensor(nvinfer1::ILayer* layer){
		int num_output = layer->getNbOutputs();
		for(int i = 0; i < num_output; ++i){

			auto output = layer->getOutput(i);
			if(output == nullptr)
				continue;

			if(output->isNetworkOutput())
				return true;
		}
		return false;
	}  

	template<typename _T>
	static void destroy_nvidia_pointer(_T* ptr) {
		if (ptr) ptr->destroy();
	}

	const char* mode_string(Mode type) {
		switch (type) {
		case Mode::FP32:
			return "FP32";
		case Mode::FP16:
			return "FP16";
		case Mode::INT8:
			return "INT8";
		default:
			return "UnknowTRTMode";
		}
	}

	void set_layer_hook_reshape(const LayerHookFuncReshape& func){
		register_layerhook_reshape(func);
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

	ModelSource::ModelSource(const char* onnxmodel){
		this->type_ = ModelSourceType::OnnX;
		this->onnxmodel_ = onnxmodel;
	}

	ModelSource::ModelSource(const std::string& onnxmodel) {
		this->type_ = ModelSourceType::OnnX;
		this->onnxmodel_ = onnxmodel;
	}

	const void* ModelSource::onnx_data() const{
		return this->onnx_data_;
	}

	size_t ModelSource::onnx_data_size() const{
		return this->onnx_data_size_;
	}

	std::string ModelSource::onnxmodel() const { return this->onnxmodel_; }
	ModelSourceType ModelSource::type() const { return this->type_; }
	std::string ModelSource::descript() const{
		if(this->type_ == ModelSourceType::OnnX)
			return format("Onnx Model '%s'", onnxmodel_.c_str());
		else if(this->type_ == ModelSourceType::OnnXData)
			return format("OnnXData Data: '%p', Size: '%lld'", onnx_data_, onnx_data_size_);
	}

	CompileOutput::CompileOutput(CompileOutputType type):type_(type){}
	CompileOutput::CompileOutput(const std::string& file):type_(CompileOutputType::File), file_(file){}
	CompileOutput::CompileOutput(const char* file):type_(CompileOutputType::File), file_(file){}
	void CompileOutput::set_data(const std::vector<uint8_t>& data){data_ = data;}

	void CompileOutput::set_data(std::vector<uint8_t>&& data){data_ = std::move(data);}
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
			files_.resize(dims.d[0]);
			checkCudaRuntime(cudaStreamCreate(&stream_));
		}

		Int8EntropyCalibrator(const vector<uint8_t>& entropyCalibratorData, nvinfer1::Dims dims, const Int8Process& preprocess) {
			Assert(preprocess != nullptr);

			this->dims_ = dims;
			this->entropyCalibratorData_ = entropyCalibratorData;
			this->preprocess_ = preprocess;
			this->fromCalibratorData_ = true;
			files_.resize(dims.d[0]);
			checkCudaRuntime(cudaStreamCreate(&stream_));
		}

		virtual ~Int8EntropyCalibrator(){
			checkCudaRuntime(cudaStreamDestroy(stream_));
		}

		int getBatchSize() const noexcept {
			return dims_.d[0];
		}

		bool next() {
			int batch_size = dims_.d[0];
			if (cursor_ + batch_size > allimgs_.size())
				return false;

			for(int i = 0; i < batch_size; ++i)
				files_[i] = allimgs_[cursor_++];

			if (!tensor_){
				tensor_.reset(new Tensor(dims_.nbDims, dims_.d));
				tensor_->set_stream(stream_);
				tensor_->set_workspace(make_shared<TRT::MixMemory>());
			}

			preprocess_(cursor_, allimgs_.size(), files_, tensor_);
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
		vector<string> files_;
		shared_ptr<Tensor> tensor_;
		vector<uint8_t> entropyCalibratorData_;
		bool fromCalibratorData_ = false;
		CUStream stream_ = nullptr;
	};

	bool compile(
		Mode mode,
		unsigned int maxBatchSize,
		const ModelSource& source,
		const CompileOutput& saveto,
		std::vector<InputDims> inputsDimsSetup,
		Int8Process int8process,
		const std::string& int8ImageDirectory,
		const std::string& int8EntropyCalibratorFile,
		const size_t maxWorkspaceSize) {

		if (mode == Mode::INT8 && int8process == nullptr) {
			INFOE("int8process must not nullptr, when in int8 mode.");
			return false;
		}

		bool hasEntropyCalibrator = false;
		vector<uint8_t> entropyCalibratorData;
		vector<string> entropyCalibratorFiles;
		if (mode == Mode::INT8) {
			if (!int8EntropyCalibratorFile.empty()) {
				if (iLogger::exists(int8EntropyCalibratorFile)) {
					entropyCalibratorData = iLogger::load_file(int8EntropyCalibratorFile);
					if (entropyCalibratorData.empty()) {
						INFOE("entropyCalibratorFile is set as: %s, but we read is empty.", int8EntropyCalibratorFile.c_str());
						return false;
					}
					hasEntropyCalibrator = true;
				}
			}
			
			if (hasEntropyCalibrator) {
				if (!int8ImageDirectory.empty()) {
					INFOW("imageDirectory is ignore, when entropyCalibratorFile is set");
				}
			}
			else {
				if (int8process == nullptr) {
					INFOE("int8process must be set. when Mode is '%s'", mode_string(mode));
					return false;
				}

				entropyCalibratorFiles = iLogger::find_files(int8ImageDirectory, "*.jpg;*.png;*.bmp;*.jpeg;*.tiff");
				if (entropyCalibratorFiles.empty()) {
					INFOE("Can not find any images(jpg/png/bmp/jpeg/tiff) from directory: %s", int8ImageDirectory.c_str());
					return false;
				}

				if(entropyCalibratorFiles.size() < maxBatchSize){
					INFOW("Too few images provided, %d[provided] < %d[max batch size], image copy will be performed", entropyCalibratorFiles.size(), maxBatchSize);
					
					int old_size = entropyCalibratorFiles.size();
                    for(int i = old_size; i < maxBatchSize; ++i)
                        entropyCalibratorFiles.push_back(entropyCalibratorFiles[i % old_size]);
				}
			}
		}
		else {
			if (hasEntropyCalibrator) {
				INFOW("int8EntropyCalibratorFile is ignore, when Mode is '%s'", mode_string(mode));
			}
		}

		INFO("Compile %s %s.", mode_string(mode), source.descript().c_str());
		shared_ptr<IBuilder> builder(createInferBuilder(gLogger), destroy_nvidia_pointer<IBuilder>);
		if (builder == nullptr) {
			INFOE("Can not create builder.");
			return false;
		}

		shared_ptr<IBuilderConfig> config(builder->createBuilderConfig(), destroy_nvidia_pointer<IBuilderConfig>);
		if (mode == Mode::FP16) {
			if (!builder->platformHasFastFp16()) {
				INFOW("Platform not have fast fp16 support");
			}
			config->setFlag(BuilderFlag::kFP16);
		}
		else if (mode == Mode::INT8) {
			if (!builder->platformHasFastInt8()) {
				INFOW("Platform not have fast int8 support");
			}
			config->setFlag(BuilderFlag::kINT8);
		}

		shared_ptr<INetworkDefinition> network;
		//shared_ptr<ICaffeParser> caffeParser;
		shared_ptr<nvonnxparser::IParser> onnxParser;
		if(source.type() == ModelSourceType::OnnX || source.type() == ModelSourceType::OnnXData){
			
			const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
			network = shared_ptr<INetworkDefinition>(builder->createNetworkV2(explicitBatch), destroy_nvidia_pointer<INetworkDefinition>);

			vector<nvinfer1::Dims> dims_setup(inputsDimsSetup.size());
			for(int i = 0; i < inputsDimsSetup.size(); ++i){
				auto s = inputsDimsSetup[i];
				dims_setup[i] = convert_to_trt_dims(s.dims());
				dims_setup[i].d[0] = -1;
			}

			//from onnx is not markOutput
			onnxParser.reset(nvonnxparser::createParser(*network, gLogger, dims_setup), destroy_nvidia_pointer<nvonnxparser::IParser>);
			if (onnxParser == nullptr) {
				INFOE("Can not create parser.");
				return false;
			}

			if(source.type() == ModelSourceType::OnnX){
				if (!onnxParser->parseFromFile(source.onnxmodel().c_str(), 1)) {
					INFOE("Can not parse OnnX file: %s", source.onnxmodel().c_str());
					return false;
				}
			}else{
				if (!onnxParser->parseFromData(source.onnx_data(), source.onnx_data_size(), 1)) {
					INFOE("Can not parse OnnX file: %s", source.onnxmodel().c_str());
					return false;
				}
			}
		}
		else {
			INFOE("not implementation source type: %d", source.type());
			Assert(false);
		}

		set_layer_hook_reshape(nullptr);
		auto inputTensor = network->getInput(0);
		auto inputDims = inputTensor->getDimensions();

		shared_ptr<Int8EntropyCalibrator> int8Calibrator;
		if (mode == Mode::INT8) {
			auto calibratorDims = inputDims;
			calibratorDims.d[0] = maxBatchSize;

			if (hasEntropyCalibrator) {
				INFO("Using exist entropy calibrator data[%d bytes]: %s", entropyCalibratorData.size(), int8EntropyCalibratorFile.c_str());
				int8Calibrator.reset(new Int8EntropyCalibrator(
					entropyCalibratorData, calibratorDims, int8process
				));
			}
			else {
				INFO("Using image list[%d files]: %s", entropyCalibratorFiles.size(), int8ImageDirectory.c_str());
				int8Calibrator.reset(new Int8EntropyCalibrator(
					entropyCalibratorFiles, calibratorDims, int8process
				));
			}
			config->setInt8Calibrator(int8Calibrator.get());
		}

		INFO("Input shape is %s", join_dims(vector<int>(inputDims.d, inputDims.d + inputDims.nbDims)).c_str());
		INFO("Set max batch size = %d", maxBatchSize);
		INFO("Set max workspace size = %.2f MB", maxWorkspaceSize / 1024.0f / 1024.0f);
		INFO("Base device: %s", CUDATools::device_description().c_str());

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
		INFO("Network has %d layers:", net_num_layers);
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
		config->setMaxWorkspaceSize(maxWorkspaceSize);

		auto profile = builder->createOptimizationProfile();
		for(int i = 0; i < net_num_input; ++i){
			auto input = network->getInput(i);
			auto input_dims = input->getDimensions();
			input_dims.d[0] = 1;
			profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
			profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
			input_dims.d[0] = maxBatchSize;
			profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
		}

		// not need
		// for(int i = 0; i < net_num_output; ++i){
		// 	auto output = network->getOutput(i);
		// 	auto output_dims = output->getDimensions();
		// 	output_dims.d[0] = 1;
		// 	profile->setDimensions(output->getName(), nvinfer1::OptProfileSelector::kMIN, output_dims);
		// 	profile->setDimensions(output->getName(), nvinfer1::OptProfileSelector::kOPT, output_dims);
		// 	output_dims.d[0] = maxBatchSize;
		// 	profile->setDimensions(output->getName(), nvinfer1::OptProfileSelector::kMAX, output_dims);
		// }
		config->addOptimizationProfile(profile);

		// error on jetson
		// auto timing_cache = shared_ptr<nvinfer1::ITimingCache>(config->createTimingCache(nullptr, 0), [](nvinfer1::ITimingCache* ptr){ptr->reset();});
		// config->setTimingCache(*timing_cache, false);
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

		if (mode == Mode::INT8) {
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
		if(saveto.type() == CompileOutputType::File){
			return iLogger::save_file(saveto.file(), seridata->data(), seridata->size());
		}else{
			((CompileOutput&)saveto).set_data(vector<uint8_t>((uint8_t*)seridata->data(), (uint8_t*)seridata->data()+seridata->size()));
			return true;
		}
	}
}; //namespace TRTBuilder
