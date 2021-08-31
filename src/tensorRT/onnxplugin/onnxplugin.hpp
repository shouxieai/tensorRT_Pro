
#ifndef ONNX_PLUGIN_HPP
#define ONNX_PLUGIN_HPP

#include <memory>
#include <vector>
#include <set>

#include <NvInfer.h>
#include <NvInferRuntimeCommon.h>
#include <cuda_fp16.h>

#include <common/cuda_tools.hpp>
#include <infer/trt_infer.hpp>
#include "plugin_binary_io.hpp"

namespace ONNXPlugin {

	enum Phase {
		CompilePhase,
		InferencePhase
	};

	struct GTensor {
		GTensor() {}
		GTensor(const TRT::Tensor& tensor);
		GTensor(float* ptr, int ndims, int* dims);

		#ifdef HAS_CUDA_HALF
		GTensor(TRT::halfloat* ptr, int ndims, int* dims);
		#endif

		int count(int start_axis = 0) const;

		template<typename ... _Args>
		int offset(int t, _Args&& ... args){
			offset_index_.clear();
			return offsetimpl(t, args...);
		}

		int offset(const std::vector<int>& index);

		template<typename _T>
		inline _T* ptr() const { return (_T*)ptr_; }

		template<typename _T, typename ... _Args>
		inline _T* ptr(int t, _Args&& ... args) const { return (_T*)ptr_ + offset(t, args...); }

		inline float* ptr_float() const { return (float*)ptr_; }

		template<typename ... _Args>
		inline float* ptr_float(int t, _Args&& ... args) const { return (float*)ptr_ + offset(t, args...); }

		#ifdef HAS_CUDA_HALF
		inline TRT::halfloat* ptr_half() const { return (TRT::halfloat*)ptr_; }

		template<typename ... _Args>
		inline TRT::halfloat* ptr_half(int t, _Args&& ... args) const { return (TRT::halfloat*)ptr_ + offset(t, args...); }
		#endif

		void* ptr_ = nullptr;
		TRT::DataType dtType_ = TRT::DataType::Float;
		std::vector<int> shape_;

	private:
		std::vector<int> offset_index_;

	private:
		int offsetimpl(int value){
			offset_index_.push_back(value);
			return offset(offset_index_);
		}

		template<typename ... _Args>
		int offsetimpl(int t, _Args&& ... args){
			offset_index_.push_back(t);
			return offsetimpl(args...);
		}
	};

	struct LayerConfig {

		///////////////////////////////////
		int nbOutput_ = 1;
		size_t workspaceSize_ = 0;
		std::set<nvinfer1::DataType> supportDataType_;
		std::set<nvinfer1::PluginFormat> supportPluginFormat_;

		std::vector<std::shared_ptr<TRT::Tensor>> weights_;
		TRT::DataType configDataType_;
		nvinfer1::PluginFormat configPluginFormat_;
		int configMaxbatchSize_ = 0;
		std::string info_;

		///////////////////////////////////
		std::vector<nvinfer1::Dims> input;
		std::vector<nvinfer1::Dims> output;
		std::string serializeData_;

		LayerConfig();
		void serialCopyTo(void* buffer);
		int serialize();
		void deserialize(const void* ptr, size_t length);
		void setup(const std::string& info, const std::vector<std::shared_ptr<TRT::Tensor>>& weights);
		virtual void seril(Plugin::BinIO& out) {}
		virtual void deseril(Plugin::BinIO& in) {}
		virtual void init(){}
	};

	#define SetupPlugin(class_)			\
		virtual const char* getPluginType() const noexcept override{return #class_;};																		\
		virtual const char* getPluginVersion() const noexcept override{return "1";};																			\
		virtual nvinfer1::IPluginV2Ext* clone() const noexcept override{return new class_(*this);}

	#define RegisterPlugin(class_)		\
	class class_##PluginCreator__ : public nvinfer1::IPluginCreator{																				\
	public:																																			\
		const char* getPluginName() const noexcept override{return #class_;}																					\
		const char* getPluginVersion() const noexcept override{return "1";}																					\
		const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override{return &mFieldCollection;}													\
																																					\
		nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override{									\
			auto plugin = new class_();																												\
			mFieldCollection = *fc;																													\
			mPluginName = name;																														\
			return plugin;																															\
		}																																			\
																																					\
		nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override{								\
			auto plugin = new class_();																												\
			plugin->pluginInit(name, serialData, serialLength);																						\
			mPluginName = name;																														\
			return plugin;																															\
		}																																			\
																																					\
		void setPluginNamespace(const char* libNamespace) noexcept override{mNamespace = libNamespace;}														\
		const char* getPluginNamespace() const noexcept override{return mNamespace.c_str();}																	\
																																					\
	private:																																		\
		std::string mNamespace;																														\
		std::string mPluginName;																													\
		nvinfer1::PluginFieldCollection mFieldCollection{0, nullptr};																				\
	};																																				\
	REGISTER_TENSORRT_PLUGIN(class_##PluginCreator__);

	class TRTPlugin : public nvinfer1::IPluginV2Ext {
	public:
		virtual nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override{return inputTypes[0];}
		virtual bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept override{return false;}
		virtual bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override{return false;}

		virtual void configurePlugin(const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims,
									int nbOutputs, const nvinfer1::DataType* inputTypes, const nvinfer1::DataType* outputTypes,
									const bool* inputIsBroadcast, const bool* outputIsBroadcast, nvinfer1::PluginFormat floatFormat, int maxBatchSize) noexcept override{
			this->configureWithFormat(inputDims, nbInputs, outputDims, nbOutputs, inputTypes[0], floatFormat, maxBatchSize);
		}

		virtual void attachToContext(cudnnContext* /*cudnn*/, cublasContext* /*cublas*/, nvinfer1::IGpuAllocator* /*allocator*/) noexcept override {}
		virtual void detachFromContext() noexcept override {}
		virtual void setPluginNamespace(const char* pluginNamespace) noexcept override{this->namespace_ = pluginNamespace;};
		virtual const char* getPluginNamespace() const noexcept override{return this->namespace_.data();};

		virtual ~TRTPlugin();
		virtual nvinfer1::Dims outputDims(int index, const nvinfer1::Dims* inputDims, int nbInputDims) = 0;
		virtual int enqueue(const std::vector<GTensor>& inputs, std::vector<GTensor>& outputs, const std::vector<GTensor>& weights, void* workspace, cudaStream_t stream) = 0;

		void pluginInit(const std::string& name, const std::string& info, const std::vector<std::shared_ptr<TRT::Tensor>>& weights);
		void pluginInit(const std::string& name, const void* serialData, size_t serialLength);
		virtual void pluginConfigFinish() {};

		virtual std::shared_ptr<LayerConfig> config(const std::string& layerName);
		virtual bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const noexcept;
		virtual void configureWithFormat(
			const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims,
			int nbOutputs, nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize) noexcept;
		virtual int getNbOutputs() const noexcept;
		virtual nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) noexcept;
		virtual int initialize() noexcept;
		virtual void terminate() noexcept;
		virtual void destroy() noexcept override{}
		virtual size_t getWorkspaceSize(int maxBatchSize) const noexcept override;
		virtual int32_t enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept;
		virtual size_t getSerializationSize() const noexcept override;
		virtual void serialize(void* buffer) const noexcept override;

	private:
		void mappingToGTensor();

	protected:
		std::string namespace_;
		std::string layerName_;
		Phase phase_ = CompilePhase;
		std::shared_ptr<LayerConfig> config_;
		std::vector<GTensor> inputTensors_;
		std::vector<GTensor> outputTensors_;
		std::vector<GTensor> weightTensors_;
	};

#define ExecuteKernel(numJobs, kernel, stream)		kernel<<<gridDims(numJobs), blockDims(numJobs), 0, stream>>>
}; //namespace Plugin

#endif //ONNX_PLUGIN_HPP