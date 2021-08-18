

#ifndef TRT_BUILDER_HPP
#define TRT_BUILDER_HPP

#include <string>
#include <vector>
#include <functional>
#include <infer/trt_infer.hpp>

namespace TRT {

	typedef std::function<void(int current, int count, std::vector<std::string>& images, std::shared_ptr<Tensor>& tensor)> Int8Process;

	enum ModelSourceType {
		ModelSourceType_FromCaffe,
		ModelSourceType_FromONNX
	};

	class ModelSource {
	public:
		ModelSource(const std::string& prototxt, const std::string& caffemodel);
		ModelSource(const std::string& onnxmodel);
		ModelSource(const char* onnxmodel);
		ModelSourceType type() const;
		std::string prototxt() const;
		std::string caffemodel() const;
		std::string onnxmodel() const;
		std::string descript() const;

	private:
		std::string prototxt_, caffemodel_;
		std::string onnxmodel_;
		ModelSourceType type_;
	};

	class InputDims {
	public:
		// 当为-1时，保留导入时的网络结构尺寸
		InputDims(const std::initializer_list<int>& dims);
		InputDims(const std::vector<int>& dims);

		const std::vector<int>& dims() const;

	private:
		std::vector<int> dims_;
	};

	enum TRTMode {
		TRTMode_FP32,
		TRTMode_FP16,
		TRTMode_INT8
	};

	const char* mode_string(TRTMode type);

	//当处于INT8模式时，int8process必须制定
	//     int8ImageDirectory和int8EntropyCalibratorFile指定一个即可
	//     如果初次生成，指定了int8EntropyCalibratorFile，calibrator会保存到int8EntropyCalibratorFile指定的文件
	//     如果已经生成过，指定了int8EntropyCalibratorFile，calibrator会从int8EntropyCalibratorFile指定的文件加载，而不是
	//          从int8ImageDirectory读取图片再重新生成
	//当处于FP32或者FP16时，int8process、int8ImageDirectory、int8EntropyCalibratorFile都不需要指定
	// 对于dynamicBatch参数，如果为true，则模型编译为动态batch，否则编译为静态batch
	//  动态batch size：1. 编译时，指定的max_batch_size，为允许推理给定的最大batch
	//                  2. 推理时，按照给定的input的size(0)为batch size数量进行推理。只要小于max_batch_size即可
	//                  3. 对于有些onnx的操作依赖batch维度调整时，动态batch会不能编译通过，例如（view操作等、shape节点等）
	//  静态batch size：1. 编译时，指定的max_batch_size，为推理时使用的batch size。即batch size固定不变
	//                  2. 推理时，所使用的batch size为max_batch_size指定的静态大小，所提供的input的size(0)也必须是max_batch_size
	//                     否则报错
	//                  3. 对于很多onnx，可以直接编译通过，不需要做任何修改，例如yolov5
	bool compile(
		TRTMode mode,
		const std::vector<std::string>& outputs,
		unsigned int maxBatchSize,
		const ModelSource& source,
		const std::string& savepath,
		const std::vector<InputDims> inputsDimsSetup = {}, bool dynamicBatch = true,
		Int8Process int8process = nullptr,
		const std::string& int8ImageDirectory = "",
		const std::string& int8EntropyCalibratorFile = "");
};

#endif //TRT_BUILDER_HPP