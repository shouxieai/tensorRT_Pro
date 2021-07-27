

#include "HSwish.hpp"

#ifdef HAS_CUDA_HALF
#include <cuda_fp16.hpp>
typedef TRTInfer::halfloat halfloat;
#endif

static __global__ void hswish_kernel_fp32(float* input, float* output, int edge) {

    KernelPositionBlock;
    float x = input[position];
    float a = x + 3;
    a = a < 0 ? 0 : (a >= 6 ? 6 : a);
	output[position] = x * a / 6;
}

#ifdef HAS_CUDA_HALF
static __global__ void hswish_kernel_fp16(halfloat* input, halfloat* output, int edge) {

	KernelPositionBlock;

    halfloat _six = 6.0f;
	halfloat x = input[position];
    halfloat a = x + halfloat(3.0f);
    halfloat _zero = 0.0f;
    a = a < _zero ? _zero : (a >= _six ? _six : a);
	output[position] = x * a / _six;
}
#endif

void HSwishConfig::init(){
    INFO("init hswish config: %s", info_.c_str());
    INFO("weights = %d", this->weights_.size());
	for(int i = 0; i < this->weights_.size(); ++i){
		auto& w = this->weights_[i];
		INFO("Weight[%d] shape is %s", i, w->shape_string());
	}
}

nvinfer1::Dims HSwish::outputDims(int index, const nvinfer1::Dims* inputDims, int nbInputDims) {
	return inputDims[0];
}

std::shared_ptr<LayerConfig> HSwish::config(const std::string& layerName) {
	auto cfg = std::shared_ptr<LayerConfig>(new HSwishConfig());

	#ifdef HAS_CUDA_HALF
		// 定义我们这个插件支持half和float格式
		// 如果支持多个，则tensorRT会实际执行不同格式，进行推理测试，以获取速度最快的那个类型
		// cfg->supportDataType_ = {nvinfer1::DataType::kHALF, nvinfer1::DataType::kFLOAT};
		cfg->supportDataType_ = {nvinfer1::DataType::kHALF};
	#else
		cfg->supportDataType_ = {nvinfer1::DataType::kFLOAT};
	#endif // HAS_CUDA_HALF
	return cfg;
}

int HSwish::enqueue(const std::vector<GTensor>& inputs, std::vector<GTensor>& outputs, const std::vector<GTensor>& weights, void* workspace, cudaStream_t stream) {

	int count = inputs[0].count();
	auto grid = cuda::grid_dims(count);
	auto block = cuda::block_dims(count);

	if (config_->configDataType_ == TRTInfer::DataType::dtFloat) {
		INFO("enqueue for float");
		hswish_kernel_fp32 <<<grid, block, 0, stream >>> (inputs[0].ptr<float>(), outputs[0].ptr<float>(), count);
	}

	#ifdef HAS_CUDA_HALF
		else if (config_->configDataType_ == TRTInfer::DataType::dtHalfloat) {
			INFO("enqueue for half");
			hswish_kernel_fp16 <<<grid, block, 0, stream >>> (inputs[0].ptr<halfloat>(), outputs[0].ptr<halfloat>(), count);
		}
	#else
		else{
			INFOF("not implement function");
		}
	#endif

	return 0;
}

RegisterPlugin(HSwish);