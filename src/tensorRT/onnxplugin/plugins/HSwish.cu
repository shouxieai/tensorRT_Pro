

#include "HSwish.hpp"

#ifdef HAS_CUDA_HALF
typedef TRTInfer::halfloat halfloat;
#endif

template<typename _T>
__global__ void HSwishKernel(_T* input, _T* output, int edge);


template<>
__global__ void HSwishKernel(float* input, float* output, int edge) {

    KernelPositionBlock;
    float x = input[position];
    float a = x + 3;
    a = a < 0 ? 0 : (a >= 6 ? 6 : a);
	output[position] = x * a / 6;
}

#ifdef HAS_CUDA_HALF
template<>
__global__ void HSwishKernel(halfloat* input, halfloat* output, int edge) {

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

	//定义我们这个插件支持half和float格式
	cfg->supportDataType_ = {nvinfer1::DataType::kHALF, nvinfer1::DataType::kFLOAT};
	return cfg;
}

int HSwish::enqueue(const std::vector<GTensor>& inputs, std::vector<GTensor>& outputs, const std::vector<GTensor>& weights, void* workspace, cudaStream_t stream) {

	int count = inputs[0].count();
	auto grid = cuda::grid_dims(count);
	auto block = cuda::block_dims(count);

	if (config_->configDataType_ == TRTInfer::DataType::dtFloat) {
		HSwishKernel <<<grid, block, 0, stream >>> (inputs[0].ptr<float>(), outputs[0].ptr<float>(), count);
	}

	#ifdef HAS_CUDA_HALF
		else if (config_->configDataType_ == TRTInfer::DataType::dtHalfloat) {
			HSwishKernel <<<grid, block, 0, stream >>> (inputs[0].ptr<halfloat>(), outputs[0].ptr<halfloat>(), count);
		}
	#else
		else{
			LOG(LFATAL) << "not implement function";
		}
	#endif

	return 0;
}

RegisterPlugin(HSwish);