

#include "HSwish.hpp"
#include <cuda_fp16.hpp>

static __global__ void hswish_kernel_fp32(float* input, float* output, int edge) {

    KernelPositionBlock;
    float x = input[position];
    float a = x + 3;
    a = a < 0 ? 0 : (a >= 6 ? 6 : a);
	output[position] = x * a / 6;
}

static __global__ void hswish_kernel_fp16(__half* input, __half* output, int edge) {

	KernelPositionBlock;
	
    __half _six = 6.0f;
	__half x = input[position];

    __half a = x + __half(3.0f);
    __half _zero = 0.0f;
    a = a < _zero ? _zero : (a >= _six ? _six : a);
	output[position] = x * a / _six;
}

void HSwishConfig::init(){
    INFO("init hswish config: %s", info_.c_str());
    INFO("weights = %d", this->weights_.size());
	for(int i = 0; i < this->weights_.size(); ++i){
		auto& w = this->weights_[i];
		if(w->type() == TRT::DataType::Float16){
			INFO("Weight[%d] shape is %s, dtype = %s, value[0] = %f", i, w->shape_string(), data_type_string(w->type()), float(w->at<__half>(0)));
		}else{
			INFO("Weight[%d] shape is %s, dtype = %s, value[0] = %f", i, w->shape_string(), data_type_string(w->type()), w->at<float>(0));
		}
	}
}

nvinfer1::DimsExprs HSwish::getOutputDimensions(
   int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept{
	return inputs[0];
}

std::shared_ptr<LayerConfig> HSwish::config(const std::string& layerName) {
	auto cfg = std::shared_ptr<LayerConfig>(new HSwishConfig());

	// 定义我们这个插件支持half和float格式
	// 如果支持多个，则tensorRT会实际执行不同格式，进行推理测试，以获取速度最快的那个类型
	//cfg->supportDataType_ = {nvinfer1::DataType::kHALF, nvinfer1::DataType::kFLOAT};
	//cfg->supportDataType_ = {nvinfer1::DataType::kHALF};
	cfg->supportDataType_ = {nvinfer1::DataType::kFLOAT};
	return cfg;
}

int HSwish::enqueue(const std::vector<GTensor>& inputs, std::vector<GTensor>& outputs, const std::vector<GTensor>& weights, void* workspace, cudaStream_t stream) {

	int count = inputs[0].count();
	auto grid = CUDATools::grid_dims(count);
	auto block = CUDATools::block_dims(count);

	if (config_->configDataType_ == TRT::DataType::Float) {
		INFO("enqueue for float");
		hswish_kernel_fp32 <<<grid, block, 0, stream >>> (inputs[0].ptr<float>(), outputs[0].ptr<float>(), count);
	}
	else if (config_->configDataType_ == TRT::DataType::Float16) {
		INFO("enqueue for half");
		hswish_kernel_fp16 <<<grid, block, 0, stream >>> (inputs[0].ptr<__half>(), outputs[0].ptr<__half>(), count);
	}
	else{
		INFOF("not implement function");
	}

	return 0;
}

RegisterPlugin(HSwish);