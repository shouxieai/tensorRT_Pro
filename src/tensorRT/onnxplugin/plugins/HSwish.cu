
#include <onnxplugin/onnxplugin.hpp>
#include <cuda_fp16.hpp>

using namespace ONNXPlugin;

static __global__ void hswish_kernel_fp32(float* input, float* output, int edge) {

    KernelPositionBlock;
    float x = input[position];
    float a = x + 3;
    a = a < 0 ? 0 : (a >= 6 ? 6 : a);
	output[position] = x * a / 6;
}

// static __global__ void hswish_kernel_fp16(__half* input, __half* output, int edge) {

// 	KernelPositionBlock;
	
//     __half _six = 6.0f;
// 	__half _three = 3.0f;
// 	__half x = input[position];
//     __half a = x + _three;
//     __half _zero = 0.0f;
//     a = a < _zero ? _zero : (a >= _six ? _six : a);
// 	output[position] = x * a / _six;
// }

class HSwish : public TRTPlugin {
public:
	SetupPlugin(HSwish);

	virtual void config_finish() override{
		 
		// INFO("init hswish config: %s", config_->info_.c_str());
		// INFO("weights = %d", config_->weights_.size());
		// for(int i = 0; i < config_->weights_.size(); ++i){
		// 	auto& w = config_->weights_[i];
		// 	if(w->type() == TRT::DataType::Float16){
		// 		INFO("Weight[%d] shape is %s, dtype = %s, value[0] = %f", i, w->shape_string(), data_type_string(w->type()), float(w->at<__half>(0)));
		// 	}else{
		// 		INFO("Weight[%d] shape is %s, dtype = %s, value[0] = %f", i, w->shape_string(), data_type_string(w->type()), w->at<float>(0));
		// 	}
		// }
	}

	virtual std::shared_ptr<LayerConfig> new_config() override{
		auto cfg = TRTPlugin::new_config();

		//cfg->support_dtype_set_ = {nvinfer1::DataType::kHALF, nvinfer1::DataType::kFLOAT};
		cfg->support_dtype_set_ = {nvinfer1::DataType::kFLOAT};
		return cfg;
	}

	virtual nvinfer1::DimsExprs getOutputDimensions(
        	int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override{

		return inputs[0];
	}

	int enqueue(const std::vector<GTensor>& inputs, std::vector<GTensor>& outputs, const std::vector<GTensor>& weights, void* workspace, cudaStream_t stream) override{
		
		int count = inputs[0].count();
		auto grid = CUDATools::grid_dims(count);
		auto block = CUDATools::block_dims(count);

		if (config_->usage_dtype_ == TRT::DataType::Float) {
			hswish_kernel_fp32 <<<grid, block, 0, stream >>> (inputs[0].ptr<float>(), outputs[0].ptr<float>(), count);
		}
		else if (config_->usage_dtype_ == TRT::DataType::Float16) {
			// hswish_kernel_fp16 <<<grid, block, 0, stream >>> (inputs[0].ptr<__half>(), outputs[0].ptr<__half>(), count);
			INFOF("not implement function");
		}
		else{
			INFOF("not implement function");
		}
		return 0;
	}
};

RegisterPlugin(HSwish);