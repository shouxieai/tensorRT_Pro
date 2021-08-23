
#include "trt_tensor.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include "cuda_tools.hpp"

#ifdef HAS_CUDA_HALF
#include <cuda_fp16.h>
#endif

using namespace cv;
using namespace std;

namespace TRT{

	int data_type_size(DataType dt){
		switch (dt) {
		case DataType::dtFloat: return sizeof(float);

		#ifdef HAS_CUDA_HALF
		case DataType::dtHalfloat: return sizeof(halfloat);
		#endif

		default: {
			INFOE("Not support dtype: %d", dt);
			return -1;
		}
		}
	}

	MixMemory::~MixMemory() {
		release_all();
	}

	void* MixMemory::gpu(size_t size) {

		if (gpu_size_ < size) {
			release_gpu();

			gpu_size_ = size;
			checkCudaRuntime(cudaMallocManaged(&gpu_, size));
			checkCudaRuntime(cudaMemset(gpu_, 0, size));
		}
		return gpu_;
	}

	void* MixMemory::cpu(size_t size) {

		if (cpu_size_ < size) {
			release_cpu();

			cpu_size_ = size;
			checkCudaRuntime(cudaMallocHost(&cpu_, size));
			Assert(cpu_ != nullptr);
			memset(cpu_, 0, size);
		}
		return cpu_;
	}

	void MixMemory::release_cpu() {
		if (cpu_) {
			checkCudaRuntime(cudaFreeHost(cpu_));
			cpu_ = nullptr;
		}
		cpu_size_ = 0;
	}

	void MixMemory::release_gpu() {
		if (gpu_) {
			checkCudaRuntime(cudaFree(gpu_));
			gpu_ = nullptr;
		}
		gpu_size_ = 0;
	}

	void MixMemory::release_all() {
		release_cpu();
		release_gpu();
	}

	Tensor& Tensor::compute_shape_string(){

		// clean string
		shape_string_[0] = 0;

		char* buffer = shape_string_;
		size_t buffer_size = sizeof(shape_string_);
		for(int i = 0; i < shape_.size(); ++i){

			int size = 0;
			if(i < shape_.size() - 1)
				size = snprintf(buffer, buffer_size, "%d x ", shape_[i]);
			else
				size = snprintf(buffer, buffer_size, "%d", shape_[i]);

			buffer += size;
			buffer_size -= size;
		}
		return *this;
	}

	Tensor::Tensor(int n, int c, int h, int w, DataType dtType) {
		this->dtype_ = dtType;
		resize(n, c, h, w);
	}

	Tensor::~Tensor() {
		release();
	}

	Tensor::Tensor(const std::vector<int>& dims, DataType dtType){
		this->dtype_ = dtType;
		resize(dims);
	}

	Tensor::Tensor(int ndims, const int* dims, DataType dtType) {

		this->dtype_ = dtType;
		resize(ndims, dims);
	}

	Tensor::Tensor(DataType dtType){
		shape_string_[0] = 0;
		dtype_ = dtType;
		data_  = make_shared<MixMemory>();
	}

	shared_ptr<Tensor> Tensor::clone(){
		auto new_tensor = make_shared<Tensor>(shape_, dtype_);
		if(head_ == DataHead_Init)
			return new_tensor;
		
		if(head_ == DataHead_InCPU){
			memcpy(new_tensor->cpu(), this->cpu(), this->bytes_);
		}else if(head_ == DataHead_InGPU){
			checkCudaRuntime(cudaMemcpyAsync(new_tensor->gpu(), this->gpu(), bytes_, cudaMemcpyDeviceToDevice, stream_));
		}
		return new_tensor;
	}

	Tensor& Tensor::copy_from_gpu(size_t offset, const void* src, size_t num_element){

		if(head_ == DataHead_Init)
			to_gpu(false);

		size_t offset_location = offset * element_size();
		if(offset_location >= bytes_){
			INFOE("Offset location[%lld] >= bytes_[%lld], out of range", offset_location, bytes_);
			return *this;
		}

		size_t copyed_bytes = num_element * element_size();
		size_t remain_bytes = bytes_ - offset_location;
		if(copyed_bytes > remain_bytes){
			INFOE("Copyed bytes[%lld] > remain bytes[%lld], out of range", copyed_bytes, remain_bytes);
			return *this;
		}

		if(head_ == DataHead_InGPU){
			checkCudaRuntime(cudaMemcpyAsync(gpu<unsigned char>() + offset_location, src, copyed_bytes, cudaMemcpyDeviceToDevice, stream_));
		}else if(head_ == DataHead_InCPU){
			checkCudaRuntime(cudaMemcpyAsync(cpu<unsigned char>() + offset_location, src, copyed_bytes, cudaMemcpyDeviceToHost, stream_));
		}else{
			INFOE("Unsupport head type %d", head_);
		}
		return *this;
	}

	Tensor& Tensor::copy_from_cpu(size_t offset, const void* src, size_t num_element){

		if(head_ == DataHead_Init)
			to_cpu(false);

		size_t offset_location = offset * element_size();
		if(offset_location >= bytes_){
			INFOE("Offset location[%lld] >= bytes_[%lld], out of range", offset_location, bytes_);
			return *this;
		}

		size_t copyed_bytes = num_element * element_size();
		size_t remain_bytes = bytes_ - offset_location;
		if(copyed_bytes > remain_bytes){
			INFOE("Copyed bytes[%lld] > remain bytes[%lld], out of range", copyed_bytes, remain_bytes);
			return *this;
		}

		if(head_ == DataHead_InGPU){
			checkCudaRuntime(cudaMemcpyAsync(data_->gpu() + offset_location, src, copyed_bytes, cudaMemcpyHostToDevice, stream_));
		}else if(head_ == DataHead_InCPU){
			checkCudaRuntime(cudaMemcpyAsync(data_->cpu() + offset_location, src, copyed_bytes, cudaMemcpyHostToHost, stream_));
		}else{
			INFOE("Unsupport head type %d", head_);
		}
		return *this;
	}

	Tensor& Tensor::release() {
		data_->release_all();
		shape_.clear();
		capacity_ = 0;
		bytes_ = 0;
		head_ = DataHead_Init;
		return *this;
	}

	bool Tensor::empty() {
		return data_->cpu() == nullptr && data_->gpu() == nullptr;
	}

	int Tensor::count(int start_axis) const {

		if(start_axis >= 0 && start_axis < shape_.size()){
			int size = 1;
			for (int i = start_axis; i < shape_.size(); ++i) 
				size *= shape_[i];
			return size;
		}else{
			return 0;
		}
	}

	Tensor& Tensor::resize(const std::vector<int>& dims) {
		return resize(dims.size(), dims.data());
	}

	int Tensor::numel(){
		int value = shape_.empty() ? 0 : 1;
		for(int i = 0; i < shape_.size(); ++i){
			value *= shape_[i];
		}
		return value;
	}

	Tensor& Tensor::resize_single_dim(int idim, int size){

		Assert(idim >= 0 && idim < shape_.size());

		auto new_shape = shape_;
		new_shape[idim] = size;
		return resize(new_shape);
	}

	Tensor& Tensor::resize(int ndims, const int* dims) {

		vector<int> setup_dims(ndims);
		for(int i = 0; i < ndims; ++i){
			int dim = dims[i];
			if(dim == -1){
				// 如果维度不同，则-1没有意义
				Assert(ndims == shape_.size());
				dim = shape_[i];
			}
			setup_dims[i] = dim;
		}
		this->shape_ = setup_dims;
		this->adajust_memory_by_update_dims_or_type();
		this->compute_shape_string();
		return *this;
	}

	Tensor& Tensor::adajust_memory_by_update_dims_or_type(){
		
		if(data_ == nullptr)
			data_ = make_shared<MixMemory>();

		int needed_size = this->numel() * element_size();
		if (needed_size > capacity_) {
			data_->release_all();
			bytes_ = 0;
			head_ = DataHead_Init;
			capacity_ = needed_size;
		}
		this->bytes_ = needed_size;
		return *this;
	}

	Tensor& Tensor::synchronize(){ 
		checkCudaRuntime(cudaStreamSynchronize(stream_));
		return *this;
	}

	Tensor& Tensor::to_gpu(bool copyedIfCPU) {

		if (head_ == DataHead_InGPU)
			return *this;

		head_ = DataHead_InGPU;
		data_->gpu(capacity_);

		if (copyedIfCPU && data_->cpu() != nullptr) {
			checkCudaRuntime(cudaMemcpyAsync(data_->gpu(), data_->cpu(), bytes_, cudaMemcpyHostToDevice, stream_));
		}
		return *this;
	}
	
	Tensor& Tensor::to_cpu(bool copyedIfGPU) {

		if (head_ == DataHead_InCPU)
			return *this;

		head_ = DataHead_InCPU;
		data_->cpu(capacity_);

		if (copyedIfGPU && data_->gpu() != nullptr) {
			checkCudaRuntime(cudaMemcpyAsync(data_->cpu(), data_->gpu(), bytes_, cudaMemcpyDeviceToHost, stream_));
			checkCudaRuntime(cudaStreamSynchronize(stream_));
		}
		return *this;
	}

	Tensor& Tensor::to_float() {

		if (type() == DataType::dtFloat)
			return *this;

		#ifdef HAS_CUDA_HALF

			if (type() != DataType::dtHalfloat) {
				INFOF("not implement function");
			}

			auto c = count();
			float* convert_memory = (float*)malloc(c * data_type_size(DataType::dtFloat));
			float* dst = convert_memory;
			halfloat* src = cpu<halfloat>();

			for (int i = 0; i < c; ++i)
				*dst++ = *src++;

			this->dtype_ = DataType::dtFloat;
			adajust_memory_by_update_dims_or_type();
			memcpy(cpu(), convert_memory, bytes_);
			free(convert_memory);

		#else
			INFOF("not implement function");
		#endif
		return *this;
	}

	#ifdef HAS_CUDA_HALF
	Tensor& Tensor::to_half() {

		if (type() == DataType::dtHalfloat)
			return *this;

		if (type() != DataType::dtFloat) {
			INFOF("not implement function");
		}

		auto c = count();
		halfloat* convert_memory = (halfloat*)malloc(c * data_type_size(DataType::dtHalfloat));
		halfloat* dst = convert_memory;
		float* src = cpu<float>();

		for (int i = 0; i < c; ++i) 
			*dst++ = *src++;

		this->dtype_ = DataType::dtHalfloat;
		adajust_memory_by_update_dims_or_type();
		memcpy(cpu(), convert_memory, bytes_);
		free(convert_memory);
		return *this;
	}
	#endif

	Tensor& Tensor::set_to(float value) {
		int c = count();
		if (dtype_ == DataType::dtFloat) {
			float* ptr = cpu<float>();
			for (int i = 0; i < c; ++i)
				*ptr++ = value;
		}
		else {
			#ifdef HAS_CUDA_HALF
				halfloat* ptr = cpu<halfloat>();
				for (int i = 0; i < c; ++i)
					*ptr++ = value;
			#else
				INFOF("not implement function");
			#endif
		}
		return *this;
	}

	int Tensor::offset(const std::vector<int>& index){
		
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

	#ifdef USE_OPENCV
	Tensor& Tensor::set_norm_mat(int n, const cv::Mat& image, float mean[3], float std[3]) {

		Assert(image.channels() == 3 && !image.empty() && type() == DataType::dtFloat);
		Assert(ndims() == 4 && n < shape_[0]);
		to_cpu(false);

		int width   = shape_[3];
		int height  = shape_[2];
		float scale = 1 / 255.0;
		cv::Mat inputframe = image;
		if(inputframe.size() != cv::Size(width, height))
			cv::resize(inputframe, inputframe, cv::Size(width, height));

		if(CV_MAT_DEPTH(inputframe.type()) != CV_32F){
			inputframe.convertTo(inputframe, CV_32F, scale);
		}

		cv::Mat ms[3];
		for (int c = 0; c < 3; ++c)
			ms[c] = cv::Mat(height, width, CV_32F, cpu<float>(n, c));

		split(inputframe, ms);
		Assert((void*)ms[0].data == (void*)cpu<float>(n));

		for (int c = 0; c < 3; ++c)
			ms[c] = (ms[c] - mean[c]) / std[c];
		return *this;
	}

	Tensor& Tensor::set_mat(int n, const cv::Mat& _image) {

		cv::Mat image = _image;
		Assert(!image.empty() && CV_MAT_DEPTH(image.type()) == CV_32F && type() == DataType::dtFloat);
		Assert(shape_.size() == 4 && n < shape_[0] && image.channels() == shape_[1]);
		to_cpu(false);

		int width  = shape_[3];
		int height = shape_[2];
		if (image.size() != cv::Size(width, height))
			cv::resize(image, image, cv::Size(width, height));

		if (image.channels() == 1) {
			memcpy(cpu<float>(n), image.data, width * height * sizeof(float));
			return *this;
		}

		vector<cv::Mat> ms(image.channels());
		for (int i = 0; i < ms.size(); ++i) 
			ms[i] = cv::Mat(height, width, CV_32F, cpu<float>(n, i));

		cv::split(image, &ms[0]);
		Assert((void*)ms[0].data == (void*)cpu<float>(n));
		return *this;
	}
	#endif // USE_OPENCV

	bool Tensor::save_to_file(const std::string& file){

		if(empty()) return false;

		FILE* f = fopen(file.c_str(), "wb");
		if(f == nullptr) return false;

		int ndims = this->ndims();
		unsigned int head[3] = {0xFCCFE2E2, ndims, static_cast<unsigned int>(dtype_)};
		fwrite(head, 1, sizeof(head), f);
		fwrite(shape_.data(), 1, sizeof(shape_[0]) * shape_.size(), f);
		fwrite(cpu(), 1, bytes_, f);
		fclose(f);
		return true;
	}

}; // TRTTensor