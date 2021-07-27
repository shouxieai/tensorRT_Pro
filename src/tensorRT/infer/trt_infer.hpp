

#ifndef TRT_INFER_HPP
#define TRT_INFER_HPP

#include <string>
#include <memory>
#include <vector>
#include <map>

// 如果不想依赖opencv，可以去掉这个定义
#define USE_OPENCV

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif // USE_OPENCV

#ifdef HAS_CUDA_HALF
struct __half;
#endif // HAS_CUDA_HALF

struct CUstream_st;

namespace TRTInfer {

	#ifdef HAS_CUDA_HALF
	typedef __half halfloat;
	#endif

	typedef CUstream_st* CUStream;

	enum DataHead{
		DataHead_Init = 0,
		DataHead_InGPU = 1,
		DataHead_InCPU = 2
	};

	#ifdef HAS_CUDA_HALF
	enum class DataType : int {
		dtFloat = 0,
		dtHalfloat = 1
	};
	#else
	enum class DataType : int {
		dtFloat = 0
	};
	#endif

	int data_type_size(DataType dt);


	class MemoryManager {
	public:
		virtual ~MemoryManager();
		void* gpu(size_t size);
		void* cpu(size_t size);
		void release_gpu();
		void release_cpu();
		void release_all();

		// 这里的GPU、CPU内存都可以用Host、Device直接访问
		// 这里的GPU内存，使用统一内存管理
		inline void* gpu() const { return gpu_; }

		// 这里的CPU内存，使用Pinned Memory，页锁定内存
		inline void* cpu() const { return cpu_; }

	private:
		void* cpu_ = nullptr;
		size_t cpu_size_ = 0;

		void* gpu_ = nullptr;
		size_t gpu_size_ = 0;
	};

	class Tensor {
	public:
		Tensor(const Tensor& other) = delete;
		Tensor& operator = (const Tensor& other) = delete;

		explicit Tensor();
		explicit Tensor(int n, int c, int h, int w, DataType dtType = DataType::dtFloat);
		explicit Tensor(int ndims, const int* dims, DataType dtType = DataType::dtFloat);
		explicit Tensor(const std::vector<int>& dims, DataType dtType = DataType::dtFloat);
		virtual ~Tensor();

		template<typename ... _Args>
		void resize(int t, _Args&& ... args){
			resized_dim_.clear();
			resize_impl(t, args...);
		}

		int numel();
		int ndims(){return shape_.size();}
		inline int size(int index) {return shape_[index];}
		inline int shape(int index) {return shape_[index];}
		inline DataType type() const { return dtType_; }
		inline const std::vector<int>& dims() const { return shape_; }
		inline int bytes() const { return bytes_; }
		inline int bytes(int start_axis) const { return count(start_axis) * element_size(); }
		inline int element_size() const { return data_type_size(dtType_); }

		void release();
		void set_to(float value);
		bool empty();
		void resize_dim(int idim, int size);
		void resize(int ndims, const int* dims);
		void resize(const std::vector<int>& dims);
		int count(int start_axis = 0) const;

		void to_gpu(bool copyedIfCPU = true);
		void to_cpu(bool copyedIfGPU = true);

		#ifdef HAS_CUDA_HALF
		void to_half();
		#endif

		void to_float();
		inline void* cpu() const { ((Tensor*)this)->to_cpu(); return memory_.cpu(); }
		inline void* gpu() const { ((Tensor*)this)->to_gpu(); return memory_.gpu(); }

		template<typename ... _Args>
		int offset(int t, _Args&& ... args){
			offset_index_.clear();
			return offset_impl(t, args...);
		}

		int offset(const std::vector<int>& index);
		
		template<typename DataT> inline const DataT* cpu() const { return (DataT*)cpu(); }
		template<typename DataT> inline const DataT* gpu() const { return (DataT*)gpu(); }
		template<typename DataT> inline DataT* cpu() { return (DataT*)cpu(); }
		template<typename DataT> inline DataT* gpu() { return (DataT*)gpu(); }

		template<typename DataT, typename ... _Args> 
		inline DataT* cpu(int t, _Args&& ... args) { return cpu<DataT>() + offset(t, args...); }

		template<typename DataT, typename ... _Args> 
		inline DataT* gpu(int t, _Args&& ... args) { return gpu<DataT>() + offset(t, args...); }

		template<typename DataT, typename ... _Args> 
		inline float& at(int t, _Args&& ... args) { return *(cpu<DataT>() + offset(t, args...)); }

		std::shared_ptr<MemoryManager> get_workspace(){return workspace_;}
		void set_workspace(std::shared_ptr<MemoryManager> workspace){workspace_ = workspace;}
		CUStream get_stream(){return stream_;}
		void set_stream(CUStream stream){stream_ = stream;}

		#ifdef USE_OPENCV
		void set_mat(int n, const cv::Mat& image);
		void set_norm_mat(int n, const cv::Mat& image, float mean[3], float std[3]);
		#endif // USE_OPENCV

		void synchronize();
		const char* shape_string() const{return shape_string_;}

	private:
		void resize_impl(int value){
			resized_dim_.push_back(value);
			resize(resized_dim_);
		}

		template<typename ... _Args>
		void resize_impl(int t, _Args&& ... args){
			resized_dim_.push_back(t);
			resize_impl(args...);
		}

		int offset_impl(int value){
			offset_index_.push_back(value);
			return offset(offset_index_);
		}

		template<typename ... _Args>
		int offset_impl(int t, _Args&& ... args){
			offset_index_.push_back(t);
			return offset_impl(args...);
		}

		void compute_shape_string();
		void adajust_memory_by_update_dims_or_type();

	private:
		std::vector<int> resized_dim_, offset_index_;
		std::vector<int> shape_;
		size_t capacity_ = 0;
		size_t bytes_ = 0;
		DataHead head_ = DataHead_Init;
		DataType dtType_ = DataType::dtFloat;
		char shape_string_[100];
		MemoryManager memory_;
		std::shared_ptr<MemoryManager> workspace_;
		CUStream stream_ = nullptr;
	};

	class Engine {
	public:
		virtual bool load(const std::string& file) = 0;
		virtual void destroy() = 0;
		virtual void forward(bool sync = true) = 0;
		virtual int get_max_batch_size() = 0;
		virtual CUStream get_stream() = 0;
		virtual void set_stream(CUStream stream) = 0;
		virtual void synchronize() = 0;
		virtual size_t get_device_memory_size() = 0;
		virtual std::shared_ptr<MemoryManager> get_workspace() = 0;
		virtual bool is_dynamic_batch_dimension() = 0;
		virtual std::shared_ptr<Tensor> input(int index = 0) = 0;
		virtual std::string get_input_name(int index = 0) = 0;
		virtual std::shared_ptr<Tensor> output(int index = 0) = 0;
		virtual std::string get_output_name(int index = 0) = 0;
		virtual bool is_output_name(const std::string& name) = 0;
		virtual bool is_input_name(const std::string& name) = 0;
		virtual std::shared_ptr<Tensor> tensor(const std::string& name) = 0;
		virtual int num_output() = 0;
		virtual int num_input() = 0;
		virtual void print() = 0;
		virtual int device() = 0;
	};

	struct MemoryInfo {
		size_t total;
		size_t available;
	};

	MemoryInfo get_current_device_memory_info();
	int get_device_count();
	int get_device();
	void set_device(int device_id);
	std::shared_ptr<Engine> load_engine(const std::string& file);
	bool init_nv_plugins();
};	//TRTInfer


#endif //TRT_INFER_HPP