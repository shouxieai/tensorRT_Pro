#include "objectbox.hpp"
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include <common/infer_controller.hpp>
#include <common/preprocess_kernel.cuh>
#include <common/monopoly_allocator.hpp>
#include <common/cuda_tools.hpp>


namespace Objdetectbox {
	using namespace cv;
	using namespace std;

	struct AffineMatrix {
		float i2d[6];       // image to dst(network), 2x3 matrix
		float d2i[6];       // dst to image, 2x3 matrix

		void compute(const cv::Size& from, const cv::Size& to) {
			float scale_x = to.width / (float)from.width;
			float scale_y = to.height / (float)from.height;

			float scale = std::min(scale_x, scale_y);

			i2d[0] = scale;  i2d[1] = 0;  i2d[2] = -scale * from.width  * 0.5 + to.width * 0.5 + scale * 0.5 - 0.5;
			i2d[3] = 0;  i2d[4] = scale;  i2d[5] = -scale * from.height * 0.5 + to.height * 0.5 + scale * 0.5 - 0.5;

			cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
			cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
			cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
		}

		cv::Mat i2d_mat() {
			return cv::Mat(2, 3, CV_32F, i2d);
		}
	};


	using ControllerImpl = InferController
		<
		Mat,                    // input
		BoxArray,               // output
		tuple<string, int>,     // start param
		AffineMatrix            // additional
		>;
	class InferImpl : public Infer, public ControllerImpl {
	public:
		/** 要求在InferImpl里面执行stop，而不是在基类执行stop **/
		virtual ~InferImpl() {
			stop();
		}

		pair<int, float> getBestClass(float* data, int start_ind, int length) {
			int   max_ind = -1;
			float max_val = 0.0;
			for (int i = start_ind; i < length; i++) {
				if (data[i] > max_val) {
					max_val = data[i];
					max_ind = i;
				}
			}
			return pair<int, float>(max_ind - start_ind, max_val);
		}

		static tuple<float, float> affine_project(float x, float y, float* pmatrix) {

			float newx = x * pmatrix[0] + y * pmatrix[1] + pmatrix[2];
			float newy = x * pmatrix[3] + y * pmatrix[4] + pmatrix[5];
			return make_tuple(newx, newy);
		}

		static float iou(const Box& a, const Box& b) {
			float cleft = max(a.left, b.left);
			float ctop = max(a.top, b.top);
			float cright = min(a.right, b.right);
			float cbottom = min(a.bottom, b.bottom);

			float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
			if (c_area == 0.0f)
				return 0.0f;

			float a_area = max(0.0f, a.right - a.left) * max(0.0f, a.bottom - a.top);
			float b_area = max(0.0f, b.right - b.left) * max(0.0f, b.bottom - b.top);
			return c_area / (a_area + b_area - c_area);
		}

		static BoxArray cpu_nms(BoxArray& boxes, float threshold) {

			std::sort(boxes.begin(), boxes.end(), [](BoxArray::const_reference a, BoxArray::const_reference b) {
				return a.confidence > b.confidence;
			});

			BoxArray output;
			output.reserve(boxes.size());

			vector<bool> remove_flags(boxes.size());
			for (int i = 0; i < boxes.size(); ++i) {

				if (remove_flags[i]) continue;

				auto& a = boxes[i];
				output.emplace_back(a);

				for (int j = i + 1; j < boxes.size(); ++j) {
					if (remove_flags[j]) continue;

					auto& b = boxes[j];
					if (b.class_label == a.class_label) {
						if (iou(a, b) >= threshold)
							remove_flags[j] = true;
					}
				}
			}
			return output;
		}

		virtual bool startup(const string& file, int gpuid, float confidence_threshold, float nms_threshold) {
			normalize_ = CUDAKernel::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDAKernel::ChannelType::Invert);
			confidence_threshold_ = confidence_threshold;
			nms_threshold_ = nms_threshold;
			return ControllerImpl::startup(make_tuple(file, gpuid));
		}

		virtual void worker(promise<bool>& result) override {

			string file = get<0>(start_param_);
			int gpuid = get<1>(start_param_);

			TRT::set_device(gpuid);
			auto engine = TRT::load_infer(file);
			if (engine == nullptr) {
				INFOE("Engine %s load failed", file.c_str());
				result.set_value(false);
				return;
			}

			engine->print();

			const int MAX_IMAGE_BBOX = 1024;
			const int NUM_BOX_ELEMENT = 7;      // left, top, right, bottom, confidence, class, keepflag
			TRT::Tensor affin_matrix_device(TRT::DataType::Float);
			int max_batch_size = engine->get_max_batch_size();
			auto input  = engine->tensor("input.1");
			auto output = engine->tensor("1428");
			int num_classes = output->size(2) - 5;


			input_width_ = input->size(3);
			input_height_ = input->size(2);
			tensor_allocator_ = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
			stream_ = engine->get_stream();
			gpu_ = gpuid;
			result.set_value(true);

			input->resize_single_dim(0, max_batch_size).to_gpu();
			affin_matrix_device.set_stream(stream_);

			// the nubmer 8 here means 8 * sizeof(float) % 32 == 0
			affin_matrix_device.resize(max_batch_size, 8).to_gpu();


			vector<Job> fetch_jobs;
			while (get_jobs_and_wait(fetch_jobs, max_batch_size)) {

				int infer_batch_size = fetch_jobs.size();
				input->resize_single_dim(0, infer_batch_size);

				for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch) {
					auto& job = fetch_jobs[ibatch];
					auto& mono = job.mono_tensor->data();

					if (mono->get_stream() != stream_) {
						checkCudaRuntime(cudaStreamSynchronize(mono->get_stream()));
					}

					affin_matrix_device.copy_from_gpu(affin_matrix_device.offset(ibatch), mono->get_workspace()->gpu(), 6);
					input->copy_from_gpu(input->offset(ibatch), mono->gpu(), mono->count());
					job.mono_tensor->release();
				}
				engine->forward(false);

				for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch) {
					auto& job = fetch_jobs[ibatch];
					float* image_based_output = output->cpu<float>(ibatch);
					auto& image_based_boxes = job.output;
					auto& affine_matrix = job.additional;

					for (int i = 0; i < output->size(1); ++i) {
						float* boxinfo = output->cpu<float>(ibatch, i);
						if (boxinfo[4] <= confidence_threshold_)
							continue;

						for (int j = 5; j < output->size(2); j++)
							boxinfo[j] *= boxinfo[4];
					
						auto out_result = getBestClass(boxinfo, 5, num_classes);
						if (out_result.second < confidence_threshold_)
							continue;

						float box_x = boxinfo[0] - boxinfo[2] / 2.;
						float box_y = boxinfo[1] - boxinfo[3] / 2.;
						float box_r = boxinfo[0] + boxinfo[2] / 2.;
						float box_b = boxinfo[1] + boxinfo[3] / 2.;

						Point box_lt,box_rb;
						tie(box_lt.x, box_lt.y) = affine_project(box_x, box_y, job.additional.d2i);
						tie(box_rb.x, box_rb.y) = affine_project(box_r, box_b, job.additional.d2i);
						image_based_boxes.emplace_back(box_lt.x, box_lt.y, box_rb.x, box_rb.y, boxinfo[4], out_result.first);
					}
					image_based_boxes = cpu_nms(image_based_boxes, nms_threshold_);
					job.pro->set_value(job.output);
				}
				fetch_jobs.clear();
			}
			stream_ = nullptr;
			tensor_allocator_.reset();
			INFO("Engine destroy.");
		}

		virtual bool preprocess(Job& job, const Mat& image) override {

			if (tensor_allocator_ == nullptr) {
				INFOE("tensor_allocator_ is nullptr");
				return false;
			}

			job.mono_tensor = tensor_allocator_->query();
			if (job.mono_tensor == nullptr) {
				INFOE("Tensor allocator query failed.");
				return false;
			}

			CUDATools::AutoDevice auto_device(gpu_);
			auto& tensor = job.mono_tensor->data();
			if (tensor == nullptr) {
				// not init
				tensor = make_shared<TRT::Tensor>();
				tensor->set_workspace(make_shared<TRT::MixMemory>());
			}

			Size input_size(input_width_, input_height_);
			job.additional.compute(image.size(), input_size);

			tensor->set_stream(stream_);
			tensor->resize(1, 3, input_height_, input_width_);

			size_t size_image = image.cols * image.rows * 3;
			size_t size_matrix = iLogger::upbound(sizeof(job.additional.d2i), 32);
			auto workspace = tensor->get_workspace();
			uint8_t* gpu_workspace = (uint8_t*)workspace->gpu(size_matrix + size_image);
			float*   affine_matrix_device = (float*)gpu_workspace;
			uint8_t* image_device = size_matrix + gpu_workspace;

			uint8_t* cpu_workspace = (uint8_t*)workspace->cpu(size_matrix + size_image);
			float* affine_matrix_host = (float*)cpu_workspace;
			uint8_t* image_host = size_matrix + cpu_workspace;

			memcpy(image_host, image.data, size_image);
			memcpy(affine_matrix_host, job.additional.d2i, sizeof(job.additional.d2i));
			checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_));
			checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(job.additional.d2i), cudaMemcpyHostToDevice, stream_));

			CUDAKernel::warp_affine_bilinear_and_normalize_plane(
				image_device, image.cols * 3, image.cols, image.rows,
				tensor->gpu<float>(), input_width_, input_height_,
				affine_matrix_device, 114, // note
				normalize_, stream_
			);
			return true;
		}

		virtual vector<shared_future<BoxArray>> commits(const vector<Mat>& images) override {
			return ControllerImpl::commits(images);
		}

		virtual std::shared_future<BoxArray> commit(const Mat& image) override {
			return ControllerImpl::commit(image);
		}

	private:
		int input_width_ = 0;
		int input_height_ = 0;
		int gpu_ = 0;
		float confidence_threshold_ = 0;
		float nms_threshold_ = 0;
		TRT::CUStream stream_ = nullptr;
		CUDAKernel::Norm normalize_;
	};

	shared_ptr<Infer> create_infer(const string& engine_file, int gpuid, float confidence_threshold, float nms_threshold) {
		shared_ptr<InferImpl> instance(new InferImpl());
		if (!instance->startup(engine_file, gpuid, confidence_threshold, nms_threshold)) {
			instance.reset();
		}
		return instance;
	}

	void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, int ibatch) {

		auto normalize = CUDAKernel::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDAKernel::ChannelType::Invert);
		Size input_size(tensor->size(3), tensor->size(2));
		AffineMatrix affine;
		affine.compute(image.size(), input_size);

		size_t size_image = image.cols * image.rows * 3;
		size_t size_matrix = iLogger::upbound(sizeof(affine.d2i), 32);
		auto workspace = tensor->get_workspace();
		uint8_t* gpu_workspace = (uint8_t*)workspace->gpu(size_matrix + size_image);
		float*   affine_matrix_device = (float*)gpu_workspace;
		uint8_t* image_device = size_matrix + gpu_workspace;

		uint8_t* cpu_workspace = (uint8_t*)workspace->cpu(size_matrix + size_image);
		float* affine_matrix_host = (float*)cpu_workspace;
		uint8_t* image_host = size_matrix + cpu_workspace;
		auto stream = tensor->get_stream();

		memcpy(image_host, image.data, size_image);
		memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
		checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream));
		checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i), cudaMemcpyHostToDevice, stream));

		CUDAKernel::warp_affine_bilinear_and_normalize_plane(
			image_device, image.cols * 3, image.cols, image.rows,
			tensor->gpu<float>(ibatch), input_size.width, input_size.height,
			affine_matrix_device, 114,
			normalize, stream
		);
		tensor->synchronize();
	}
};


