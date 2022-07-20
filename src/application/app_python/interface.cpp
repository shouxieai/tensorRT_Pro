
#ifdef HAS_PYTHON

#include <stdio.h>
#include <iostream>
#include <tools/pybind11.hpp>
#include <app_yolo_gpuptr/yolo_gpuptr.hpp>
#include <app_retinaface/retinaface.hpp>
#include <app_scrfd/scrfd.hpp>
#include <app_arcface/arcface.hpp>
#include <app_alphapose/alpha_pose.hpp>
#include <app_fall_gcn/fall_gcn.hpp>
#include <app_centernet/centernet.hpp>
#include <builder/trt_builder.hpp>
#include <common/preprocess_kernel.cuh>
#include <common/ilogger.hpp>
#include <common/trt_tensor.hpp>
#include <string>

using namespace std;
using namespace cv;
namespace py = pybind11;

class YoloInfer { 
public:
	YoloInfer(
		string engine, YoloGPUPtr::Type type, int device_id, float confidence_threshold, float nms_threshold,
		YoloGPUPtr::NMSMethod nms_method, int max_objects
	){
		instance_ = YoloGPUPtr::create_infer(
			engine, 
			type,
			device_id,
			confidence_threshold,
			nms_threshold,
			nms_method, max_objects
		);
	}

	bool valid(){
		return instance_ != nullptr;
	}

	shared_future<ObjectDetector::BoxArray> commit_gpu(int64_t pimage, int width, int height, int device_id, YoloGPUPtr::ImageType imtype, int64_t stream){

		if(!valid())
			throw py::buffer_error("Invalid engine instance, please makesure your construct");

		return instance_->commit(
			YoloGPUPtr::Image((uint8_t*)pimage, width, height, device_id, (cudaStream_t)stream, imtype)
		);
	}

	shared_future<ObjectDetector::BoxArray> commit(const py::array& image){

		if(!valid())
			throw py::buffer_error("Invalid engine instance, please makesure your construct");

		if(!image.owndata())
			throw py::buffer_error("Image muse be owner, slice is unsupport, use image.copy() inside, image[1:-1, 1:-1] etc.");

		cv::Mat cvimage(image.shape(0), image.shape(1), CV_8UC3, (unsigned char*)image.data(0));
		return instance_->commit(cvimage);
	}

	vector<shared_future<ObjectDetector::BoxArray>> commit_array(const vector<py::array>& image_array){

		if(!valid())
			throw py::buffer_error("Invalid engine instance, please makesure your construct");

		for(auto& image : image_array){
			if(!image.owndata())
				throw py::buffer_error("Image muse be owner, slice is unsupport, use image.copy() instead, image[1:-1, 1:-1] etc.");
		}

		vector<YoloGPUPtr::Image> images(image_array.size());
		for(int i = 0; i < images.size(); ++i){
			auto& pyimage = image_array[i];
			images[i] = cv::Mat(pyimage.shape(0), pyimage.shape(1), CV_8UC3, (unsigned char*)pyimage.data(0));
		}
		return instance_->commits(images);
	}

private:
	shared_ptr<YoloGPUPtr::Infer> instance_;
}; 

class CenterNetInfer { 
public:
	CenterNetInfer(string engine, int device_id, float confidence_threshold, float nms_threshold){

		instance_ = CenterNet::create_infer(
			engine, 
			device_id,
			confidence_threshold,
			nms_threshold
		);
	}

	bool valid(){
		return instance_ != nullptr;
	}

	shared_future<ObjectDetector::BoxArray> commit(const py::array& image){

		if(!valid())
			throw py::buffer_error("Invalid engine instance, please makesure your construct");

		if(!image.owndata())
			throw py::buffer_error("Image muse be owner, slice is unsupport, use image.copy() inside, image[1:-1, 1:-1] etc.");

		cv::Mat cvimage(image.shape(0), image.shape(1), CV_8UC3, (unsigned char*)image.data(0));
		return instance_->commit(cvimage);
	}

private:
	shared_ptr<CenterNet::Infer> instance_;
}; 

class RetinafaceInfer { 
public:
	RetinafaceInfer(string engine, int device_id, float confidence_threshold, float nms_threshold){

		instance_ = RetinaFace::create_infer(
			engine, 
			device_id,
			confidence_threshold,
			nms_threshold
		);
	}

	bool valid(){
		return instance_ != nullptr;
	}

	shared_future<FaceDetector::BoxArray> commit(const py::array& image){

		if(!valid())
			throw py::buffer_error("Invalid engine instance, please makesure your construct");

		if(!image.owndata())
			throw py::buffer_error("Image muse be owner, slice is unsupport, use image.copy() inside, image[1:-1, 1:-1] etc.");

		cv::Mat cvimage(image.shape(0), image.shape(1), CV_8UC3, (unsigned char*)image.data(0));
		return instance_->commit(cvimage);
	}

	py::tuple crop_face_and_landmark(const py::array& image, const FaceDetector::Box& box, float scale_box){

		if(!image.owndata())
			throw py::buffer_error("Image muse be owner, slice is unsupport, use image.copy() inside, image[1:-1, 1:-1] etc.");

		cv::Mat cvimage(image.shape(0), image.shape(1), CV_8UC3, (unsigned char*)image.data(0));
		auto output  = RetinaFace::crop_face_and_landmark(cvimage, box, scale_box);
		auto crop    = get<0>(output);
		auto py_crop = py::array(py::dtype("uint8"), vector<int>{crop.rows, crop.cols, 3}, crop.ptr<unsigned char>(0));
		return py::make_tuple(py_crop, get<1>(output));
	}

private:
	shared_ptr<RetinaFace::Infer> instance_;
}; 

class ScrfdInfer { 
public:
	ScrfdInfer(string engine, int device_id, float confidence_threshold, float nms_threshold){

		instance_ = Scrfd::create_infer(
			engine, 
			device_id,
			confidence_threshold,
			nms_threshold
		);
	}

	bool valid(){
		return instance_ != nullptr;
	}

	shared_future<FaceDetector::BoxArray> commit(const py::array& image){

		if(!valid())
			throw py::buffer_error("Invalid engine instance, please makesure your construct");

		if(!image.owndata())
			throw py::buffer_error("Image muse be owner, slice is unsupport, use image.copy() inside, image[1:-1, 1:-1] etc.");

		cv::Mat cvimage(image.shape(0), image.shape(1), CV_8UC3, (unsigned char*)image.data(0));
		return instance_->commit(cvimage);
	}

	py::tuple crop_face_and_landmark(const py::array& image, const FaceDetector::Box& box, float scale_box){

		if(!image.owndata())
			throw py::buffer_error("Image muse be owner, slice is unsupport, use image.copy() inside, image[1:-1, 1:-1] etc.");

		cv::Mat cvimage(image.shape(0), image.shape(1), CV_8UC3, (unsigned char*)image.data(0));
		auto output  = Scrfd::crop_face_and_landmark(cvimage, box, scale_box);
		auto crop    = get<0>(output);
		auto py_crop = py::array(py::dtype("uint8"), vector<int>{crop.rows, crop.cols, 3}, crop.ptr<unsigned char>(0));
		return py::make_tuple(py_crop, get<1>(output));
	}

private:
	shared_ptr<Scrfd::Infer> instance_;
}; 

class ArcfaceInfer { 
public:
	ArcfaceInfer(string engine, int device_id){

		instance_ = Arcface::create_infer(
			engine, 
			device_id
		);
	}

	bool valid(){
		return instance_ != nullptr;
	}

	shared_future<Arcface::feature> commit(const py::array& image, const py::array& landmark){

		if(!valid())
			throw py::buffer_error("Invalid engine instance, please makesure your construct");

		if(landmark.size() != 10)
			throw py::buffer_error("landmark must 10 elements, x, y, x, y, x, y");

		if(!image.owndata())
			throw py::buffer_error("Image muse be owner, slice is unsupport, use image.copy() inside, image[1:-1, 1:-1] etc.");

		cv::Mat cvimage(image.shape(0), image.shape(1), CV_8UC3, (unsigned char*)image.data(0));
		Arcface::landmarks lmk;
		memcpy(lmk.points, landmark.data(0), 10 * sizeof(float));
		return instance_->commit(make_tuple(cvimage, lmk));
	}

	py::array face_alignment(const py::array& image, const py::array& landmark){
		if(landmark.size() != 10)
			throw py::buffer_error("landmark must 10 elements, x, y, x, y, x, y");

		if(!image.owndata())
			throw py::buffer_error("Image muse be owner, slice is unsupport, use image.copy() inside, image[1:-1, 1:-1] etc.");

		Arcface::landmarks lmk;
		cv::Mat cvimage(image.shape(0), image.shape(1), CV_8UC3, (unsigned char*)image.data(0));
		memcpy(lmk.points, landmark.data(0), 10 * sizeof(float));
		auto output = Arcface::face_alignment(cvimage, lmk);
		return py::array(py::dtype("uint8"), vector<int>{output.rows, output.cols, 3}, output.ptr<unsigned char>(0));
	}

private:
	shared_ptr<Arcface::Infer> instance_;
}; 

class AlphaPoseInfer { 
public:
	AlphaPoseInfer(string engine, int device_id){

		instance_ = AlphaPose::create_infer(
			engine, 
			device_id
		);
	}

	bool valid(){
		return instance_ != nullptr;
	}

	shared_future<vector<Point3f>> commit(const py::array& image, const py::list& box){

		if(!valid())
			throw py::buffer_error("Invalid engine instance, please makesure your construct");

		if(box.size() != 4)
			throw py::value_error("Box must be 4 number, left, top, right, bottom");

		if(!image.owndata())
			throw py::buffer_error("Image muse be owner, slice is unsupport, use image.copy() inside, image[1:-1, 1:-1] etc.");

		cv::Mat cvimage(image.shape(0), image.shape(1), CV_8UC3, (unsigned char*)image.data(0));
		int left   = box[0].cast<float>();
		int top    = box[1].cast<float>();
		int right  = box[2].cast<float>();
		int bottom = box[3].cast<float>();
		return instance_->commit(make_tuple(cvimage, Rect(
			left, top, right-left, bottom-top
		)));
	}

private:
	shared_ptr<AlphaPose::Infer> instance_;
}; 

class FallInfer { 
public:
	FallInfer(string engine, int device_id){

		instance_ = FallGCN::create_infer(
			engine, 
			device_id
		);
	}

	bool valid(){
		return instance_ != nullptr;
	}

	shared_future<tuple<FallGCN::FallState, float>> commit(const py::array& keys, const py::list& box){

		if(!valid())
			throw py::buffer_error("Invalid engine instance, please makesure your construct");

		if(box.size() != 4)
			throw py::value_error("Box must be 4 number, left, top, right, bottom");

		if(keys.size() != 16*3 || keys.shape(0) != 16 || keys.shape(1) != 3 || keys.dtype() != py::dtype::of<float>())
			throw py::value_error("Keys must be 16x3 dtype=float32 ndarray");

		vector<Point3f> points;
		for(int i = 0; i < 16; ++i){
			float x = *(float*)keys.data(i, 0);
			float y = *(float*)keys.data(i, 1);
			float z = *(float*)keys.data(i, 2);
			points.emplace_back(x, y, z);
		}

		int left   = box[0].cast<float>();
		int top    = box[1].cast<float>();
		int right  = box[2].cast<float>();
		int bottom = box[3].cast<float>();
		return instance_->commit(make_tuple(points, Rect(left, top, right-left, bottom-top)));
	}

private:
	shared_ptr<FallGCN::Infer> instance_;
}; 

static TRT::Int8Process g_int8_process_func;
static bool compileTRT(
	unsigned int max_batch_size,
	const TRT::ModelSource& source,
	const TRT::CompileOutput& saveto,
	TRT::Mode mode, 
	const py::array inputs_dims, 
	int device_id,
	CUDAKernel::Norm int8_norm,
	int int8_preprocess_const_value,
	string int8_image_directory,
	string int8_entropy_calibrator_file,
	size_t max_workspace_size
){ 
	vector<TRT::InputDims> trt_inputs_dims;
	if(inputs_dims.size() != 0){
		if(inputs_dims.ndim() != 2 || inputs_dims.dtype() != py::dtype::of<int>()){
			INFOW("inputs_dims.ndim() = %d", inputs_dims.ndim());
			throw py::value_error("inputs_dims must be num x dims dtype=int ndarray");
		}

		int rows = inputs_dims.shape(0);
		int cols = inputs_dims.shape(1);
		trt_inputs_dims.resize(rows);
		for(int i = 0; i < rows; ++i){

			vector<int> dims;
			for(int j = 0; j < cols; ++j){
				int vdim = *(int*)inputs_dims.data(i, j);
				dims.emplace_back(vdim);
			}
			trt_inputs_dims[i] = dims;
		}
	}

	TRT::Int8Process int8process = [=](
		int current, int count, const std::vector<std::string>& files, 
		std::shared_ptr<TRT::Tensor>& tensor
	){
		auto workspace        = tensor->get_workspace();
		auto net_size         = Size(tensor->width(), tensor->height());
		int basic_size        = max(net_size.width, net_size.height);
		int basic_image_size  = basic_size * basic_size * 3;
		int basic_matrix_size = iLogger::upbound(6*sizeof(float), 32);
		int batch_bytes       = basic_matrix_size + basic_image_size;
		auto cpu_memory       = (char*)workspace->cpu(batch_bytes * files.size());
		auto gpu_memory       = (char*)workspace->gpu(batch_bytes * files.size());
		auto stream           = tensor->get_stream();

		for(int i = 0; i < files.size(); ++i){
			INFO("Process image %d / %d : %s", i+1, files.size(), files[i].c_str());

			auto image = imread(files[i]);
			if(image.empty()){
				INFOE("Load %s failed", files[i].c_str());
				continue;
			}

			float scale_to_basic = basic_size / (float)max(image.rows, image.cols);
			resize(image, image, Size(), scale_to_basic, scale_to_basic);
			
			auto from = image.size();
			float scale_x = net_size.width / (float)from.width;
            float scale_y = net_size.height / (float)from.height;

			float i2d[6], d2i[6];
            float scale = std::min(scale_x, scale_y);
            i2d[0] = scale;  i2d[1] = 0;  i2d[2] = -scale * from.width  * 0.5  + net_size.width * 0.5;
            i2d[3] = 0;  i2d[4] = scale;  i2d[5] = -scale * from.height * 0.5 + net_size.height * 0.5;

            // 有了i2d矩阵，我们求其逆矩阵，即可得到d2i（用以解码时还原到原始图像分辨率上）
            cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
            cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
            cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);

			char* matrix_host_ptr   = cpu_memory + i * batch_bytes;
			char* image_host_ptr    = cpu_memory + i * batch_bytes + basic_matrix_size;
			char* matrix_device_ptr = gpu_memory + i * batch_bytes;
			char* image_device_ptr  = gpu_memory + i * batch_bytes + basic_matrix_size;
			memcpy(matrix_host_ptr, d2i, sizeof(d2i));
			memcpy(image_host_ptr, image.data, image.rows * image.cols * 3);

			checkCudaRuntime(cudaMemcpyAsync(
				gpu_memory + i * batch_bytes, 
				cpu_memory + i * batch_bytes, 
				batch_bytes, cudaMemcpyHostToDevice, 
				stream
			));

			CUDAKernel::warp_affine_bilinear_and_normalize_plane(
				(uint8_t*)image_device_ptr, image.cols * 3, image.cols, image.rows, 
				(float*)tensor->gpu<float>(i), net_size.width, net_size.height, 
				(float*)matrix_device_ptr, int8_preprocess_const_value, int8_norm, stream
			);
		}
		tensor->synchronize();
	};

	if(g_int8_process_func){
		INFOV("Usage new process func");
		int8process = g_int8_process_func;
	}

	TRT::set_device(device_id);
	return TRT::compile(
		mode, max_batch_size, source, saveto, trt_inputs_dims, 
		int8process, int8_image_directory, 
		int8_entropy_calibrator_file, max_workspace_size
	);
}

static void set_compile_hook_reshape_layer(const py::function& func){
	auto hook_reshape_layer_function = [=](const string& name, const std::vector<int64_t>& shape){
		auto output = func(name, shape);
		return py::cast<vector<int64_t>>(output);
	};
	TRT::set_layer_hook_reshape(hook_reshape_layer_function);
}

static const char* norm_channel_type_string(CUDAKernel::ChannelType t){
	switch(t){
	case CUDAKernel::ChannelType::None:     return "NONE";
	case CUDAKernel::ChannelType::Invert:   return "Invert";
	default: return "Unknow";
	}
}

static const char* norm_type_string(CUDAKernel::NormType t){
	switch(t){
	case CUDAKernel::NormType::None:        return "NONE";
	case CUDAKernel::NormType::AlphaBeta:   return "AlphaBeta";
	case CUDAKernel::NormType::MeanStd:     return "MeanStd";
	default: return "Unknow";
	}
}

enum class ptr_base : int{
	host   = 0,
	device = 1
};

template <class T, ptr_base base=ptr_base::host> class ptr_wrapper{
    public:
        ptr_wrapper() : ptr(nullptr) {}
        ptr_wrapper(T* ptr) : ptr(ptr) {}
        ptr_wrapper(const ptr_wrapper& other) : ptr(other.ptr) {}
        T* get() const { return ptr; }
        void destroy() { ptr = nullptr; }
        T operator[](std::size_t idx) const {
			if(ptr == nullptr){
				INFOE("Invalid asccess to nullptr pointer with index=%d", idx);
				return T(0);
			}
			return ptr[idx]; 
		}
    private:
        T* ptr;
};

PYBIND11_MODULE(libpytrtc, m) {
	py::class_<ObjectDetector::Box>(m, "ObjectBox")
		.def_property("left",        [](ObjectDetector::Box& self){return self.left;}, [](ObjectDetector::Box& self, float nv){self.left = nv;})
		.def_property("top",         [](ObjectDetector::Box& self){return self.top;}, [](ObjectDetector::Box& self, float nv){self.top = nv;})
		.def_property("right",       [](ObjectDetector::Box& self){return self.right;}, [](ObjectDetector::Box& self, float nv){self.right = nv;})
		.def_property("bottom",      [](ObjectDetector::Box& self){return self.bottom;}, [](ObjectDetector::Box& self, float nv){self.bottom = nv;})
		.def_property("confidence",  [](ObjectDetector::Box& self){return self.confidence;}, [](ObjectDetector::Box& self, float nv){self.confidence = nv;})
		.def_property("class_label", [](ObjectDetector::Box& self){return self.class_label;}, [](ObjectDetector::Box& self, int nv){self.class_label = nv;})
		.def_property_readonly("width", [](ObjectDetector::Box& self){return self.right - self.left;})
		.def_property_readonly("height", [](ObjectDetector::Box& self){return self.bottom - self.top;})
		.def_property_readonly("cx", [](ObjectDetector::Box& self){return (self.left + self.right) / 2;})
		.def_property_readonly("cy", [](ObjectDetector::Box& self){return (self.top + self.bottom) / 2;})
		.def("__repr__", [](ObjectDetector::Box& obj){
			return iLogger::format(
				"<Box: left=%.2f, top=%.2f, right=%.2f, bottom=%.2f, class_label=%d, confidence=%.5f>",
				obj.left, obj.top, obj.right, obj.bottom, obj.class_label, obj.confidence
			);	
		});

	py::class_<FaceDetector::Box>(m, "FaceBox")
		.def_property("left",       &FaceDetector::Box::get_left,       &FaceDetector::Box::set_left)
		.def_property("top",        &FaceDetector::Box::get_top,        &FaceDetector::Box::set_top)
		.def_property("right",      &FaceDetector::Box::get_right,      &FaceDetector::Box::set_right)
		.def_property("bottom",     &FaceDetector::Box::get_bottom,     &FaceDetector::Box::set_bottom)
		.def_property("confidence", &FaceDetector::Box::get_confidence, &FaceDetector::Box::set_confidence)
		.def_property_readonly("width", [](FaceDetector::Box& self){return self.right - self.left;})
		.def_property_readonly("height", [](FaceDetector::Box& self){return self.bottom - self.top;})
		.def_property_readonly("cx", [](FaceDetector::Box& self){return (self.left + self.right) / 2;})
		.def_property_readonly("cy", [](FaceDetector::Box& self){return (self.top + self.bottom) / 2;})
		.def_property_readonly("landmark", [](FaceDetector::Box& self){
			return py::array(py::dtype("float32"), vector<int>{5, 2}, self.landmark);
		})
		.def("__repr__", [](FaceDetector::Box& self){
			return iLogger::format(
				"<Box: left=%.2f, top=%.2f, right=%.2f, bottom=%.2f, confidence=%.5f, landmark=ndarray(5x2)>",
				self.left, self.top, self.right, self.bottom, self.confidence
			);	
		});

	py::class_<shared_future<ObjectDetector::BoxArray>>(m, "SharedFutureObjectBoxArray")
		.def("get", &shared_future<ObjectDetector::BoxArray>::get);

	py::class_<shared_future<FaceDetector::BoxArray>>(m, "SharedFutureFaceBoxArray")
		.def("get", &shared_future<FaceDetector::BoxArray>::get);

	py::class_<shared_future<Arcface::feature>>(m, "SharedFutureArcfaceFeature")
		.def("get", [](shared_future<Arcface::feature>& self){
			auto feat = self.get();
			return py::array(py::dtype("float32"), vector<int>{1, feat.cols}, feat.ptr<float>(0));
		});

	py::class_<shared_future<vector<Point3f>>>(m, "SharedFutureAlphaPosePoints")
		.def("get", [](shared_future<vector<Point3f>>& self){
			auto points = self.get();
			return py::array(py::dtype("float32"), vector<int>{(int)points.size(), 3}, (float*)points.data());
		});

	py::enum_<FallGCN::FallState>(m, "FallState")
		.value("Fall",      FallGCN::FallState::Fall)
		.value("Stand",     FallGCN::FallState::Stand)
		.value("UnCertain", FallGCN::FallState::UnCertain);

	py::class_<shared_future<tuple<FallGCN::FallState, float>>>(m, "SharedFutureFallState")
		.def("get", [](shared_future<tuple<FallGCN::FallState, float>>& self){
			auto state = self.get();
			return py::make_tuple(get<0>(state), get<1>(state));
		});

	py::enum_<TRT::Mode>(m, "Mode")
		.value("FP32", TRT::Mode::FP32)
		.value("FP16", TRT::Mode::FP16)
		.value("INT8", TRT::Mode::INT8);

	py::enum_<YoloGPUPtr::ImageType>(m, "ImageType")
		.value("CVMat", YoloGPUPtr::ImageType::CVMat)
		.value("GPUYUVNV12", YoloGPUPtr::ImageType::GPUYUVNV12)
		.value("GPUBGR", YoloGPUPtr::ImageType::GPUBGR);

	py::enum_<CUDAKernel::NormType>(m, "NormType")
		.value("NONE", CUDAKernel::NormType::None)
		.value("MeanStd", CUDAKernel::NormType::MeanStd)
		.value("AlphaBeta", CUDAKernel::NormType::AlphaBeta);

	py::enum_<CUDAKernel::ChannelType>(m, "ChannelType")
		.value("NONE", CUDAKernel::ChannelType::None)
		.value("Invert", CUDAKernel::ChannelType::Invert);

	py::enum_<YoloGPUPtr::Type>(m, "YoloType")
		.value("V5", YoloGPUPtr::Type::V5)
		.value("V3", YoloGPUPtr::Type::V3)
		.value("V7", YoloGPUPtr::Type::V7)
		.value("X", YoloGPUPtr::Type::X);

	py::enum_<YoloGPUPtr::NMSMethod>(m, "NMSMethod")
		.value("CPU",     YoloGPUPtr::NMSMethod::CPU)
		.value("FastGPU", YoloGPUPtr::NMSMethod::FastGPU);

	py::class_<CUDAKernel::Norm>(m, "Norm")
		.def_property_readonly("mean", [](CUDAKernel::Norm& self){return vector<float>(self.mean, self.mean+3);})
		.def_property_readonly("std", [](CUDAKernel::Norm& self){return vector<float>(self.std, self.std+3);})
		.def_property_readonly("alpha", [](CUDAKernel::Norm& self){return self.alpha;})
		.def_property_readonly("beta", [](CUDAKernel::Norm& self){return self.beta;})
		.def_property_readonly("type", [](CUDAKernel::Norm& self){return self.type;})
		.def_property_readonly("channel_type", [](CUDAKernel::Norm& self){return self.channel_type;})
		.def_static("mean_std", [](const vector<float>& mean, const vector<float>& std, float alpha, CUDAKernel::ChannelType ct){
			if(mean.size() != 3 || std.size() != 3)
				throw py::value_error("mean or std must 3 element");
			return CUDAKernel::Norm::mean_std(mean.data(), std.data(), alpha, ct);
		}, py::arg("mean"), py::arg("std"), py::arg("alpha")=1.0f/255.0f, py::arg("channel_type")=CUDAKernel::ChannelType::None)
		.def_static("alpha_beta", CUDAKernel::Norm::alpha_beta, py::arg("alpha"), py::arg("beta"), py::arg("channel_type")=CUDAKernel::ChannelType::None)
		.def_static("none", CUDAKernel::Norm::None)
		.def("__repr__", [](CUDAKernel::Norm& self){
			string repr;
			if(self.type == CUDAKernel::NormType::MeanStd){
				repr = iLogger::format(
					"<Norm type=NormType.MeanStd mean=[%.5f, %.5f, %.5f], std=[%.5f, %.5f, %.5f], alpha=%.5f, channel_type=%s>",
					self.mean[0], self.mean[1], self.mean[2], self.std[0], self.std[1], self.std[2], self.alpha, norm_channel_type_string(self.channel_type)
				);
			}else if(self.type == CUDAKernel::NormType::AlphaBeta){
				repr = iLogger::format(
					"<Norm type=NormType.AlphaBeta alpha=%.5f, beta=%.5f, channel_type=%s>",
					self.alpha, self.beta, norm_channel_type_string(self.channel_type)
				);
			}else{
				repr = iLogger::format("<Norm type=%s>", norm_type_string(self.type));
			}
			return repr;
		});

	py::class_<YoloInfer>(m, "Yolo")
		.def(py::init<string, YoloGPUPtr::Type, int, float, float, YoloGPUPtr::NMSMethod, int>(), 
			py::arg("engine"), 
			py::arg("type")                 = YoloGPUPtr::Type::V5, 
			py::arg("device_id")            = 0, 
			py::arg("confidence_threshold") = 0.4f,
			py::arg("nms_threshold") = 0.5f,
			py::arg("nms_method")    = YoloGPUPtr::NMSMethod::FastGPU,
			py::arg("max_objects")   = 1024
		)
		.def_property_readonly("valid", &YoloInfer::valid, "Infer is valid")
		.def("commit", &YoloInfer::commit, py::arg("image"))
		.def("commit_array", &YoloInfer::commit_array, py::arg("image_array"))
		.def("commit_gpu", &YoloInfer::commit_gpu, 
			py::arg("pimage"), py::arg("width"), py::arg("height"), py::arg("device_id"), py::arg("imtype"), py::arg("stream")
		);

	py::class_<CenterNetInfer>(m, "CenterNet")
		.def(py::init<string, int, float, float>(), 
			py::arg("engine"), 
			py::arg("device_id")=0, 
			py::arg("confidence_threshold")=0.4f,
			py::arg("nms_threshold")=0.5f
		)
		.def_property_readonly("valid", &CenterNetInfer::valid, "Infer is valid")
		.def("commit", &CenterNetInfer::commit, py::arg("image"));

	py::class_<RetinafaceInfer>(m, "Retinaface")
		.def(py::init<string, int, float, float>(), 
			py::arg("engine"), 
			py::arg("device_id")=0, 
			py::arg("confidence_threshold")=0.7f,
			py::arg("nms_threshold")=0.5f
		)
		.def_property_readonly("valid", &RetinafaceInfer::valid, "Infer is valid")
		.def("commit", &RetinafaceInfer::commit, py::arg("image"))
		.def("crop_face_and_landmark", &RetinafaceInfer::crop_face_and_landmark, py::arg("image"), py::arg("Box"), py::arg("scale_box")=1.5f);

	py::class_<ScrfdInfer>(m, "Scrfd")
		.def(py::init<string, int, float, float>(), 
			py::arg("engine"), 
			py::arg("device_id")=0, 
			py::arg("confidence_threshold")=0.7f,
			py::arg("nms_threshold")=0.5f
		)
		.def_property_readonly("valid", &ScrfdInfer::valid, "Infer is valid")
		.def("commit", &ScrfdInfer::commit, py::arg("image"))
		.def("crop_face_and_landmark", &ScrfdInfer::crop_face_and_landmark, py::arg("image"), py::arg("Box"), py::arg("scale_box")=1.5f);

	py::class_<ArcfaceInfer>(m, "Arcface")
		.def(py::init<string, int>(), 
			py::arg("engine"), 
			py::arg("device_id")=0
		)
		.def_property_readonly("valid", &ArcfaceInfer::valid, "Infer is valid")
		.def("commit", &ArcfaceInfer::commit, py::arg("image"), py::arg("landmark"))
		.def("face_alignment", &ArcfaceInfer::face_alignment, py::arg("image"), py::arg("landmark"));

	py::class_<AlphaPoseInfer>(m, "AlphaPose")
		.def(py::init<string, int>(), 
			py::arg("engine"), 
			py::arg("device_id")=0
		)
		.def_property_readonly("valid", &AlphaPoseInfer::valid, "Infer is valid")
		.def("commit", &AlphaPoseInfer::commit, py::arg("image"), py::arg("box"));

	py::class_<FallInfer>(m, "Fall")
		.def(py::init<string, int>(), 
			py::arg("engine"), 
			py::arg("device_id")=0
		)
		.def_property_readonly("valid", &FallInfer::valid, "Infer is valid")
		.def("commit", &FallInfer::commit, py::arg("keys"), py::arg("box"));

	py::enum_<TRT::ModelSourceType>(m, "ModelSourceType")
		.value("OnnX", TRT::ModelSourceType::OnnX)
		.value("OnnXData", TRT::ModelSourceType::OnnXData);

	py::class_<TRT::ModelSource>(m, "ModelSource")
		.def_property_readonly("type", [](TRT::ModelSource& self){return self.type();})
		.def_property_readonly("onnxmodel", [](TRT::ModelSource& self){return self.onnxmodel();})
		.def_property_readonly("descript", [](TRT::ModelSource& self){return self.descript();})
		.def_property_readonly("onnx_data", [](TRT::ModelSource& self){return py::bytes((char*)self.onnx_data(), self.onnx_data_size());})
		.def_static("from_onnx", [](const string& file){return TRT::ModelSource::onnx(file);}, py::arg("file"))
		.def_static("from_onnx_data", [](const py::buffer& data){
			auto info = data.request();
			return TRT::ModelSource::onnx_data(info.ptr, info.itemsize * info.size);}, py::arg("data"))
		.def("__repr__", [](TRT::ModelSource& self){return iLogger::format("<ModelSource %s>", self.descript().c_str());});

	py::enum_<TRT::CompileOutputType>(m, "CompileOutputType")
		.value("File", TRT::CompileOutputType::File)
		.value("Memory", TRT::CompileOutputType::Memory);

	py::class_<TRT::CompileOutput>(m, "CompileOutput")
		.def_property_readonly("type", [](TRT::CompileOutput& self){return self.type();})
		.def_property_readonly("data", [](TRT::CompileOutput& self){return py::bytes((char*)self.data().data(), self.data().size());})
		.def_property_readonly("file", [](TRT::CompileOutput& self){return self.file();})
		.def_static("to_file", [](const string& file){return TRT::CompileOutput(file);}, py::arg("file"))
		.def_static("to_memory", [](){return TRT::CompileOutput();});

	m.def(
		"compileTRT", compileTRT,
		py::arg("max_batch_size"),
		py::arg("source"),
		py::arg("output"),
		py::arg("mode")                         = TRT::Mode::FP32,
		py::arg("inputs_dims")                  = py::array_t<int>(),
		py::arg("device_id")                    = 0,
		py::arg("int8_norm")                    = CUDAKernel::Norm::None(),
		py::arg("int8_preprocess_const_value")  = 114,
		py::arg("int8_image_directory")         = ".",
		py::arg("int8_entropy_calibrator_file") = "",
		py::arg("max_workspace_size")           = 1ul << 30
	);

	py::class_<ptr_wrapper<float, ptr_base::host  >>(m, "HostFloatPointer"  )
		.def_property_readonly("ptr", [](ptr_wrapper<float, ptr_base::host>& self){
			return (uint64_t)self.get();
		})
		.def("__getitem__", [](ptr_wrapper<float, ptr_base::host>& self, int index){return self[index];})
		.def("__repr__", [](ptr_wrapper<float, ptr_base::host>& self){
			return iLogger::format("<HostFloatPointer ptr=%p>", self.get());
		});

	py::class_<ptr_wrapper<float, ptr_base::device>>(m, "DeviceFloatPointer")
		.def_property_readonly("ptr", [](ptr_wrapper<float, ptr_base::device>& self){
			return (uint64_t)self.get();
		})
		// .def("__getitem__", [](ptr_wrapper<float, ptr_base::device>& self, int index){return self[index];})
		.def("__repr__", [](ptr_wrapper<float, ptr_base::device>& self){
			return iLogger::format("<DeviceFloatPointer ptr=%p>", self.get());
		});
	
	py::enum_<TRT::DataHead>(m, "DataHead")
		.value("Init",   TRT::DataHead::Init)
		.value("Device", TRT::DataHead::Device)
		.value("Host",   TRT::DataHead::Host);

	py::enum_<TRT::DataType>(m, "DataType")
		.value("Float",   TRT::DataType::Float)
		.value("Float16", TRT::DataType::Float16);

	py::class_<TRT::MixMemory, shared_ptr<TRT::MixMemory>>(m, "MixMemory")
		.def(py::init([](uint64_t cpu, size_t cpu_size, uint64_t gpu, size_t gpu_size){
			return make_shared<TRT::MixMemory>(
				(void*)cpu, cpu_size, (void*)gpu, gpu_size
			);
		}), py::arg("cpu")=0, py::arg("cpu_size")=0, py::arg("gpu")=0, py::arg("gpu_size")=0)
		.def_property_readonly("cpu", [](TRT::MixMemory& self){return ptr_wrapper<float, ptr_base::host  >((float*)self.cpu());})
		.def_property_readonly("gpu", [](TRT::MixMemory& self){return ptr_wrapper<float, ptr_base::device>((float*)self.gpu());})
		.def("aget_cpu", [](TRT::MixMemory& self, size_t size){return ptr_wrapper<float, ptr_base::host  >((float*)self.cpu(size));})
		.def("aget_gpu", [](TRT::MixMemory& self, size_t size){return ptr_wrapper<float, ptr_base::device>((float*)self.gpu(size));})
		.def("release_cpu", [](TRT::MixMemory& self){self.release_cpu();})
		.def("release_gpu", [](TRT::MixMemory& self){self.release_gpu();})
		.def("release_all", [](TRT::MixMemory& self){self.release_all();})
		.def_property_readonly("owner_cpu", [](TRT::MixMemory& self){return self.owner_cpu();})
		.def_property_readonly("owner_gpu", [](TRT::MixMemory& self){return self.owner_gpu();})
		.def_property_readonly("cpu_size", [](TRT::MixMemory& self){return self.cpu_size();})
		.def_property_readonly("gpu_size", [](TRT::MixMemory& self){return self.gpu_size();})
		.def("__repr__", [](TRT::MixMemory& self){
			return iLogger::format(
				"<MixMemory cpu=%p[owner=%s, %lld bytes], gpu=%p[owner=%s, %lld bytes]>", 
				self.cpu(), self.owner_cpu()?"True":"False", self.cpu_size(), self.gpu(), self.owner_gpu()?"True":"False", self.gpu_size()
			);
		});
 
	py::class_<TRT::Tensor, shared_ptr<TRT::Tensor>>(m, "Tensor")
		.def(py::init([](const vector<int>& dims, const shared_ptr<TRT::MixMemory>& data)              
		{
			return make_shared<TRT::Tensor>(dims, TRT::DataType::Float, data);
		}), py::arg("dims"), py::arg("data")=nullptr)
		.def_property_readonly("shape",  [](TRT::Tensor& self)                    {return self.dims();})
		.def_property_readonly("ndim",   [](TRT::Tensor& self)                    {return self.ndims();})
		.def_property("stream", [](TRT::Tensor& self){return (uint64_t)self.get_stream();}, [](TRT::Tensor& self, uint64_t new_stream){self.set_stream((TRT::CUStream)new_stream);})
		.def_property_readonly("workspace",   [](TRT::Tensor& self)               {return self.get_workspace();})
		.def_property_readonly("data",   [](TRT::Tensor& self)                    {return self.get_data();})
		.def("to_cpu", [](TRT::Tensor& self, float copy_if_need)                  {self.to_cpu(copy_if_need);}, py::arg("copy_if_need")=true)
		.def("to_gpu", [](TRT::Tensor& self, float copy_if_need)                  {self.to_gpu(copy_if_need);}, py::arg("copy_if_need")=true)
		.def_property("numpy",  [](TRT::Tensor& self){
			return py::array(py::memoryview::from_buffer(
				self.cpu<float>(),
				self.dims(),
				self.strides()
			));
		}, [](TRT::Tensor& self, const py::array& new_value){})
		.def_property_readonly("empty", [](TRT::Tensor& self)                     {return self.empty();})
		.def_property_readonly("numel", [](TRT::Tensor& self)                     {return self.numel();})
		.def("resize",            [](TRT::Tensor& self, const vector<int>& dims)  {self.resize(dims);})
		.def("resize_single_dim", [](TRT::Tensor& self, int dim, int size)        {self.resize_single_dim(dim, size);})
		.def("count",             [](TRT::Tensor& self, int start_axis)           {return self.count(start_axis);})
		.def("offset",            [](TRT::Tensor& self, const vector<int>& indexs){return self.offset_array(indexs);})
		.def("cpu_at",               [](TRT::Tensor& self, const vector<int>& indexs){return ptr_wrapper<float, ptr_base::host  >(self.cpu<float>() + self.offset_array(indexs));})
		.def("gpu_at",               [](TRT::Tensor& self, const vector<int>& indexs){return ptr_wrapper<float, ptr_base::device>(self.gpu<float>() + self.offset_array(indexs));})
		.def_property_readonly("cpu", [](TRT::Tensor& self){return ptr_wrapper<float, ptr_base::host  >(self.cpu<float>());})
		.def_property_readonly("gpu", [](TRT::Tensor& self){return ptr_wrapper<float, ptr_base::device>(self.gpu<float>());})
		.def_property_readonly("head",    [](TRT::Tensor& self){return self.head();})
		.def("reference_data", [](TRT::Tensor& self, const vector<int>& shape, uint64_t cpu, size_t cpu_size, uint64_t gpu, size_t gpu_size){
			self.reference_data(shape, (void*)cpu, cpu_size, (void*)gpu, gpu_size, TRT::DataType::Float);
		})
		.def_property_readonly("dtype", [](TRT::Tensor& self){return self.type();})
		.def("__repr__", [](TRT::Tensor& self){
			return iLogger::format(
				"<Tensor shape=%s, head=%s, dtype=%s, this=%p>", 
				self.shape_string(), TRT::data_head_string(self.head()), TRT::data_type_string(self.type()), &self
			);
		});

	py::class_<TRT::Infer, shared_ptr<TRT::Infer>>(m, "Infer")
		.def("forward", [](TRT::Infer& self, bool sync){return self.forward(sync);}, py::arg("sync")=true)
		.def("input", [](TRT::Infer& self, int index){return self.input(index);}, py::arg("index")=0)
		.def("output", [](TRT::Infer& self, int index){return self.output(index);}, py::arg("index")=0)
		.def_property("stream", [](TRT::Infer& self){return (uint64_t)self.get_stream();}, [](TRT::Infer& self, uint64_t new_stream){self.set_stream((TRT::CUStream)new_stream);})
		.def("synchronize", [](shared_ptr<TRT::Infer>& self){self->synchronize(); return self;})
		.def("is_input_name", [](TRT::Infer& self, const string& name){return self.is_input_name(name);})
		.def("is_output_name", [](TRT::Infer& self, const string& name){return self.is_output_name(name);})
		.def_property_readonly("num_input", [](TRT::Infer& self){return self.num_input();})
		.def_property_readonly("num_output", [](TRT::Infer& self){return self.num_output();})
		.def("get_input_name", [](TRT::Infer& self, int index){return self.get_input_name(index);}, py::arg("index")=0)
		.def("get_output_name", [](TRT::Infer& self, int index){return self.get_output_name(index);}, py::arg("index")=0)
		.def_property_readonly("max_batch_size", [](TRT::Infer& self){return self.get_max_batch_size();})
		.def("tensor", [](TRT::Infer& self, const string& name){return self.tensor(name);})
		.def_property_readonly("device", [](TRT::Infer& self){return self.device();})
		.def("print", [](shared_ptr<TRT::Infer>& self){self->print(); return self;})
		.def_property_readonly("workspace", [](TRT::Infer& self){return self.get_workspace();})
		.def("set_input", [](TRT::Infer& self, int index, shared_ptr<TRT::Tensor> tensor){self.set_input(index, tensor);}, py::arg("index"), py::arg("new_tensor"))
		.def("set_output", [](TRT::Infer& self, int index, shared_ptr<TRT::Tensor> tensor){self.set_output(index, tensor);}, py::arg("index"), py::arg("new_tensor"))
		.def("serial_engine", [](TRT::Infer& self){
			auto data = self.serial_engine();
			return py::bytes((char*)data->data(), data->size());
		});

	m.def("load_infer_file", [](const string& file){
		return TRT::load_infer(file);
	}, py::arg("file"));

	m.def("load_infer_data", [](const py::buffer& data){
		auto info = data.request();
		return TRT::load_infer_from_memory(info.ptr, info.itemsize * info.size);
	}, py::arg("data"));

	m.def("set_compile_hook_reshape_layer", set_compile_hook_reshape_layer);
	m.def("set_compile_int8_process", [](const py::function& func){
		g_int8_process_func = [=](int current, int count, const vector<string>& files, shared_ptr<TRT::Tensor>& tensor){
			func(current, count, files, tensor);
		};
	});

	py::enum_<iLogger::LogLevel>(m, "LogLevel")
		.value("Debug",   iLogger::LogLevel::Debug)
		.value("Verbose", iLogger::LogLevel::Verbose)
		.value("Info",    iLogger::LogLevel::Info)
		.value("Warning", iLogger::LogLevel::Warning)
		.value("Error",   iLogger::LogLevel::Error)
		.value("Fatal",   iLogger::LogLevel::Fatal);

	m.def("set_devie", [](int device_id){TRT::set_device(device_id);});
	m.def("get_devie", [](){return TRT::get_device();});
	m.def("set_log_level", [](iLogger::LogLevel level){iLogger::set_log_level(level);});
	m.def("get_log_level", [](){return iLogger::get_log_level();});
	m.def("random_color", [](int idd){return iLogger::random_color(idd);});
	m.def("init_nv_plugins", [](){TRT::init_nv_plugins();});
}
#endif // HAS_PYTHON