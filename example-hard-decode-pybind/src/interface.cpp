
// #include <opencv2/opencv.hpp>
#include <algorithm>

#include <unordered_map>
#include <memory>
#include "pybind11.hpp"
#include "ffhdd/ffmpeg_demuxer.hpp"
#include "ffhdd/cuvid_decoder.hpp"
#include "ffhdd/cuda_tools.hpp"

using namespace std;

namespace py=pybind11;

class FFmpegDemuxer { 
public:
	FFmpegDemuxer(std::string uri, bool auto_reboot = false){

		instance_ = FFHDDemuxer::create_ffmpeg_demuxer(
			uri, 
			auto_reboot
		);
	}

	bool valid(){
		return instance_ != nullptr;
	}

	FFHDDemuxer::IAVCodecID get_video_codec() {return instance_->get_video_codec();}
	virtual FFHDDemuxer::IAVPixelFormat get_chroma_format(){return instance_->get_chroma_format();}
	virtual int get_width() {return instance_->get_width();}
	virtual int get_height() {return instance_->get_height();}
	virtual int get_bit_depth() {return instance_->get_bit_depth();}
	virtual int get_fps() {return instance_->get_fps();}
	virtual int get_total_frames() {return instance_->get_total_frames();}
	virtual py::tuple get_extra_data() {
		uint8_t* pdata = nullptr;
		int pbytes = 0;
		instance_->get_extra_data(&pdata, &pbytes);
		return py::make_tuple((uint64_t)pdata, pbytes);
	}

	virtual bool isreboot() {return instance_->isreboot();}
	virtual void reset_reboot_flag() {instance_->reset_reboot_flag();}
	virtual py::tuple demux() {

		uint8_t* pdata = nullptr;
		int pbytes = 0;
		bool iskey = false;
		bool ok = instance_->demux(&pdata, &pbytes, &time_pts_, &iskey);
		return py::make_tuple((uint64_t)pdata, pbytes, time_pts_, iskey, ok);
	}

	virtual bool reopen() {return instance_->reopen();}

private:
	int64_t time_pts_ = 0;
	shared_ptr<FFHDDemuxer::FFmpegDemuxer> instance_;
}; 

class CUVIDDecoder { 
public:
	CUVIDDecoder(bool bUseDeviceFrame, FFHDDemuxer::IAVCodecID eCodec, int max_cache, int gpu_id,
        int cl, int ct, int cr, int cb, int rw, int rh, bool output_bgr){
		
		FFHDDecoder::IcudaVideoCodec codec = FFHDDecoder::ffmpeg2NvCodecId(eCodec);
		FFHDDecoder::CropRect crop{0, 0, 0, 0};
		FFHDDecoder::ResizeDim resize{0, 0};
		if(cr - cl > 0 && cb - ct > 0){
			crop.l = cl;
			crop.t = ct;
			crop.r = cr;
			crop.b = cb;
		}

		if(rw > 0 && rh > 0){
			resize.w = rw;
			resize.h = rh;
		}

		output_bgr_ = output_bgr;
		instance_ = FFHDDecoder::create_cuvid_decoder(
			bUseDeviceFrame, codec, max_cache, gpu_id, &crop, &resize, output_bgr
		);
	}

	bool valid(){
		return instance_ != nullptr;
	}

	py::array get_numpy(uint8_t* ptr){
		py::array image;
		vector<int> shape;

		if(output_bgr_){
			shape = {instance_->get_height(), instance_->get_width(), 3};
		}else{
			shape = {int(instance_->get_height() * 1.5f), instance_->get_width()};
		}

		int gpu_id = instance_->device();
		if(instance_->is_gpu_frame()){
			image = py::array(py::dtype::of<unsigned char>(), shape);
			
			int old_id = 0;
			checkCudaRuntime(cudaGetDevice(&old_id));
			checkCudaRuntime(cudaSetDevice(gpu_id));
			checkCudaRuntime(cudaMemcpyAsync(image.mutable_data(0), ptr, instance_->get_frame_bytes(), cudaMemcpyDeviceToHost, instance_->get_stream()));
			checkCudaRuntime(cudaStreamSynchronize(instance_->get_stream()));
			checkCudaRuntime(cudaSetDevice(old_id));
		}else{
			image = py::array(py::dtype::of<unsigned char>(), shape, ptr);
		}
		return image;
	}

	int get_frame_bytes() {return instance_->get_frame_bytes();}
	int get_width() {return instance_->get_width();}
	int get_height() {return instance_->get_height();}
	unsigned int get_frame_index() {return instance_->get_frame_index();}
	unsigned int get_num_decoded_frame() {return instance_->get_num_decoded_frame();}
	py::tuple get_frame(bool return_numpy) {
		int64_t pts = 0;
		unsigned int frame_index;
		auto ptr = instance_->get_frame(&pts, &frame_index);

		if(return_numpy){
			return py::make_tuple((uint64_t)ptr, pts, frame_index, get_numpy(ptr));
		}else{
			return py::make_tuple((uint64_t)ptr, pts, frame_index);
		}
	}
	int decode(uint64_t pData, int nSize, int64_t nTimestamp=0) {
		const uint8_t* ptr = (const uint8_t*)pData;
		return instance_->decode(ptr, nSize, nTimestamp);
	}
	int64_t get_stream() {return (uint64_t)instance_->get_stream();}

private:
	shared_ptr<FFHDDecoder::CUVIDDecoder> instance_;
	bool output_bgr_ = false;
}; 

int main(){

	auto demuxer = FFHDDemuxer::create_ffmpeg_demuxer("exp/number100.mp4");
    if(demuxer == nullptr){
        printf("demuxer create failed\n");
        return 0;
    }

    auto decoder = FFHDDecoder::create_cuvid_decoder(
        false, FFHDDecoder::ffmpeg2NvCodecId(demuxer->get_video_codec()), -1, 0, nullptr, nullptr, true
    );

    if(decoder == nullptr){
        printf("decoder create failed\n");
        return 0;
    }

    uint8_t* packet_data = nullptr;
    int packet_size = 0;
    int64_t pts = 0;

    demuxer->get_extra_data(&packet_data, &packet_size);
    decoder->decode(packet_data, packet_size);

	printf("packet_size = %d\n", packet_size);

    unsigned int frame_index = 0;
    do{
        demuxer->demux(&packet_data, &packet_size, &pts);
        int ndecoded_frame = decoder->decode(packet_data, packet_size, pts);
        for(int i = 0; i < ndecoded_frame; ++i){

            /* 因为decoder获取的frame内存，是YUV-NV12格式的。储存内存大小是 [height * 1.5] * width byte
             因此构造一个height * 1.5,  width 大小的空间
             然后由opencv函数，把YUV-NV12转换到BGR，转换后的image则是正常的height, width, CV_8UC3
            */
            //cv::Mat image(decoder->get_height() * 1.5, decoder->get_width(), CV_8U, decoder->get_frame(&pts, &frame_index));
            //cv::cvtColor(image, image, cv::COLOR_YUV2BGR_NV12);
            frame_index = frame_index + 1;
            // INFO("write imgs/img_%05d.jpg  %dx%d", frame_index, decoder->get_width(), decoder->get_height());
        }
    }while(packet_size > 0);
	printf("C++ Demo: %d frames\n", frame_index);
	return 0;
}

PYBIND11_MODULE(libffhdd, m){

	py::class_<FFmpegDemuxer>(m, "FFmpegDemuxer")
		.def(py::init<string, bool>(), py::arg("uri"), py::arg("auto_reboot")=false)
		.def_property_readonly("valid", &FFmpegDemuxer::valid)
		.def("get_video_codec", &FFmpegDemuxer::get_video_codec)
		.def("get_chroma_format", &FFmpegDemuxer::get_chroma_format)
		.def("get_width", &FFmpegDemuxer::get_width)
		.def("get_height", &FFmpegDemuxer::get_height)
		.def("get_bit_depth", &FFmpegDemuxer::get_bit_depth)
		.def("get_fps", &FFmpegDemuxer::get_fps)
		.def("get_total_frames", &FFmpegDemuxer::get_total_frames)
		.def("get_extra_data", &FFmpegDemuxer::get_extra_data)
		.def("isreboot", &FFmpegDemuxer::isreboot)
		.def("reset_reboot_flag", &FFmpegDemuxer::reset_reboot_flag)
		.def("demux", &FFmpegDemuxer::demux)
		.def("reopen", &FFmpegDemuxer::reopen);

	py::class_<CUVIDDecoder>(m, "CUVIDDecoder")
		.def(py::init<bool, FFHDDemuxer::IAVCodecID, int, int, int, int, int, int, int, int, bool>(), 
			py::arg("bUseDeviceFrame"), py::arg("codec"), py::arg("max_cache"), py::arg("gpu_id"),
			py::arg("crop_left")=0, py::arg("crop_top")=0, py::arg("crop_right")=0, py::arg("crop_bottom")=0,
			py::arg("resize_width")=0, py::arg("resize_height")=0,
			py::arg("output_bgr")=false
		)
		.def_property_readonly("valid", &CUVIDDecoder::valid)
		.def("get_frame_bytes", &CUVIDDecoder::get_frame_bytes)
		.def("get_width", &CUVIDDecoder::get_width)
		.def("get_height", &CUVIDDecoder::get_height)
		.def("get_frame_index", &CUVIDDecoder::get_frame_index)
		.def("get_num_decoded_frame", &CUVIDDecoder::get_num_decoded_frame)
		.def("get_frame", &CUVIDDecoder::get_frame, py::arg("return_numpy")=false)
		.def("decode", &CUVIDDecoder::decode)
		.def("get_stream", &CUVIDDecoder::get_stream);
};