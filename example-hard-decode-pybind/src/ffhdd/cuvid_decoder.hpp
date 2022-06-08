
#ifndef CUVID_DECODER_HPP
#define CUVID_DECODER_HPP

#include <memory>
// 就不用在这里包含cuda_runtime.h

struct CUstream_st;

namespace FFHDDecoder{

    #define IcudaVideoCodec_H264            4

    typedef CUstream_st* ICUStream;
    typedef unsigned int IcudaVideoCodec;

    struct CropRect {
        int l, t, r, b;
    };

    struct ResizeDim {
        int w, h;
    };

    class CUVIDDecoder{
    public:
        virtual int get_frame_bytes() = 0;
        virtual int get_width() = 0;
        virtual int get_height() = 0;
        virtual unsigned int get_frame_index() = 0;
        virtual unsigned int get_num_decoded_frame() = 0;
        virtual uint8_t* get_frame(int64_t* pTimestamp = nullptr, unsigned int* pFrameIndex = nullptr) = 0;
        virtual int decode(const uint8_t *pData, int nSize, int64_t nTimestamp=0) = 0;
        virtual ICUStream get_stream() = 0;
        virtual int device() = 0;
        virtual bool is_gpu_frame() = 0;
    };

    IcudaVideoCodec ffmpeg2NvCodecId(int ffmpeg_codec_id);

    /* max_cache 取 -1 时，无限缓存，根据实际情况缓存。实际上一般不超过5帧 */
    // gpu_id = -1, current_device_id
    std::shared_ptr<CUVIDDecoder> create_cuvid_decoder(
        bool use_device_frame, IcudaVideoCodec codec, int max_cache = -1, int gpu_id = -1, 
        const CropRect *crop_rect = nullptr, const ResizeDim *resize_dim = nullptr, bool output_bgr = false
    );
}; // FFHDDecoder

#endif // CUVID_DECODER_HPP