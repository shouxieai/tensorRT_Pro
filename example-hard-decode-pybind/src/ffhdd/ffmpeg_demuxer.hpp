
#ifndef FFMPEG_DEMUXER_HPP
#define FFMPEG_DEMUXER_HPP

#include <stdint.h>
#include <memory>
#include <string>


namespace FFHDDemuxer{

    typedef int IAVCodecID;
    typedef int IAVPixelFormat;

    class DataProvider {
    public:
        virtual int get_data(uint8_t *pBuf, int nBuf) = 0;
    };

    class FFmpegDemuxer{
    public:
        virtual IAVCodecID get_video_codec() = 0;
        virtual IAVPixelFormat get_chroma_format() = 0;
        virtual int get_width() = 0;
        virtual int get_height() = 0;
        virtual int get_bit_depth() = 0;
        virtual int get_fps() = 0;
        virtual int get_total_frames() = 0;
        virtual void get_extra_data(uint8_t **ppData, int *bytes) = 0;
        virtual bool isreboot() = 0;
        virtual void reset_reboot_flag() = 0;
        virtual bool demux(uint8_t **ppVideo, int *pnVideoBytes, int64_t *pts = nullptr, bool *iskey_frame = nullptr) = 0;
        virtual bool reopen() = 0;
    };

    std::shared_ptr<FFmpegDemuxer> create_ffmpeg_demuxer(const std::string& uri, bool auto_reboot = false);
    std::shared_ptr<FFmpegDemuxer> create_ffmpeg_demuxer(std::shared_ptr<DataProvider> provider);
}; // namespace FFHDDemuxer

#endif // FFMPEG_DEMUXER_HPP