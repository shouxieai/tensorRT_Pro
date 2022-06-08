
#include "ffmpeg_demuxer.hpp"
#include <iostream>
#include "simple-logger.hpp"

extern "C"
{
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libavutil/opt.h>
    #include <libswscale/swscale.h>
};

using namespace std;

namespace FFHDDemuxer{

    static inline bool check_ffmpeg_retvalue(int e, const char* call, int iLine, const char *szFile) {
        if (e < 0) {
            std::cout << "FFMPEGDemuxer error " << call << ", cdoe = " << e << " in file " << szFile << ":" << iLine << std::endl;
            return false;
        }
        return true;
    }

    #define checkFFMPEG(call) check_ffmpeg_retvalue(call, #call, __LINE__, __FILE__)

    static bool string_begin_with(const string& str, const string& with){

        if(str.size() < with.size()) return false;
        if(with.empty()) return true;

        return memcmp(str.c_str(), with.c_str(), with.size()) == 0;
    }

    class FFmpegDemuxerImpl : public FFmpegDemuxer{
    public:
        bool open(const string& uri, bool auto_reboot = true, int64_t timescale = 1000 /*Hz*/){
            this->uri_opened_ = uri;
            this->time_scale_opened_ = timescale;
            this->auto_reboot_ = auto_reboot;
            return this->open(this->CreateFormatContext(uri), timescale);
        }

        bool open(shared_ptr<DataProvider> pDataProvider){
            bool ok = this->open(this->CreateFormatContext(pDataProvider));
            if(ok){
                m_avioc = m_fmtc->pb;
            }
            return ok;
        }

        bool reopen() override{
            if(m_pDataProvider)  return false;
            if(!flag_is_opened_) return false;

            close();
            return this->open(this->uri_opened_, this->auto_reboot_, this->time_scale_opened_);
        }

        void close(){
            if (!m_fmtc) 
                return;

            if (m_pkt.data) {
                av_packet_unref(&m_pkt);
            }
            if (m_pktFiltered.data) {
                av_packet_unref(&m_pktFiltered);
            }

            if (m_bsfc) {
                av_bsf_free(&m_bsfc);
            }
            
            avformat_close_input(&m_fmtc);

            if (m_avioc) {
                av_freep(&m_avioc->buffer);
                av_freep(&m_avioc);
            }

            if (m_pDataWithHeader) {
                av_free(m_pDataWithHeader);
            }
            flag_is_opened_ = false;
        }

        ~FFmpegDemuxerImpl() {
            close();
        }

        IAVCodecID get_video_codec() override{
            return m_eVideoCodec;
        }

        IAVPixelFormat get_chroma_format() override{
            return m_eChromaFormat;
        }

        int get_fps() override{
            return m_fps;
        }

        int get_total_frames() override{
            return m_total_frames;
        }

        int get_width() override{
            return m_nWidth;
        }

        int get_height() override{
            return m_nHeight;
        }

        int get_bit_depth() override{
            return m_nBitDepth;
        }

        int get_frame_size() {
            return m_nWidth * (m_nHeight + m_nChromaHeight) * m_nBPP;
        }

        void get_extra_data(uint8_t **ppData, int *bytes) override{

            // AVBitStreamFilterContext* bsfc = av_bitstream_filter_init("h264_mp4toannexb");
            // av_bitstream_filter_filter(bsfc, m_fmtc->streams[m_iVideoStream]->codec, nullptr, ppData, bytes, *ppData, *bytes, 0);
            // av_bitstream_filter_close(bsfc);
            *ppData = m_fmtc->streams[m_iVideoStream]->codec->extradata;
            *bytes = m_fmtc->streams[m_iVideoStream]->codec->extradata_size;
        }

        bool demux(uint8_t **ppVideo, int *pnVideoBytes, int64_t *pts = nullptr, bool *iskey_frame = nullptr) override{
            
            *pnVideoBytes = 0;
            *ppVideo = nullptr;

            if (!m_fmtc) {
                return false;
            }

            if (m_pkt.data) {
                av_packet_unref(&m_pkt);
            }

            int e = 0;
            while ((e = av_read_frame(m_fmtc, &m_pkt)) >= 0 && m_pkt.stream_index != m_iVideoStream) 
                av_packet_unref(&m_pkt);

            if(iskey_frame){
                *iskey_frame = m_pkt.flags & AV_PKT_FLAG_KEY;
            }

            if (e < 0) {
                if(auto_reboot_){
                    bool open_ok = this->reopen();
                    if(!open_ok){
                        INFOE("Reopen failed.");
                        return false;
                    }
                    is_reboot_ = true;
                    return this->demux(ppVideo, pnVideoBytes, pts);
                }
                return false;
            }

            int64_t local_pts = 0;
            if (m_bMp4H264 || m_bMp4HEVC) {
                if (m_pktFiltered.data) {
                    av_packet_unref(&m_pktFiltered);
                }
                checkFFMPEG(av_bsf_send_packet(m_bsfc, &m_pkt));
                checkFFMPEG(av_bsf_receive_packet(m_bsfc, &m_pktFiltered));
                *ppVideo = m_pktFiltered.data;
                *pnVideoBytes = m_pktFiltered.size;
                local_pts = (int64_t) (m_pktFiltered.pts * m_userTimeScale * m_timeBase);
            } else {

                if (m_bMp4MPEG4 && (m_frameCount == 0)) {

                    int extraDataSize = m_fmtc->streams[m_iVideoStream]->codecpar->extradata_size;

                    if (extraDataSize > 0) {

                        // extradata contains start codes 00 00 01. Subtract its size
                        m_pDataWithHeader = (uint8_t *)av_malloc(extraDataSize + m_pkt.size - 3*sizeof(uint8_t));

                        if (!m_pDataWithHeader) {
                            INFOE("FFmpeg error, m_pDataWithHeader alloc failed");
                            return false;
                        }

                        memcpy(m_pDataWithHeader, m_fmtc->streams[m_iVideoStream]->codecpar->extradata, extraDataSize);
                        memcpy(m_pDataWithHeader+extraDataSize, m_pkt.data+3, m_pkt.size - 3*sizeof(uint8_t));

                        *ppVideo = m_pDataWithHeader;
                        *pnVideoBytes = extraDataSize + m_pkt.size - 3*sizeof(uint8_t);
                    }

                } else {
                    *ppVideo = m_pkt.data;
                    *pnVideoBytes = m_pkt.size;
                }
                local_pts = (int64_t)(m_pkt.pts * m_userTimeScale * m_timeBase);
            }

            if(pts)
                *pts = local_pts;
            m_frameCount++;
            return true;
        }

        virtual bool isreboot() override{
            return is_reboot_;
        }

        virtual void reset_reboot_flag() override{
            is_reboot_ = false;
        }

        static int ReadPacket(void *opaque, uint8_t *pBuf, int nBuf) {
            return ((DataProvider *)opaque)->get_data(pBuf, nBuf);
        }

    private:
        double r2d(AVRational r) const{
            return r.num == 0 || r.den == 0 ? 0. : (double)r.num / (double)r.den;
        }

        bool open(AVFormatContext *fmtc, int64_t timeScale = 1000 /*Hz*/) {
            if (!fmtc) {
                INFOE("No AVFormatContext provided.");
                return false;
            }

            this->m_fmtc = fmtc;
            // LOG(LINFO) << "Media format: " << fmtc->iformat->long_name << " (" << fmtc->iformat->name << ")";

            if(!checkFFMPEG(avformat_find_stream_info(fmtc, nullptr))) return false;
            m_iVideoStream = av_find_best_stream(fmtc, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
            if (m_iVideoStream < 0) {
                INFOE("FFmpeg error: Could not find stream in input file");
                return false;
            }

            m_frameCount = 0;
            //fmtc->streams[iVideoStream]->need_parsing = AVSTREAM_PARSE_NONE;
            m_eVideoCodec = fmtc->streams[m_iVideoStream]->codecpar->codec_id;
            m_nWidth = fmtc->streams[m_iVideoStream]->codecpar->width;
            m_nHeight = fmtc->streams[m_iVideoStream]->codecpar->height;
            m_eChromaFormat = (AVPixelFormat)fmtc->streams[m_iVideoStream]->codecpar->format;
            AVRational rTimeBase = fmtc->streams[m_iVideoStream]->time_base;
            m_timeBase = av_q2d(rTimeBase);
            m_userTimeScale = timeScale;
            m_fps = r2d(fmtc->streams[m_iVideoStream]->avg_frame_rate);
            m_total_frames = fmtc->streams[m_iVideoStream]->nb_frames;

            // Set bit depth, chroma height, bits per pixel based on eChromaFormat of input
            switch (m_eChromaFormat)
            {
            case AV_PIX_FMT_YUV420P10LE:
                m_nBitDepth = 10;
                m_nChromaHeight = (m_nHeight + 1) >> 1;
                m_nBPP = 2;
                break;
            case AV_PIX_FMT_YUV420P12LE:
                m_nBitDepth = 12;
                m_nChromaHeight = (m_nHeight + 1) >> 1;
                m_nBPP = 2;
                break;
            case AV_PIX_FMT_YUV444P10LE:
                m_nBitDepth = 10;
                m_nChromaHeight = m_nHeight << 1;
                m_nBPP = 2;
                break;
            case AV_PIX_FMT_YUV444P12LE:
                m_nBitDepth = 12;
                m_nChromaHeight = m_nHeight << 1;
                m_nBPP = 2;
                break;
            case AV_PIX_FMT_YUV444P:
                m_nBitDepth = 8;
                m_nChromaHeight = m_nHeight << 1;
                m_nBPP = 1;
                break;
            case AV_PIX_FMT_YUV420P:
            case AV_PIX_FMT_YUVJ420P:
            case AV_PIX_FMT_YUVJ422P:   // jpeg decoder output is subsampled to NV12 for 422/444 so treat it as 420
            case AV_PIX_FMT_YUVJ444P:   // jpeg decoder output is subsampled to NV12 for 422/444 so treat it as 420
                m_nBitDepth = 8;
                m_nChromaHeight = (m_nHeight + 1) >> 1;
                m_nBPP = 1;
                break;
            default:
                INFOW("ChromaFormat not recognized. Assuming 420");
                m_nBitDepth = 8;
                m_nChromaHeight = (m_nHeight + 1) >> 1;
                m_nBPP = 1;
            }

            m_bMp4H264 = m_eVideoCodec == AV_CODEC_ID_H264 && (
                    !strcmp(fmtc->iformat->long_name, "QuickTime / MOV") 
                    || !strcmp(fmtc->iformat->long_name, "FLV (Flash Video)") 
                    || !strcmp(fmtc->iformat->long_name, "Matroska / WebM")
                );
            m_bMp4HEVC = m_eVideoCodec == AV_CODEC_ID_HEVC && (
                    !strcmp(fmtc->iformat->long_name, "QuickTime / MOV")
                    || !strcmp(fmtc->iformat->long_name, "FLV (Flash Video)")
                    || !strcmp(fmtc->iformat->long_name, "Matroska / WebM")
                );

            m_bMp4MPEG4 = m_eVideoCodec == AV_CODEC_ID_MPEG4 && (
                    !strcmp(fmtc->iformat->long_name, "QuickTime / MOV")
                    || !strcmp(fmtc->iformat->long_name, "FLV (Flash Video)")
                    || !strcmp(fmtc->iformat->long_name, "Matroska / WebM")
                );

            //Initialize packet fields with default values
            av_init_packet(&m_pkt);
            m_pkt.data = nullptr;
            m_pkt.size = 0;
            
            av_init_packet(&m_pktFiltered);
            m_pktFiltered.data = nullptr;
            m_pktFiltered.size = 0;

            // Initialize bitstream filter and its required resources
            if (m_bMp4H264) {
                const AVBitStreamFilter *bsf = av_bsf_get_by_name("h264_mp4toannexb");
                if (!bsf) {
                    INFOE("FFmpeg error: av_bsf_get_by_name() failed");
                    return false;
                }
                if(!checkFFMPEG(av_bsf_alloc(bsf, &m_bsfc))) return false;
                avcodec_parameters_copy(m_bsfc->par_in, fmtc->streams[m_iVideoStream]->codecpar);
                if(!checkFFMPEG(av_bsf_init(m_bsfc))) return false;
            }
            if (m_bMp4HEVC) {
                const AVBitStreamFilter *bsf = av_bsf_get_by_name("hevc_mp4toannexb");
                if (!bsf) {
                    INFOE("FFmpeg error: av_bsf_get_by_name() failed");
                    return false;
                }
                if(!checkFFMPEG(av_bsf_alloc(bsf, &m_bsfc))) return false;
                avcodec_parameters_copy(m_bsfc->par_in, fmtc->streams[m_iVideoStream]->codecpar);
                if(!checkFFMPEG(av_bsf_init(m_bsfc))) return false;
            }
            this->flag_is_opened_ = true;
            return true;
        }

        AVFormatContext *CreateFormatContext(shared_ptr<DataProvider> pDataProvider) {
            
            AVFormatContext *ctx = nullptr;
            if (!(ctx = avformat_alloc_context())) {
                INFOE("FFmpeg error");
                return nullptr;
            }

            uint8_t *avioc_buffer = nullptr;
            int avioc_buffer_size = 8 * 1024 * 1024;
            avioc_buffer = (uint8_t *)av_malloc(avioc_buffer_size);
            if (!avioc_buffer) {
                INFOE("FFmpeg error");
                return nullptr;
            }

            m_pDataProvider = pDataProvider;
            m_avioc = avio_alloc_context(avioc_buffer, avioc_buffer_size,
                0, pDataProvider.get(), &ReadPacket, nullptr, nullptr);
            if (!m_avioc) {
                INFOE("FFmpeg error");
                return nullptr;
            }
            ctx->pb = m_avioc;

            // 如果open失败，ctx会设置为nullptr
            checkFFMPEG(avformat_open_input(&ctx, nullptr, nullptr, nullptr));
            return ctx;
        }

        AVFormatContext *CreateFormatContext(const string& uri) {
            avformat_network_init();

            AVDictionary* options = nullptr;
            if (string_begin_with(uri, "rtsp://")){
                av_dict_set(&options, "rtsp_transport", "tcp", 0);
                av_dict_set(&options, "buffer_size", "1024000", 0); /* 设置缓存大小，1080p可将值调大 */
                av_dict_set(&options, "stimeout", "2000000", 0); /* 设置超时断开连接时间，单位微秒 */
                av_dict_set(&options, "max_delay", "1000000", 0); /* 设置最大时延，单位微秒 */
            }

            // 如果open失败，ctx会设置为nullptr
            AVFormatContext *ctx = nullptr;
            checkFFMPEG(avformat_open_input(&ctx, uri.c_str(), nullptr, &options));
            return ctx;
        }

    private:
        shared_ptr<DataProvider> m_pDataProvider;
        AVFormatContext *m_fmtc = nullptr;
        AVIOContext *m_avioc = nullptr;
        AVPacket m_pkt, m_pktFiltered; /*!< AVPacket stores compressed data typically exported by demuxers and then passed as input to decoders */
        AVBSFContext *m_bsfc = nullptr;
        int m_fps = 0;
        int m_total_frames = 0;
        int m_iVideoStream = -1;
        bool m_bMp4H264, m_bMp4HEVC, m_bMp4MPEG4;
        AVCodecID m_eVideoCodec;
        AVPixelFormat m_eChromaFormat;
        int m_nWidth, m_nHeight, m_nBitDepth, m_nBPP, m_nChromaHeight;
        double m_timeBase = 0.0;
        int64_t m_userTimeScale = 0; 
        uint8_t *m_pDataWithHeader = nullptr;
        unsigned int m_frameCount = 0;
        string uri_opened_;
        int64_t time_scale_opened_ = 0;
        bool flag_is_opened_ = false;
        bool auto_reboot_ = false;
        bool is_reboot_ = false;
    };


    std::shared_ptr<FFmpegDemuxer> create_ffmpeg_demuxer(const std::string& path, bool auto_reboot){
        std::shared_ptr<FFmpegDemuxerImpl> instance(new FFmpegDemuxerImpl());
        if(!instance->open(path, auto_reboot))
            instance.reset();
        return instance;
    }

    std::shared_ptr<FFmpegDemuxer> create_ffmpeg_demuxer(std::shared_ptr<DataProvider> provider){
        std::shared_ptr<FFmpegDemuxerImpl> instance(new FFmpegDemuxerImpl());
        if(!instance->open(provider))
            instance.reset();
        return instance;
    }
}; // FFHDDemuxer