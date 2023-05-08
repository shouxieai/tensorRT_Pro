
#include "cuvid_decoder.hpp"
#include "cuda_tools.hpp"
#include <nvcuvid.h>
#include <mutex>
#include <vector>
#include <sstream>
#include <string.h>
#include <assert.h>

using namespace std;


void convert_nv12_to_bgr_invoker(
    const uint8_t* y, const uint8_t* uv, int width, int height, int linesize, uint8_t* dst_bgr,
    cudaStream_t stream
);

namespace FFHDDecoder{
    static float GetChromaHeightFactor(cudaVideoSurfaceFormat eSurfaceFormat)
    {
        float factor = 0.5;
        switch (eSurfaceFormat)
        {
        case cudaVideoSurfaceFormat_NV12:
        case cudaVideoSurfaceFormat_P016:
            factor = 0.5;
            break;
        case cudaVideoSurfaceFormat_YUV444:
        case cudaVideoSurfaceFormat_YUV444_16Bit:
            factor = 1.0;
            break;
        }

        return factor;
    }

    static int GetChromaPlaneCount(cudaVideoSurfaceFormat eSurfaceFormat)
    {
        int numPlane = 1;
        switch (eSurfaceFormat)
        {
        case cudaVideoSurfaceFormat_NV12:
        case cudaVideoSurfaceFormat_P016:
            numPlane = 1;
            break;
        case cudaVideoSurfaceFormat_YUV444:
        case cudaVideoSurfaceFormat_YUV444_16Bit:
            numPlane = 2;
            break;
        }

        return numPlane;
    }

    IcudaVideoCodec ffmpeg2NvCodecId(int ffmpeg_codec_id) {
        switch (ffmpeg_codec_id) {
            /*AV_CODEC_ID_MPEG1VIDEO*/ case 1   : return cudaVideoCodec_MPEG1;        
            /*AV_CODEC_ID_MPEG2VIDEO*/ case 2   : return cudaVideoCodec_MPEG2;        
            /*AV_CODEC_ID_MPEG4*/ case 12       : return cudaVideoCodec_MPEG4;        
            /*AV_CODEC_ID_VC1*/ case 70         : return cudaVideoCodec_VC1;          
            /*AV_CODEC_ID_H264*/ case 27        : return cudaVideoCodec_H264;         
            /*AV_CODEC_ID_HEVC*/ case 173       : return cudaVideoCodec_HEVC;         
            /*AV_CODEC_ID_VP8*/ case 139        : return cudaVideoCodec_VP8;          
            /*AV_CODEC_ID_VP9*/ case 167        : return cudaVideoCodec_VP9;          
            /*AV_CODEC_ID_MJPEG*/ case 7        : return cudaVideoCodec_JPEG;         
            default                             : return cudaVideoCodec_NumCodecs;
        }
    }

    class CUVIDDecoderImpl : public CUVIDDecoder{
    public:
        bool create(bool bUseDeviceFrame, int gpu_id, cudaVideoCodec eCodec, bool bLowLatency = false,
                const CropRect *pCropRect = nullptr, const ResizeDim *pResizeDim = nullptr, int max_cache = -1,
                int maxWidth = 0, int maxHeight = 0, unsigned int clkRate = 1000, bool output_bgr=false)
            {
            
            m_bUseDeviceFrame = bUseDeviceFrame;
            m_eCodec = eCodec;
            m_nMaxWidth = maxWidth;
            m_nMaxHeight = maxHeight;
            m_nMaxCache  = max_cache;
            m_gpuID      = gpu_id;
            m_output_bgr = output_bgr;

            if(m_gpuID == -1){
                checkCudaRuntime(cudaGetDevice(&m_gpuID));
            }

            CUDATools::AutoDevice auto_device_exchange(m_gpuID);
            if (pCropRect) m_cropRect = *pCropRect;
            if (pResizeDim) m_resizeDim = *pResizeDim;
            CUcontext cuContext = nullptr;
            checkCudaDriver(cuCtxGetCurrent(&cuContext));

            if(cuContext == nullptr){
                INFOE("Current Context is nullptr.");
                return false;
            }

            if(!checkCudaDriver(cuvidCtxLockCreate(&m_ctxLock, cuContext))) return false;
            if(!checkCudaRuntime(cudaStreamCreate(&m_cuvidStream))) return false;

            CUVIDPARSERPARAMS videoParserParameters = {};
            videoParserParameters.CodecType = eCodec;
            videoParserParameters.ulMaxNumDecodeSurfaces = 1;
            videoParserParameters.ulClockRate = clkRate;
            videoParserParameters.ulMaxDisplayDelay = bLowLatency ? 0 : 1;
            videoParserParameters.pUserData = this;
            videoParserParameters.pfnSequenceCallback = handleVideoSequenceProc;
            videoParserParameters.pfnDecodePicture = handlePictureDecodeProc;
            videoParserParameters.pfnDisplayPicture = handlePictureDisplayProc;
            if(!checkCudaDriver(cuvidCreateVideoParser(&m_hParser, &videoParserParameters))) return false;
            return true;
        }

        int decode(const uint8_t *pData, int nSize, int64_t nTimestamp=0) override
        {
            m_nDecodedFrame = 0;
            m_nDecodedFrameReturned = 0;
            CUVIDSOURCEDATAPACKET packet = { 0 };
            packet.payload = pData;
            packet.payload_size = nSize;
            packet.flags = CUVID_PKT_TIMESTAMP;
            packet.timestamp = nTimestamp;
            if (!pData || nSize == 0) {
                packet.flags |= CUVID_PKT_ENDOFSTREAM;
            }

            try{
                CUDATools::AutoDevice auto_device_exchange(m_gpuID);
                if(!checkCudaDriver(cuvidParseVideoData(m_hParser, &packet)))
                    return -1;
            }catch(...){
                return -1;
            }
            return m_nDecodedFrame;
        }

        static int CUDAAPI handleVideoSequenceProc(void *pUserData, CUVIDEOFORMAT *pVideoFormat) { return ((CUVIDDecoderImpl *)pUserData)->handleVideoSequence(pVideoFormat); }
        static int CUDAAPI handlePictureDecodeProc(void *pUserData, CUVIDPICPARAMS *pPicParams) { return ((CUVIDDecoderImpl *)pUserData)->handlePictureDecode(pPicParams); }
        static int CUDAAPI handlePictureDisplayProc(void *pUserData, CUVIDPARSERDISPINFO *pDispInfo) { return ((CUVIDDecoderImpl *)pUserData)->handlePictureDisplay(pDispInfo); }
        
        virtual int device() override{
            return this->m_gpuID;
        }

        virtual bool is_gpu_frame() override{
            return this->m_bUseDeviceFrame;
        }

        int handleVideoSequence(CUVIDEOFORMAT *pVideoFormat){
            int nDecodeSurface = pVideoFormat->min_num_decode_surfaces;
            CUVIDDECODECAPS decodecaps;
            memset(&decodecaps, 0, sizeof(decodecaps));

            decodecaps.eCodecType = pVideoFormat->codec;
            decodecaps.eChromaFormat = pVideoFormat->chroma_format;
            decodecaps.nBitDepthMinus8 = pVideoFormat->bit_depth_luma_minus8;

            checkCudaDriver(cuvidGetDecoderCaps(&decodecaps));
            if(!decodecaps.bIsSupported){
                throw std::runtime_error("Codec not supported on this GPU");
                return nDecodeSurface;
            }

            if ((pVideoFormat->coded_width > decodecaps.nMaxWidth) ||
                (pVideoFormat->coded_height > decodecaps.nMaxHeight)){

                std::ostringstream errorString;
                errorString << std::endl
                            << "Resolution          : " << pVideoFormat->coded_width << "x" << pVideoFormat->coded_height << std::endl
                            << "Max Supported (wxh) : " << decodecaps.nMaxWidth << "x" << decodecaps.nMaxHeight << std::endl
                            << "Resolution not supported on this GPU";

                const std::string cErr = errorString.str();
                throw std::runtime_error(cErr);
                return nDecodeSurface;
            }

            if ((pVideoFormat->coded_width>>4)*(pVideoFormat->coded_height>>4) > decodecaps.nMaxMBCount){

                std::ostringstream errorString;
                errorString << std::endl
                            << "MBCount             : " << (pVideoFormat->coded_width >> 4)*(pVideoFormat->coded_height >> 4) << std::endl
                            << "Max Supported mbcnt : " << decodecaps.nMaxMBCount << std::endl
                            << "MBCount not supported on this GPU";

                const std::string cErr = errorString.str();
                throw std::runtime_error(cErr);
                return nDecodeSurface;
            }

            // eCodec has been set in the constructor (for parser). Here it's set again for potential correction
            m_eCodec = pVideoFormat->codec;
            m_eChromaFormat = pVideoFormat->chroma_format;
            m_nBitDepthMinus8 = pVideoFormat->bit_depth_luma_minus8;
            m_nBPP = m_nBitDepthMinus8 > 0 ? 2 : 1;

            // Set the output surface format same as chroma format
            if (m_eChromaFormat == cudaVideoChromaFormat_420)
                m_eOutputFormat = pVideoFormat->bit_depth_luma_minus8 ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12;
            else if (m_eChromaFormat == cudaVideoChromaFormat_444)
                m_eOutputFormat = pVideoFormat->bit_depth_luma_minus8 ? cudaVideoSurfaceFormat_YUV444_16Bit : cudaVideoSurfaceFormat_YUV444;
            else if (m_eChromaFormat == cudaVideoChromaFormat_422)
                m_eOutputFormat = cudaVideoSurfaceFormat_NV12;  // no 4:2:2 output format supported yet so make 420 default

            // Check if output format supported. If not, check falback options
            if (!(decodecaps.nOutputFormatMask & (1 << m_eOutputFormat)))
            {
                if (decodecaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_NV12))
                    m_eOutputFormat = cudaVideoSurfaceFormat_NV12;
                else if (decodecaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_P016))
                    m_eOutputFormat = cudaVideoSurfaceFormat_P016;
                else if (decodecaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_YUV444))
                    m_eOutputFormat = cudaVideoSurfaceFormat_YUV444;
                else if (decodecaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_YUV444_16Bit))
                    m_eOutputFormat = cudaVideoSurfaceFormat_YUV444_16Bit;
                else 
                    throw std::runtime_error("No supported output format found");
            }
            m_videoFormat = *pVideoFormat;

            CUVIDDECODECREATEINFO videoDecodeCreateInfo = { 0 };
            videoDecodeCreateInfo.CodecType = pVideoFormat->codec;
            videoDecodeCreateInfo.ChromaFormat = pVideoFormat->chroma_format;
            videoDecodeCreateInfo.OutputFormat = m_eOutputFormat;
            videoDecodeCreateInfo.bitDepthMinus8 = pVideoFormat->bit_depth_luma_minus8;
            if (pVideoFormat->progressive_sequence)
                videoDecodeCreateInfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
            else
                videoDecodeCreateInfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Adaptive;
            videoDecodeCreateInfo.ulNumOutputSurfaces = 2;
            // With PreferCUVID, JPEG is still decoded by CUDA while video is decoded by NVDEC hardware
            videoDecodeCreateInfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;
            videoDecodeCreateInfo.ulNumDecodeSurfaces = nDecodeSurface;
            videoDecodeCreateInfo.vidLock = m_ctxLock;
            videoDecodeCreateInfo.ulWidth = pVideoFormat->coded_width;
            videoDecodeCreateInfo.ulHeight = pVideoFormat->coded_height;
            if (m_nMaxWidth < (int)pVideoFormat->coded_width)
                m_nMaxWidth = pVideoFormat->coded_width;
            if (m_nMaxHeight < (int)pVideoFormat->coded_height)
                m_nMaxHeight = pVideoFormat->coded_height;
            videoDecodeCreateInfo.ulMaxWidth = m_nMaxWidth;
            videoDecodeCreateInfo.ulMaxHeight = m_nMaxHeight;

            if (!(m_cropRect.r && m_cropRect.b) && !(m_resizeDim.w && m_resizeDim.h)) {
                m_nWidth = pVideoFormat->display_area.right - pVideoFormat->display_area.left;
                m_nLumaHeight = pVideoFormat->display_area.bottom - pVideoFormat->display_area.top;
                videoDecodeCreateInfo.ulTargetWidth = pVideoFormat->coded_width;
                videoDecodeCreateInfo.ulTargetHeight = pVideoFormat->coded_height;
            } else {
                if (m_resizeDim.w && m_resizeDim.h) {
                    videoDecodeCreateInfo.display_area.left = pVideoFormat->display_area.left;
                    videoDecodeCreateInfo.display_area.top = pVideoFormat->display_area.top;
                    videoDecodeCreateInfo.display_area.right = pVideoFormat->display_area.right;
                    videoDecodeCreateInfo.display_area.bottom = pVideoFormat->display_area.bottom;
                    m_nWidth = m_resizeDim.w;
                    m_nLumaHeight = m_resizeDim.h;
                }

                if (m_cropRect.r && m_cropRect.b) {
                    videoDecodeCreateInfo.display_area.left = m_cropRect.l;
                    videoDecodeCreateInfo.display_area.top = m_cropRect.t;
                    videoDecodeCreateInfo.display_area.right = m_cropRect.r;
                    videoDecodeCreateInfo.display_area.bottom = m_cropRect.b;
                    m_nWidth = m_cropRect.r - m_cropRect.l;
                    m_nLumaHeight = m_cropRect.b - m_cropRect.t;
                }
                videoDecodeCreateInfo.ulTargetWidth = m_nWidth;
                videoDecodeCreateInfo.ulTargetHeight = m_nLumaHeight;
            }

            m_nChromaHeight = (int)(m_nLumaHeight * GetChromaHeightFactor(m_eOutputFormat));
            m_nNumChromaPlanes = GetChromaPlaneCount(m_eOutputFormat);
            m_nSurfaceHeight = videoDecodeCreateInfo.ulTargetHeight;
            m_nSurfaceWidth = videoDecodeCreateInfo.ulTargetWidth;
            m_displayRect.b = videoDecodeCreateInfo.display_area.bottom;
            m_displayRect.t = videoDecodeCreateInfo.display_area.top;
            m_displayRect.l = videoDecodeCreateInfo.display_area.left;
            m_displayRect.r = videoDecodeCreateInfo.display_area.right;

            checkCudaDriver(cuvidCreateDecoder(&m_hDecoder, &videoDecodeCreateInfo));
            return nDecodeSurface;
        }

        int handlePictureDecode(CUVIDPICPARAMS *pPicParams){

            if (!m_hDecoder)
            {
                throw std::runtime_error("Decoder not initialized.");
                return false;
            }
            m_nPicNumInDecodeOrder[pPicParams->CurrPicIdx] = m_nDecodePicCnt++;
            checkCudaDriver(cuvidDecodePicture(m_hDecoder, pPicParams));
            return 1;
        }

        int handlePictureDisplay(CUVIDPARSERDISPINFO *pDispInfo){
            CUVIDPROCPARAMS videoProcessingParameters = {};
            videoProcessingParameters.progressive_frame = pDispInfo->progressive_frame;
            videoProcessingParameters.second_field = pDispInfo->repeat_first_field + 1;
            videoProcessingParameters.top_field_first = pDispInfo->top_field_first;
            videoProcessingParameters.unpaired_field = pDispInfo->repeat_first_field < 0;
            videoProcessingParameters.output_stream = m_cuvidStream;

            CUdeviceptr dpSrcFrame = 0;
            unsigned int nSrcPitch = 0;
            checkCudaDriver(cuvidMapVideoFrame(m_hDecoder, pDispInfo->picture_index, &dpSrcFrame,
                &nSrcPitch, &videoProcessingParameters));

            CUVIDGETDECODESTATUS DecodeStatus;
            memset(&DecodeStatus, 0, sizeof(DecodeStatus));

            CUresult result = cuvidGetDecodeStatus(m_hDecoder, pDispInfo->picture_index, &DecodeStatus);
            if (result == CUDA_SUCCESS && (DecodeStatus.decodeStatus == cuvidDecodeStatus_Error || DecodeStatus.decodeStatus == cuvidDecodeStatus_Error_Concealed))
            {
                printf("Decode Error occurred for picture %d\n", m_nPicNumInDecodeOrder[pDispInfo->picture_index]);
            }

            uint8_t *pDecodedFrame = nullptr;
            {
                if ((unsigned)++m_nDecodedFrame > m_vpFrame.size())
                {
                    /*
                        如果超过了缓存限制，则覆盖最后一个图
                    */
                    bool need_alloc = true;
                    if(m_nMaxCache != -1){
                        if(m_vpFrame.size() >= m_nMaxCache){
                            --m_nDecodedFrame;
                            need_alloc = false;
                        }
                    }

                    if(need_alloc){
                        uint8_t *pFrame = nullptr;
                        if (m_bUseDeviceFrame)
                            //checkCudaDriver(cuMemAlloc((CUdeviceptr *)&pFrame, get_frame_bytes()));
                            checkCudaRuntime(cudaMalloc(&pFrame, get_frame_bytes()));
                        else
                            checkCudaRuntime(cudaMallocHost(&pFrame, get_frame_bytes()));
                            
                        m_vpFrame.push_back(pFrame);
                        m_vTimestamp.push_back(0);
                    }
                }
                pDecodedFrame = m_vpFrame[m_nDecodedFrame - 1];
                m_vTimestamp[m_nDecodedFrame - 1] = pDispInfo->timestamp;
            }

            if(m_output_bgr){
                if(m_pYUVFrame == 0){
                    checkCudaDriver(cuMemAlloc(&m_pYUVFrame, m_nWidth * (m_nLumaHeight + m_nChromaHeight * m_nNumChromaPlanes) * m_nBPP));
                }
                if(m_pBGRFrame == 0){
                    checkCudaDriver(cuMemAlloc(&m_pBGRFrame, m_nWidth * m_nLumaHeight * 3));
                }
                CUDA_MEMCPY2D m = { 0 };
                m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
                m.srcDevice = dpSrcFrame;
                m.srcPitch = nSrcPitch; 
                m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
                m.dstDevice = (CUdeviceptr)(m.dstHost = (uint8_t*)m_pYUVFrame);
                m.dstPitch = m_nWidth * m_nBPP;
                m.WidthInBytes = m_nWidth * m_nBPP;
                m.Height = m_nLumaHeight;
                checkCudaDriver(cuMemcpy2DAsync(&m, m_cuvidStream));

                m.srcDevice = (CUdeviceptr)((uint8_t *)dpSrcFrame + m.srcPitch * m_nSurfaceHeight);
                m.dstDevice = (CUdeviceptr)(m.dstHost = (uint8_t*)m_pYUVFrame + m.dstPitch * m_nLumaHeight);
                m.Height = m_nChromaHeight;
                checkCudaDriver(cuMemcpy2DAsync(&m, m_cuvidStream));

                uint8_t* y  = (uint8_t*)m_pYUVFrame;
                uint8_t* uv = y + m_nWidth * m_nLumaHeight;
                convert_nv12_to_bgr_invoker(y, uv, m_nWidth, m_nLumaHeight, m_nWidth, (uint8_t*)m_pBGRFrame, m_cuvidStream);

                if(m_bUseDeviceFrame){
                    checkCudaDriver(cuMemcpyDtoDAsync((CUdeviceptr)pDecodedFrame, m_pBGRFrame, m_nWidth * m_nLumaHeight * 3, m_cuvidStream));
                }else{
                    checkCudaDriver(cuMemcpyDtoHAsync(pDecodedFrame, m_pBGRFrame, m_nWidth * m_nLumaHeight * 3, m_cuvidStream));
                }
            }else{
                CUDA_MEMCPY2D m = { 0 };
                m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
                m.srcDevice = dpSrcFrame;
                m.srcPitch = nSrcPitch; 
                m.dstMemoryType = m_bUseDeviceFrame ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST;
                m.dstDevice = (CUdeviceptr)(m.dstHost = pDecodedFrame);
                m.dstPitch = m_nWidth * m_nBPP;
                m.WidthInBytes = m_nWidth * m_nBPP;
                m.Height = m_nLumaHeight;
                checkCudaDriver(cuMemcpy2DAsync(&m, m_cuvidStream));

                m.srcDevice = (CUdeviceptr)((uint8_t *)dpSrcFrame + m.srcPitch * m_nSurfaceHeight);
                m.dstDevice = (CUdeviceptr)(m.dstHost = pDecodedFrame + m.dstPitch * m_nLumaHeight);
                m.Height = m_nChromaHeight;
                checkCudaDriver(cuMemcpy2DAsync(&m, m_cuvidStream));

                if (m_nNumChromaPlanes == 2){
                    m.srcDevice = (CUdeviceptr)((uint8_t *)dpSrcFrame + m.srcPitch * m_nSurfaceHeight * 2);
                    m.dstDevice = (CUdeviceptr)(m.dstHost = pDecodedFrame + m.dstPitch * m_nLumaHeight * 2);
                    m.Height = m_nChromaHeight;
                    checkCudaDriver(cuMemcpy2DAsync(&m, m_cuvidStream));
                }
            }
            
            if(!m_bUseDeviceFrame){
                // 确保数据是到位的
                checkCudaDriver(cuStreamSynchronize(m_cuvidStream));
            }
            checkCudaDriver(cuvidUnmapVideoFrame(m_hDecoder, dpSrcFrame));
            return 1;
        }

        virtual ICUStream get_stream() override{
            return m_cuvidStream;
        }

        int get_frame_bytes() override { 
            assert(m_nWidth); 
            if(m_output_bgr){
                return m_nWidth * m_nLumaHeight * 3; 
            }
            return m_nWidth * (m_nLumaHeight + m_nChromaHeight * m_nNumChromaPlanes) * m_nBPP; 
        }

        int get_width() override { assert(m_nWidth); return m_nWidth; }

        int get_height() override { assert(m_nLumaHeight); return m_nLumaHeight; }

        unsigned int get_frame_index() override { return m_iFrameIndex; }

        unsigned int get_num_decoded_frame() override {return m_nDecodedFrame;}

        cudaVideoSurfaceFormat get_output_format() { return m_eOutputFormat; }

        uint8_t* get_frame(int64_t* pTimestamp = nullptr, unsigned int* pFrameIndex = nullptr) override{
            if (m_nDecodedFrame > 0){
                if (pFrameIndex)
                    *pFrameIndex = m_iFrameIndex;

                if (pTimestamp)
                    *pTimestamp = m_vTimestamp[m_nDecodedFrameReturned];

                m_nDecodedFrame--;
                m_iFrameIndex++;
                return m_vpFrame[m_nDecodedFrameReturned++];
            }
            return nullptr;
        }

        virtual ~CUVIDDecoderImpl(){
            
            if (m_hParser) 
                cuvidDestroyVideoParser(m_hParser);

            if (m_hDecoder) 
                cuvidDestroyDecoder(m_hDecoder);

            for (uint8_t *pFrame : m_vpFrame){
                if (m_bUseDeviceFrame)
                    //cuMemFree((CUdeviceptr)pFrame);
                    cudaFree(pFrame);
                else
                    cudaFreeHost(pFrame);
            }

            if(m_pYUVFrame){
                cuMemFree((CUdeviceptr)m_pYUVFrame);
                m_pYUVFrame = 0;
            }

            if(m_pBGRFrame){
                cuMemFree((CUdeviceptr)m_pBGRFrame);
                m_pBGRFrame = 0;
            }
            
            //2023-05-08 释放cudastream Mike
            if (m_cuvidStream) {cudaStreamDestroy(m_cuvidStream);}
            
            cuvidCtxLockDestroy(m_ctxLock);
        }

    private:
        CUvideoctxlock m_ctxLock = nullptr;
        CUvideoparser m_hParser = nullptr;
        CUvideodecoder m_hDecoder = nullptr;
        bool m_bUseDeviceFrame = false;
        // dimension of the output
        unsigned int m_nWidth = 0, m_nLumaHeight = 0, m_nChromaHeight = 0;
        unsigned int m_nNumChromaPlanes = 0;
        // height of the mapped surface 
        int m_nSurfaceHeight = 0;
        int m_nSurfaceWidth = 0;
        cudaVideoCodec m_eCodec = cudaVideoCodec_NumCodecs;
        cudaVideoChromaFormat m_eChromaFormat;
        cudaVideoSurfaceFormat m_eOutputFormat;
        int m_nBitDepthMinus8 = 0;
        int m_nBPP = 1;
        CUVIDEOFORMAT m_videoFormat = {};
        CropRect m_displayRect = {};
        mutex m_lock;
        // stock of frames
        std::vector<uint8_t *> m_vpFrame;
        CUdeviceptr m_pYUVFrame = 0;
        CUdeviceptr m_pBGRFrame = 0;
        // timestamps of decoded frames
        std::vector<int64_t> m_vTimestamp;
        int m_nDecodedFrame = 0, m_nDecodedFrameReturned = 0;
        int m_nDecodePicCnt = 0, m_nPicNumInDecodeOrder[32];
        CUstream m_cuvidStream = 0;
        CropRect m_cropRect = {};
        ResizeDim m_resizeDim = {};
        unsigned int m_iFrameIndex = 0;
        int m_nMaxCache = -1;
        int m_gpuID = -1;
        unsigned int m_nMaxWidth = 0, m_nMaxHeight = 0;
        bool m_output_bgr = true;
    };

    std::shared_ptr<CUVIDDecoder> create_cuvid_decoder(
        bool bUseDeviceFrame, IcudaVideoCodec eCodec, int max_cache, int gpu_id,
        const CropRect *pCropRect, const ResizeDim *pResizeDim, bool output_bgr){

        shared_ptr<CUVIDDecoderImpl> instance(new CUVIDDecoderImpl());
        if(!instance->create(bUseDeviceFrame, gpu_id, (cudaVideoCodec)eCodec, false, pCropRect, pResizeDim, max_cache, 0, 0, 1000, output_bgr))
            instance.reset();
        return instance;
    }
}; //FFHDDecoder
