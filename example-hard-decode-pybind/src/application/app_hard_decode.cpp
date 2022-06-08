
// #include <opencv2/opencv.hpp>
// #include <common/ilogger.hpp>
// #include <ffhdd/ffmpeg_demuxer.hpp>
// #include <ffhdd/cuvid_decoder.hpp>
// #include <ffhdd/nalu.hpp>

// using namespace std;

// static void test_hard_decode(){

//     auto demuxer = FFHDDemuxer::create_ffmpeg_demuxer("exp/number100.mp4");
//     if(demuxer == nullptr){
//         INFOE("demuxer create failed");
//         return;
//     }

//     auto decoder = FFHDDecoder::create_cuvid_decoder(
//         false, FFHDDecoder::ffmpeg2NvCodecId(demuxer->get_video_codec()), -1, 0
//     );

//     if(decoder == nullptr){
//         INFOE("decoder create failed");
//         return;
//     }

//     uint8_t* packet_data = nullptr;
//     int packet_size = 0;
//     int64_t pts = 0;

//     demuxer->get_extra_data(&packet_data, &packet_size);
//     decoder->decode(packet_data, packet_size);

//     iLogger::rmtree("imgs");
//     iLogger::mkdir("imgs");

//     do{
//         demuxer->demux(&packet_data, &packet_size, &pts);
//         int ndecoded_frame = decoder->decode(packet_data, packet_size, pts);
//         for(int i = 0; i < ndecoded_frame; ++i){
//             unsigned int frame_index = 0;

//             /* 因为decoder获取的frame内存，是YUV-NV12格式的。储存内存大小是 [height * 1.5] * width byte
//              因此构造一个height * 1.5,  width 大小的空间
//              然后由opencv函数，把YUV-NV12转换到BGR，转换后的image则是正常的height, width, CV_8UC3
//             */
//             cv::Mat image(decoder->get_height() * 1.5, decoder->get_width(), CV_8U, decoder->get_frame(&pts, &frame_index));
//             cv::cvtColor(image, image, cv::COLOR_YUV2BGR_NV12);

//             frame_index = frame_index + 1;
//             INFO("write imgs/img_%05d.jpg  %dx%d", frame_index, image.cols, image.rows);
//             cv::imwrite(cv::format("imgs/img_%05d.jpg", frame_index), image);
//         }
//     }while(packet_size > 0);
// }

// int app_hard_decode(){

//     test_hard_decode();
//     return 0;
// }