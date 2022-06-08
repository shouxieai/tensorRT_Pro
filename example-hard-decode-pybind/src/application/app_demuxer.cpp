
// #include <common/ilogger.hpp>
// #include <ffhdd/ffmpeg_demuxer.hpp>
// #include <ffhdd/nalu.hpp>

// using namespace std;

// static void test_demuxer(){

//     auto demuxer = FFHDDemuxer::create_ffmpeg_demuxer("exp/fall_video.mp4");
//     if(demuxer == nullptr){
//         INFOE("demuxer create failed");
//         return;
//     }

//     INFO("demuxer create done.");

//     uint8_t* packet_data = nullptr;
//     int packet_size = 0;
//     int64_t pts = 0;

//     demuxer->get_extra_data(&packet_data, &packet_size);

//     vector<uint8_t> extra_data(packet_size + 3);
//     memcpy(extra_data.data() + 3, packet_data, packet_size);

//     int ipacket = 0;
//     auto frame_type = NALU::format_nalu_type(NALU::find_all_nalu_info(extra_data.data(), packet_size, 0));
//     INFO("Extra Data size: %d, type: %s", packet_size, frame_type.c_str());

//     do{
//         demuxer->demux(&packet_data, &packet_size, &pts);

//         frame_type = "Empty";
//         if(packet_size > 0){
//             frame_type = NALU::format_nalu_frame_type(NALU::find_all_nalu_info(packet_data, packet_size, 0));
//         }

//         INFO("Packet %d NALU size: %d, pts = %lld, type = %s", 
//             ipacket++,
//             packet_size, 
//             pts, 
//             frame_type.c_str()
//         );

//     }while(packet_size > 0);
// }

// /*
//     一个GOP，就是一个group，有N个frame
//     N又 = I + B/P * M         M = N - 1

//     GOP是H264的最小单元。要是理解了这些。就可以轻易操作H264分为一段一段的。储存也好，解码也好。都会很容易 
//  */
// int app_demuxer(){

//     test_demuxer();
//     //INFO("%s", NALU::slice_type_string(NALU::get_slice_type_from_slice_header(0x00D8E002)));
//     return 0;
// }