#ifndef NALU_HPP
#define NALU_HPP

#include <vector>
#include <tuple>
#include <string.h>

namespace NALU{

    enum class nal_unit_type_t : unsigned char{
        unuse = 0,
        slice_nonidr_layer_without_partitioning_rbsp = 1,
        slice_data_partition_a_layer_rbsp = 2,
        slice_data_partition_b_layer_rbsp = 3,
        slice_data_partition_c_layer_rbsp = 4,
        slice_idr_layer_without_partitioning_rbsp = 5,
        sei_rbsp = 6,
        seq_parameter_set_rbsp = 7,
        pic_parameter_set_rbsp = 8,
        access_unit_delimiter_rbsp = 9,
        end_of_seq_rbsp = 10,
        end_of_stream_rbsp = 11,
        filler_data_rbsp = 12,
        seq_parameter_set_extension_rbsp = 13,
        reserve = 14, // .. 14..18  reserve
        slice_layer_without_partitioning_rbsp = 19,
        reserve2 = 20, // 20..23  reserve
        unuse2 = 24, // 24..31 unuse
    };

    // 当nal_unit_type_t = slice_idr_layer_without_partitioning_rbsp时
    // slice_type_t必定等于I/SI/EX_I/EX_SI
    // 当slice_type_t的值为5-9时，其值应该减去5
    /* 7.4.3节描述 */
    enum class slice_type_t : unsigned char{
        UNKNOW = 0xFF,
        P = 0,
        B = 1,
        I = 2,
        SP = 3,
        SI = 4,
        EX_P = 5,
        EX_B = 6,
        EX_I = 7,
        EX_SP = 8,
        EX_SI = 9
    };

    struct nal_unit_t{
        nal_unit_type_t nal_unit_type : 5;
        unsigned char nal_ref_idc : 2;
        unsigned char forbidden_zero_bit : 1;
    };

    struct nal_unit_info{
        nal_unit_t head;
        slice_type_t slice_type;
        int offset;
        int flag_size;
    };

    inline const char* nal_unit_type_string(nal_unit_type_t t){
        switch(t){
        case nal_unit_type_t::unuse: return "unuse";
        case nal_unit_type_t::slice_nonidr_layer_without_partitioning_rbsp: return "slice_nonidr_layer_without_partitioning_rbsp";
        case nal_unit_type_t::slice_data_partition_a_layer_rbsp: return "slice_data_partition_a_layer_rbsp";
        case nal_unit_type_t::slice_data_partition_b_layer_rbsp: return "slice_data_partition_b_layer_rbsp";
        case nal_unit_type_t::slice_data_partition_c_layer_rbsp: return "slice_data_partition_c_layer_rbsp";
        case nal_unit_type_t::slice_idr_layer_without_partitioning_rbsp: return "slice_idr_layer_without_partitioning_rbsp";
        case nal_unit_type_t::sei_rbsp: return "sei_rbsp";
        case nal_unit_type_t::seq_parameter_set_rbsp: return "seq_parameter_set_rbsp";
        case nal_unit_type_t::pic_parameter_set_rbsp: return "pic_parameter_set_rbsp";
        case nal_unit_type_t::access_unit_delimiter_rbsp: return "access_unit_delimiter_rbsp";
        case nal_unit_type_t::end_of_seq_rbsp: return "end_of_seq_rbsp";
        case nal_unit_type_t::end_of_stream_rbsp: return "end_of_stream_rbsp";
        case nal_unit_type_t::filler_data_rbsp: return "filler_data_rbsp";
        case nal_unit_type_t::seq_parameter_set_extension_rbsp: return "seq_parameter_set_extension_rbsp";
        case nal_unit_type_t::reserve: return "reserve";
        case nal_unit_type_t::slice_layer_without_partitioning_rbsp: return "slice_layer_without_partitioning_rbsp";
        case nal_unit_type_t::reserve2: return "reserve2"; // 20..23  reserve
        case nal_unit_type_t::unuse2: return "unuse2"; // 24..31 unuse
        default: return "unknow";
        }
    }

    inline const char* nal_unit_type_short_string(nal_unit_type_t t){
        switch(t){
        case nal_unit_type_t::unuse: return "unuse";
        case nal_unit_type_t::slice_nonidr_layer_without_partitioning_rbsp: return "nonidr";
        case nal_unit_type_t::slice_data_partition_a_layer_rbsp: return "slice_a";
        case nal_unit_type_t::slice_data_partition_b_layer_rbsp: return "slice_b";
        case nal_unit_type_t::slice_data_partition_c_layer_rbsp: return "slice_c";
        case nal_unit_type_t::slice_idr_layer_without_partitioning_rbsp: return "idr";
        case nal_unit_type_t::sei_rbsp: return "sei";
        case nal_unit_type_t::seq_parameter_set_rbsp: return "sps";
        case nal_unit_type_t::pic_parameter_set_rbsp: return "pps";
        case nal_unit_type_t::access_unit_delimiter_rbsp: return "aud";
        case nal_unit_type_t::end_of_seq_rbsp: return "eos";
        case nal_unit_type_t::end_of_stream_rbsp: return "eostr";
        case nal_unit_type_t::filler_data_rbsp: return "filter";
        case nal_unit_type_t::seq_parameter_set_extension_rbsp: return "sps_ext";
        case nal_unit_type_t::reserve: return "reserve";
        case nal_unit_type_t::slice_layer_without_partitioning_rbsp: return "slice";
        case nal_unit_type_t::reserve2: return "reserve2"; // 20..23  reserve
        case nal_unit_type_t::unuse2: return "unuse2"; // 24..31 unuse
        default: return "unknow";
        }
    }

    inline const char* slice_type_string(slice_type_t t){
        //SI和SP：即Switch I和Switch P，是一种特殊的编解码条带，可以保证在视频流之间进行有效的切换，并且解码器可以任意的访问。比如，同一个视频源被编码成各种码率的码流，在传输的过程中可以根据网络环境进行实时的切换；
        //SI宏块是一种特殊类型的内部编码宏块，按Intra_4x4预测宏块编码。
        switch(t){
        case slice_type_t::EX_P: return "P";
        case slice_type_t::P: return "P";
        case slice_type_t::EX_B: return "B";
        case slice_type_t::B: return "B";
        case slice_type_t::EX_I: return "I";
        case slice_type_t::I: return "I";
        case slice_type_t::EX_SP: return "SP";
        case slice_type_t::SP: return "SP";
        case slice_type_t::EX_SI: return "SI";
        case slice_type_t::SI: return "SI";
        case slice_type_t::UNKNOW: return "UNKNOW";
        default: return "UNKNOW";
        }
    }

    static slice_type_t get_slice_type_from_slice_header(unsigned char slice_header){

        // slice_header = (
        //     ((slice_header & 0xFF) << 24) |
        //     ((slice_header & 0xFF00) << 8) |
        //     ((slice_header & 0xFF0000) >> 8) |
        //     ((slice_header & 0xFF000000) >> 24)
        // );
        /* 如果图简单，这一句话也可以判断类型，但是这并不靠谱，只能判断I或者P */
        // return (slice_header & 0x40) == 0 ? slice_type_t::I : slice_type_t::P;
        // 01000000  40
        // 11100000  E0
        // 10111000  B8
        // 对于 10111000，按照h264语法 7.3.3，slice_header()
        // first_mb_in_slice = ue(v)
        // slice_type = ue(v)
        // 9.1节 ue(v) 是哥伦布编码格式
        /*
            leadingZeroBits = -1;
            for(b = 0; b == 0; leadingZeroBits++)
                b = read_bits(1);

            codeNumber = pow(2, leadingZeroBits) - 1 + read_bits(leadingZeroBits)
        */
        // 对于10111000而言 = 0xB8
        // first_mb_in_slice = ue(1) = pow(2, 0) - 1 + 0 = 0 属于P帧
        // slice_type = ue(011) = pow(2, 1) - 1 + 1 = 2      属于I帧
        // 读取0的个数为leadingZeroBits，然后跳过第一个1，再读取leadingZeroBits个bit为整数，按照公式计算即可
        int state = 0;  // 0找0统计leadingZeroBits，1read_bits，2计算code_number
        int leading_zero_bits = 0;
        int i_code = 0;

        unsigned int code_number_read_bits = 0;
        unsigned int code_number_exponent = 0;
        for(int bit_index = 0; bit_index < sizeof(slice_header) * 8; ++bit_index){
            unsigned int bit_value = (slice_header >> (sizeof(slice_header) * 8 - bit_index - 1)) & 0x01;

            if(state == 0){
                if(bit_value == 0)
                    leading_zero_bits++;
                else{
                    code_number_exponent = leading_zero_bits;
                    if(leading_zero_bits == 0)
                        state = 2;
                    else
                        state = 1;
                }
            }else if(state == 1){
                code_number_read_bits <<= 1;
                code_number_read_bits |= bit_value;

                if(--leading_zero_bits == 0)
                    state = 2;
            }
            
            if(state == 2){
                unsigned int code_number = (1 << code_number_exponent) - 1 + code_number_read_bits;
                if(i_code == 1){
                    if(code_number >= 5)
                        code_number -= 5;

                    return (slice_type_t)code_number;
                }

                state = 0;
                leading_zero_bits = 0;
                code_number_read_bits = 0;
                i_code++;
            }
        }
        return slice_type_t::UNKNOW;
    }

    /* 在h264_data的内存，以start为起点，查找nalu的头（0x00, 0x00, 0x01 或者 0x00, 0x00, 0x00, 0x01），找到则返回起始位置 */
    static std::tuple<size_t, size_t> find_nalu(const uint8_t* h264_data, size_t end, size_t start = 0){

        const uint8_t* ptr = h264_data;
        uint8_t head2[] = {0x00, 0x00, 0x00, 0x01};

        for(size_t i = start; i < end; ++i){
            if(ptr[i] == 0x00){
                if(end - i >= sizeof(head2)){
                    if(memcmp(ptr + i, head2, sizeof(head2)) == 0)
                        return std::make_tuple(i, 4);
                }
            }
        }
        return std::make_tuple(0, 0);
    }

    static std::vector<nal_unit_info> find_all_nalu_info(const uint8_t* h264_data, size_t end, size_t start = 0){

        int pos = 0, flag_size = 0;
        size_t cursor = start;
        std::vector<nal_unit_info> output;

        do{
            std::tie(pos, flag_size) = find_nalu(h264_data, end, cursor);
            if(flag_size == 0)
                break;
                
            nal_unit_info item;
            memcpy(&item.head, h264_data + pos + flag_size, sizeof(item.head));
            item.flag_size = flag_size;
            item.offset = pos;

            if(item.head.nal_unit_type == nal_unit_type_t::slice_idr_layer_without_partitioning_rbsp ||
               item.head.nal_unit_type == nal_unit_type_t::slice_nonidr_layer_without_partitioning_rbsp){
                item.slice_type = get_slice_type_from_slice_header(*(unsigned char*)(h264_data + pos + flag_size + 1));
            }else{
                item.slice_type = slice_type_t::UNKNOW;
            }
            output.emplace_back(item);
            cursor = pos + flag_size + 1;
        }while(cursor < end);
        return output;
    }

    static std::string format_nalu_frame_type(const std::vector<nal_unit_info>& info_array){

        std::string output;
        for(int i = 0; i < info_array.size(); ++i){
            auto& item = info_array[i];

            if(item.head.nal_unit_type == nal_unit_type_t::slice_idr_layer_without_partitioning_rbsp ||
                item.head.nal_unit_type == nal_unit_type_t::slice_nonidr_layer_without_partitioning_rbsp){
                output += slice_type_string(item.slice_type);
            }else{
                output += nal_unit_type_short_string(item.head.nal_unit_type);
            }

            if(i + 1 < info_array.size())
                output += ",";
        }
        return output;
    }

    static std::string format_nalu_type(const std::vector<nal_unit_info>& info_array){

        std::string output;
        for(int i = 0; i < info_array.size(); ++i){
            auto& item = info_array[i];
            output += nal_unit_type_short_string(item.head.nal_unit_type);

            if(i + 1 < info_array.size())
                output += ",";
        }
        return output;
    }

}; // namespace NALU

#endif NALU_HPP