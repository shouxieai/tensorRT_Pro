
#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include <unordered_map>
#include <string>
#include <vector>

using namespace std;

#ifdef U_OS_WINDOWS
const char* class_label[] = {
    "finance", // finance
    "realty", // realty
    "stocks", // stocks
    "education", // education
    "science", // science
    "society", // society
    "politics", // politics
    "sports", // sports
    "game", // game
    "entertainment"  // entertainment
};
#else
/* 如果在windows下编译报错，请屏蔽掉这个文件 */
const char* class_label[] = {
    "金融", // finance
    "房地产", // realty
    "股票", // stocks
    "教育", // education
    "科学", // science
    "社会", // society
    "政治", // politics
    "体育", // sports
    "游戏", // game
    "娱乐"  // entertainment
};
#endif

bool requires(const char* name);


unordered_map<string, int> load_vocab(const string& file){

    unordered_map<string, int> vocab;
    auto lines = iLogger::split_string(iLogger::load_text_file(file), "\n");
    for(int i = 0; i < lines.size(); ++i){
        auto token = lines[i];
        vocab[token] = i;
    }
    return vocab;
}

int find_token(const string& token, const unordered_map<string, int>& vocab){
    auto iter = vocab.find(token);
    if(iter == vocab.end())
        return -1;
    return iter->second;
}

/* utf-8
  拆分utf8的汉字，把汉字部分独立，ascii部分连续为一个
  for example:
    你jok我good呀  -> ["你", "job", "我", "good", "呀"] */
vector<string> split_chinese(const string& text){

    // 1字节：0xxxxxxx 
    // 2字节：110xxxxx 10xxxxxx 
    // 3字节：1110xxxx 10xxxxxx 10xxxxxx 
    // 4字节：11110xxx 10xxxxxx 10xxxxxx 10xxxxxx 
    // 5字节：111110xx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx
    // 6字节：11111110 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx
    unsigned char* up = (unsigned char*)text.c_str();
    int offset = 0;
    int length = text.size();
    unsigned char lab_char[] = {
    // 11111110  11111000  11110000  11100000  11000000  01111111
        0xFE,    0xF8,     0xF0,     0xE0,     0xC0,     0x80
    };

    int char_size_table[] = {
        6, 5, 4, 3, 2, 0
    };

    vector<string> tokens;
    string ascii;
    int state = 0;   // 0 none,  1 wait ascii
    while(offset < length){
        unsigned char token = up[offset];
        int char_size = 1;
        for(int i = 0; i < 6; ++i){
            if(token >= lab_char[i]){
                char_size = char_size_table[i];
                break;
            }
        }

        if(char_size == 0){
            // invalid char
            offset++;
            continue;
        }

        if(offset + char_size > length)
            break;

        auto char_string = text.substr(offset, char_size);
        if(char_size == 1){
            // ascii 
            if(state == 0){
                ascii = char_string;
                state = 1;
            }else if(state == 1){
                ascii += char_string;
            }
        }else{
            if(state == 1){
                tokens.emplace_back(ascii);
                state = 0;
            }
            tokens.emplace_back(char_string);
        }
        offset += char_size;
    }

    if(state == 1)
        tokens.emplace_back(ascii);

    return tokens;
}

/* 把字符串拆分为词组，汉字单个为一组 */
vector<string> tokenize(const string& text, const unordered_map<string, int>& vocab){

    vector<string> tokens = split_chinese(text);
    vector<string> output;
    int max_input_chars_per_word = 100;
    auto UNK = "[UNK]";

    for(int itoken = 0; itoken < tokens.size(); ++itoken){
        auto& chars = tokens[itoken];
        if(chars.size() > max_input_chars_per_word){
            output.push_back(UNK);
            continue;
        }

        bool is_bad = false;
        int start = 0;
        vector<string> sub_tokens;
        while(start < chars.size()){
            int end = chars.size();
            string cur_substr;
            while(start < end){
                auto substr = chars.substr(start, end-start);
                for(int k = 0; k < substr.size(); ++k){
                    auto& c = substr[k];
                    if(c >= 'A' && c <= 'Z')
                        c = c - 'A' + 'a';
                }

                if(start > 0)
                    substr = "##" + substr;

                auto token_id = find_token(substr, vocab);
                if(token_id != -1){
                    cur_substr = substr;
                    break;
                }
                end -= 1;
            }

            if(cur_substr.empty()){
                is_bad = true;
                break;
            }
            sub_tokens.push_back(cur_substr);
            start = end;
        }

        if(is_bad){
            output.push_back(UNK);
        }else{
            output.insert(output.end(), sub_tokens.begin(), sub_tokens.end());
        }
    }
    return output;
}

vector<int> tokens_to_ids(const vector<string>& tokens, const unordered_map<string, int>& vocab){
    vector<int> output(tokens.size());
    for(int i =0 ; i < tokens.size(); ++i)
        output[i] = find_token(tokens[i], vocab);
    return output;
}

tuple<vector<int>, vector<int>> align_and_pad(
    const vector<string>& tokens, int pad_size, 
    const unordered_map<string, int>& vocab
){
    auto CLS = find_token("[CLS]", vocab);
    vector<int> output = tokens_to_ids(tokens, vocab);
    vector<int> mask(pad_size, 1);
    output.insert(output.begin(), CLS);

    int old_size = output.size();
    output.resize(pad_size);

    if(old_size < pad_size){
        std::fill(output.begin() + old_size, output.end(),   0);
        std::fill(mask.begin()   + old_size, mask.end(),     0);
    }
    return make_tuple(output, mask);
}

tuple<vector<int>, vector<int>> make_text_data(const string& text, const unordered_map<string, int>& vocab){

    auto tokens = tokenize(text, vocab);
    return align_and_pad(tokens, 32, vocab);
}

void softmax(float* ptr, int num){
    float sum = 0;
    for(int i = 0; i < num; ++i)
        sum += exp(ptr[i]);
    
    for(int i = 0; i < num; ++i)
        ptr[i] = exp(ptr[i]) / sum;
}

int app_bert(){

    auto name = "bert";
    if(not requires(name))
        return 0;

    auto onnx_file = iLogger::format("%s.onnx", name);
    auto model_file = iLogger::format("%s.trtmodel", name);
    auto vocab = load_vocab("vocab.txt");

    if(not iLogger::exists(model_file)){
        TRT::compile(
            TRT::Mode::FP32, 1,
            onnx_file, model_file
        );
    }

    auto engine = TRT::load_infer(model_file);
    engine->print();

    string line;
    while(true){
        printf("Input content: ");
        if(getline(cin, line)){
            vector<int> tokens, mask;
            tie(tokens, mask) = make_text_data(line, vocab);

            memcpy(engine->input(0)->cpu<int>(), tokens.data(), sizeof(int) * tokens.size());
            memcpy(engine->input(1)->cpu<int>(), mask.data(), sizeof(int) * mask.size());

            engine->forward();

            auto ptr = engine->output()->cpu<float>();
            int num_classes = engine->output()->size(1);
            softmax(ptr, num_classes);

            int label = std::max_element(ptr, ptr + num_classes) - ptr;
            INFO("Predict: %s, %.3f", class_label[label], ptr[label]);
        }
    }
    return 0;
}