
#ifndef ILOGGER_HPP
#define ILOGGER_HPP


#include <string>
#include <vector>
#include <tuple>
#include <time.h>

#define ILOGGER_VERBOSE				4
#define ILOGGER_INFO				3
#define ILOGGER_WARNING			    2
#define ILOGGER_ERROR				1
#define ILOGGER_FATAL				0
#define INFOV(...)			iLogger::__log_func(__FILE__, __LINE__, ILOGGER_VERBOSE, __VA_ARGS__)
#define INFO(...)			iLogger::__log_func(__FILE__, __LINE__, ILOGGER_INFO, __VA_ARGS__)
#define INFOW(...)			iLogger::__log_func(__FILE__, __LINE__, ILOGGER_WARNING, __VA_ARGS__)
#define INFOE(...)			iLogger::__log_func(__FILE__, __LINE__, ILOGGER_ERROR, __VA_ARGS__)
#define INFOF(...)			iLogger::__log_func(__FILE__, __LINE__, ILOGGER_FATAL, __VA_ARGS__)

namespace iLogger{

    using namespace std;

    string date_now();
    string time_now();
    string gmtime_now();
	string gmtime(time_t t);
	time_t gmtime2ctime(const string& gmt);
    void sleep(int ms);

    bool isfile(const string& file);
    bool mkdir(const string& path);
    bool mkdirs(const string& path);
    bool exists(const string& path);
    string format(const char* fmt, ...);
    FILE* fopen_mkdirs(const string& path, const string& mode);
    string file_name(const string& path, bool include_suffix);
    string directory(const string& path);
    long long timestamp_now();
    time_t last_modify(const string& file);
    vector<uint8_t> load_file(const string& file);
    string load_text_file(const string& file);
    size_t file_size(const string& file);

    bool begin_with(const string& str, const string& with);
	bool end_with(const string& str, const string& with);
    vector<string> split_string(const string& str, const std::string& spstr);
    string replace_string(const string& str, const string& token, const string& value);

    // h[0-1], s[0-1], v[0-1]
    // return, 0-255, 0-255, 0-255
    tuple<uint8_t, uint8_t, uint8_t> hsv2rgb(float h, float s, float v);
    tuple<uint8_t, uint8_t, uint8_t> random_color(int id);

    //   abcdefg.pnga          *.png      > false
	//   abcdefg.png           *.png      > true
	//   abcdefg.png          a?cdefg.png > true
	bool pattern_match(const char* str, const char* matcher, bool igrnoe_case = true);
    vector<string> find_files(
        const string& directory, 
        const string& filter = "*", bool findDirectory = false, bool includeSubDirectory = false);

    string align_blank(const string& input, int align_size, char blank=' ');
    bool save_file(const string& file, const vector<uint8_t>& data, bool mk_dirs = true);
    bool save_file(const string& file, const string& data, bool mk_dirs = true);
	bool save_file(const string& file, const void* data, size_t length, bool mk_dirs = true);

    // 循环等待，并捕获例如ctrl+c等终止信号，收到信号后循环跳出并返回信号类型
    // 捕获：SIGINT(2)、SIGQUIT(3)
    int while_loop();

    // 关于logger的api
    const char* level_string(int level);
    void set_logger_save_directory(const string& loggerDirectory);

    // 当日志的级别低于这个设置时，会打印出来，否则会直接跳过
    void set_log_level(int level);
    void __log_func(const char* file, int line, int level, const char* fmt, ...);
    void destroy_logger();

    vector<uint8_t> base64_decode(const string& base64);
    string base64_encode(const void* data, size_t size);
};


#endif // ILOGGER_HPP