
#ifndef ILOGGER_HPP
#define ILOGGER_HPP


#include <string>
#include <vector>
#include <stdio.h>

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

    std::string date_now();
    std::string time_now();
    bool mkdir(const std::string& path);
    bool mkdirs(const std::string& path);
    FILE* fopen_mkdirs(const std::string& path, const std::string& mode);
    bool exists(const std::string& path);
    std::string format(const char* fmt, ...);
    std::string file_name(const std::string& path, bool include_suffix);
    long long timestamp_now();
    std::vector<uint8_t> load_file(const std::string& file);

    // h[0-1], s[0-1], v[0-1]
    // return, 0-255, 0-255, 0-255
    std::tuple<uint8_t, uint8_t, uint8_t> hsv2rgb(float h, float s, float v);
    std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id);

    //   abcdefg.pnga          *.png      > false
	//   abcdefg.png           *.png      > true
	//   abcdefg.png          a?cdefg.png > true
	bool pattern_match(const char* str, const char* matcher, bool igrnoe_case = true);
    std::vector<std::string> find_files(
        const std::string& directory, 
        const std::string& filter = "*", bool findDirectory = false, bool includeSubDirectory = false);

    std::string align_blank(const std::string& input, int align_size, char blank=' ');
    bool save_file(const std::string& file, const std::vector<uint8_t>& data, bool mk_dirs = true);
	bool save_file(const std::string& file, const void* data, size_t length, bool mk_dirs = true);

    // 关于logger的api
    const char* level_string(int level);
    void set_save_directory(const std::string& loggerDirectory);

    // 当日志的级别低于这个设置时，会打印出来，否则会直接跳过
    void set_log_level(int level);
    void __log_func(const char* file, int line, int level, const char* fmt, ...);
    void destroy();
};


#endif // ILOGGER_HPP