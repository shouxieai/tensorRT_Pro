
#include "simple-logger.hpp"
#include <string>
#include <stdarg.h>

using namespace std;

namespace SimpleLogger{

    static LogLevel g_level = LogLevel::Info;

    const char* level_string(LogLevel level){
        switch (level){
            case LogLevel::Debug: return "debug";
            case LogLevel::Verbose: return "verbo";
            case LogLevel::Info: return "info";
            case LogLevel::Warning: return "warn";
            case LogLevel::Error: return "error";
            case LogLevel::Fatal: return "fatal";
            default: return "unknow";
        }
    }

    void set_log_level(LogLevel level){
        g_level = level;
    }

    LogLevel get_log_level(){
        return g_level;
    }

    string file_name(const string& path, bool include_suffix){

        if (path.empty()) return "";

        int p = path.rfind('/');
        p += 1;

        //include suffix
        if (include_suffix)
            return path.substr(p);

        int u = path.rfind('.');
        if (u == -1)
            return path.substr(p);

        if (u <= p) u = path.size();
        return path.substr(p, u - p);
    }

    string time_now(){
        char time_string[20];
        time_t timep;							
        time(&timep);							
        tm& t = *(tm*)localtime(&timep);

        sprintf(time_string, "%04d-%02d-%02d %02d:%02d:%02d", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec);
        return time_string;
    }

    void __log_func(const char* file, int line, LogLevel level, const char* fmt, ...){
        if(level > g_level) return;

        va_list vl;
        va_start(vl, fmt);
        
        char buffer[2048];
        auto now = time_now();
        string filename = file_name(file, true);
        int n = snprintf(buffer, sizeof(buffer), "[%s]", now.c_str());

        if (level == LogLevel::Fatal or level == LogLevel::Error) {
            n += snprintf(buffer + n, sizeof(buffer) - n, "[\033[31m%s\033[0m]", level_string(level));
        }
        else if (level == LogLevel::Warning) {
            n += snprintf(buffer + n, sizeof(buffer) - n, "[\033[33m%s\033[0m]", level_string(level));
        }
        else if (level == LogLevel::Info) {
            n += snprintf(buffer + n, sizeof(buffer) - n, "[\033[35m%s\033[0m]", level_string(level));
        }
        else if (level == LogLevel::Verbose) {
            n += snprintf(buffer + n, sizeof(buffer) - n, "[\033[34m%s\033[0m]", level_string(level));
        }
        else {
            n += snprintf(buffer + n, sizeof(buffer) - n, "[%s]", level_string(level));
        }

        n += snprintf(buffer + n, sizeof(buffer) - n, "[%s:%d]:", filename.c_str(), line);
        vsnprintf(buffer + n, sizeof(buffer) - n, fmt, vl);
        fprintf(stdout, "%s\n", buffer);

        if(level == LogLevel::Fatal || level == LogLevel::Error){
            fflush(stdout);
            abort();
        }
    }
};