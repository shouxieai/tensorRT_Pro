#ifndef SIMPLE_LOGGER_HPP
#define SIMPLE_LOGGER_HPP

#include <stdio.h>

#define INFOD(...)			SimpleLogger::__log_func(__FILE__, __LINE__, SimpleLogger::LogLevel::Debug, __VA_ARGS__)
#define INFOV(...)			SimpleLogger::__log_func(__FILE__, __LINE__, SimpleLogger::LogLevel::Verbose, __VA_ARGS__)
#define INFO(...)			SimpleLogger::__log_func(__FILE__, __LINE__, SimpleLogger::LogLevel::Info, __VA_ARGS__)
#define INFOW(...)			SimpleLogger::__log_func(__FILE__, __LINE__, SimpleLogger::LogLevel::Warning, __VA_ARGS__)
//#define INFOE(...)			SimpleLogger::__log_func(__FILE__, __LINE__, SimpleLogger::LogLevel::Error, __VA_ARGS__)
#define INFOE(...)			SimpleLogger::__log_func(__FILE__, __LINE__, SimpleLogger::LogLevel::Info, __VA_ARGS__) //修改为info，使用infoe python层会crash MIke 2020--5-11
#define INFOF(...)			SimpleLogger::__log_func(__FILE__, __LINE__, SimpleLogger::LogLevel::Fatal, __VA_ARGS__)


namespace SimpleLogger{

    enum class LogLevel : int{
        Debug   = 5,
        Verbose = 4,
        Info    = 3,
        Warning = 2,
        Error   = 1,
        Fatal   = 0
    };

    void set_log_level(LogLevel level);
    LogLevel get_log_level();
    void __log_func(const char* file, int line, LogLevel level, const char* fmt, ...);

};  // SimpleLogger

#endif // SIMPLE_LOGGER_HPP
