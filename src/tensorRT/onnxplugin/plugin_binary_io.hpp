#ifndef PLUGIN_BINARY_IO_HPP
#define PLUGIN_BINARY_IO_HPP

#include <string>
#include <vector>

namespace Plugin{

    class BinIO {
    public:
        enum Head {
            MemoryRead = 1,
            MemoryWrite = 2
        };

        BinIO() { openMemoryWrite(); }
        BinIO(const void* ptr, int memoryLength = -1) { openMemoryRead(ptr, memoryLength); }
        virtual ~BinIO();
        bool opened();
        bool openMemoryRead(const void* ptr, int memoryLength = -1);
        void openMemoryWrite();
        const std::string& writedMemory() { return memoryWrite_; }
        void close();
        int write(const void* pdata, size_t length);
        int writeData(const std::string& data);
        int read(void* pdata, size_t length);
        std::string readData(int numBytes);
        int readInt();
        float readFloat();
        bool eof();

        BinIO& operator >> (std::string& value);
        BinIO& operator << (const std::string& value);
        BinIO& operator << (const char* value);
        BinIO& operator << (const std::vector<std::string>& value);
        BinIO& operator >> (std::vector<std::string>& value);

        template<typename _T>
        BinIO& operator >> (std::vector<_T>& value) {
            int length = 0;
            (*this) >> length;

            value.resize(length);
            read(value.data(), length * sizeof(_T));
            return *this;
        }

        template<typename _T>
        BinIO& operator << (const std::vector<_T>& value) {
            (*this) << (int)value.size();
            write(value.data(), sizeof(_T) * value.size());
            return *this;
        }

        template<typename _T>
        BinIO& operator >> (_T& value) {
            read(&value, sizeof(_T));
            return *this;
        }

        template<typename _T>
        BinIO& operator << (const _T& value) {
            write(&value, sizeof(_T));
            return *this;
        }

        bool opstate() const {
            return opstate_;
        }

    private:
        size_t readModeEndSEEK_ = 0;
        std::string memoryWrite_;
        const char* memoryRead_ = nullptr;
        int memoryCursor_ = 0;
        int memoryLength_ = -1;
        Head flag_ = MemoryWrite;
        bool opstate_ = true;
    };
}; // namespace Plugin

#endif //PLUGIN_BINARY_IO_HPP