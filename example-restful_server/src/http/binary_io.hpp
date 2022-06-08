#ifndef BINARY_IO_HPP
#define BINARY_IO_HPP

#include <string>
#include <vector>

class BinaryIO {
public:
    enum Head {
        MemoryRead = 1,
        MemoryWrite = 2
    };

    BinaryIO() { openMemoryWrite(); }
    BinaryIO(const void* ptr, int memoryLength = -1) { openMemoryRead(ptr, memoryLength); }
    virtual ~BinaryIO();
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

    BinaryIO& operator >> (std::string& value);
    BinaryIO& operator << (const std::string& value);
    BinaryIO& operator << (const char* value);
    BinaryIO& operator << (const std::vector<std::string>& value);
    BinaryIO& operator >> (std::vector<std::string>& value);

    template<typename _T>
    BinaryIO& operator >> (std::vector<_T>& value) {
        int length = 0;
        (*this) >> length;

        value.resize(length);
        read(value.data(), length * sizeof(_T));
        return *this;
    }

    template<typename _T>
    BinaryIO& operator << (const std::vector<_T>& value) {
        (*this) << (int)value.size();
        write(value.data(), sizeof(_T) * value.size());
        return *this;
    }

    template<typename _T>
    BinaryIO& operator >> (_T& value) {
        read(&value, sizeof(_T));
        return *this;
    }

    template<typename _T>
    BinaryIO& operator << (const _T& value) {
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

#endif //BINARY_IO_HPP