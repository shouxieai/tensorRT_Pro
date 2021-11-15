
#include "binary_io.hpp"
#include "ilogger.hpp"
#include <string.h>

using namespace std;

BinaryIO::~BinaryIO(){
	close();
}

bool BinaryIO::opened(){
	if (flag_ == MemoryRead)
		return memoryRead_ != nullptr;
	else if (flag_ == MemoryWrite)
		return true;
	return false;
}

void BinaryIO::close(){
	if (flag_ == MemoryRead) {
		memoryRead_ = nullptr;
		memoryCursor_ = 0;
		memoryLength_ = -1;
	}
	else if (flag_ == MemoryWrite) {
		memoryWrite_.clear();
		memoryCursor_ = 0;
		memoryLength_ = -1;
	}
}

string BinaryIO::readData(int numBytes){
	string output;
	output.resize(numBytes);

	int readlen = read((void*)output.data(), output.size());
	output.resize(readlen);
	return output;
}

int BinaryIO::read(void* pdata, size_t length){

	if (flag_ == MemoryRead) {
		if (memoryLength_ != -1) {
			
			if (memoryLength_ < memoryCursor_ + length) {
				int remain = memoryLength_ - memoryCursor_;
				if (remain > 0) {
					memcpy(pdata, memoryRead_ + memoryCursor_, remain);
					memoryCursor_ += remain;
					return remain;
				}
				else {
					return -1;
				}
			}
		}
		memcpy(pdata, memoryRead_ + memoryCursor_, length);
		memoryCursor_ += length;
		return length;
	}
	else {
		return -1;
	}
}

bool BinaryIO::eof(){
	if (!opened()) return true;

	if (flag_ == MemoryRead){
		return this->memoryCursor_ >= this->memoryLength_;
	}
	else if (flag_ == MemoryWrite){
		return false;
	}
	else {
		opstate_ = false;
		INFO("Unsupport flag: %d", flag_);
		return true;
	}
}

int BinaryIO::write(const void* pdata, size_t length){

	if (flag_ == MemoryWrite) {
		memoryWrite_.append((char*)pdata, (char*)pdata + length);
		return length;
	}
	else {
		return -1;
	}
}

int BinaryIO::writeData(const string& data){
	return write(data.data(), data.size());
}

BinaryIO& BinaryIO::operator >> (string& value){
	//read
	int length = 0;
	(*this) >> length;
	value = readData(length);
	return *this;
}

int BinaryIO::readInt(){
	int value = 0;
	(*this) >> value;
	return value;
}

float BinaryIO::readFloat(){
	float value = 0;
	(*this) >> value;
	return value;
}

BinaryIO& BinaryIO::operator << (const string& value){
	//write
	(*this) << (int)value.size();
	writeData(value);
	return *this;
}

BinaryIO& BinaryIO::operator << (const char* value){

	int length = strlen(value);
	(*this) << (int)length;
	write(value, length);
	return *this;
}

BinaryIO& BinaryIO::operator << (const vector<string>& value){
	(*this) << (int)value.size();
	for (int i = 0; i < value.size(); ++i){
		(*this) << value[i];
	}
	return *this;
}

BinaryIO& BinaryIO::operator >> (vector<string>& value){
	int num;
	(*this) >> num;

	value.resize(num);
	for (int i = 0; i < value.size(); ++i)
		(*this) >> value[i];
	return *this;
}

bool BinaryIO::openMemoryRead(const void* ptr, int memoryLength) {
	close();

	if (!ptr) return false;
	memoryRead_ = (const char*)ptr;
	memoryCursor_ = 0;
	memoryLength_ = memoryLength;
	flag_ = MemoryRead;
	return true;
}

void BinaryIO::openMemoryWrite() {
	close();

	memoryWrite_.clear();
	memoryCursor_ = 0;
	memoryLength_ = -1;
	flag_ = MemoryWrite;
}