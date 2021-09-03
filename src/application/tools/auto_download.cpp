
#include <stdlib.h>
#include <common/ilogger.hpp>

using namespace std;

bool requires(const char* name) {

	auto onnx_file = iLogger::format("%s.onnx", name);
	if (not iLogger::exists(onnx_file)) {
		INFO("Auto download %s", onnx_file.c_str());
		system(iLogger::format("wget http://zifuture.com:1556/fs/25.shared/%s", onnx_file.c_str()).c_str());
	}

	bool exists = iLogger::exists(onnx_file);
	if (not exists) {
		INFOE("Download %s failed", onnx_file.c_str());
	}
	return exists;
}