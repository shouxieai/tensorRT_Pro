/*
 这个文件在windows下有效，linux下请忽略
*/

#if defined(_WIN32)
#	define U_OS_WINDOWS
#else
#   define U_OS_LINUX
#endif

#ifdef U_OS_WINDOWS
#if defined(_DEBUG)
#	pragma comment(lib, "opencv_world346d.lib")
#else
#	pragma comment(lib, "opencv_world346.lib")
#endif

//导入cuda
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "cudnn.lib")

//导入tensorRT
#pragma comment(lib, "nvinfer.lib")
#pragma comment(lib, "nvinfer_plugin.lib")
#pragma comment(lib, "nvonnxparser.lib")

#endif // U_OS_WINDOWS