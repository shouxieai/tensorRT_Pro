@echo off
copy ..\lean\cuda10.1\bin\cublas64_100.dll .\trtpy\
copy ..\lean\cuda10.1\bin\cublasLt64_10.dll .\trtpy\
copy ..\lean\cuda10.1\bin\cudart64_101.dll .\trtpy\
copy ..\lean\cuda10.1\bin\cublas64_10.dll .\trtpy\

copy ..\lean\opencv3.4.6\lib\opencv_world346.dll .\trtpy\
copy ..\lean\TensorRT-8.0.1.6\lib\nvinfer.dll .\trtpy\
copy ..\lean\TensorRT-8.0.1.6\lib\nvinfer_plugin.dll .\trtpy\
copy ..\lean\cudnn8.2.2.26\*.dll .\trtpy\

