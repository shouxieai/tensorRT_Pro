@echo off
copy ..\lean\cuda10.1\bin\cublas64_100.dll .\pytrt\
copy ..\lean\cuda10.1\bin\cublasLt64_10.dll .\pytrt\
copy ..\lean\cuda10.1\bin\cudart64_101.dll .\pytrt\
copy ..\lean\cuda10.1\bin\cublas64_10.dll .\pytrt\

copy ..\lean\opencv3.4.6\lib\opencv_world346.dll .\pytrt\
copy ..\lean\TensorRT-8.0.1.6\lib\nvinfer.dll .\pytrt\
copy ..\lean\TensorRT-8.0.1.6\lib\nvinfer_plugin.dll .\pytrt\
copy ..\lean\cudnn8.2.2.26\*.dll .\pytrt\

