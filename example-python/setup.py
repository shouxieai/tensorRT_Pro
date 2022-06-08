
from setuptools import find_packages
from setuptools import setup
import platform
import os

os_name = platform.system()
if os_name == "Linux":
    cpp_library = ["libpytrtc.so", "libplugin_list.so"]
elif os_name == "Windows":
    os.system("copy_dll_to_pytrt.bat")
    cpp_library = [
        "libpytrtc.pyd",
        "cublas64_10.dll", 
        "cublas64_100.dll", 
        "cublasLt64_10.dll", 
        "cudart64_101.dll", 
        "cudnn64_8.dll", 
        "cudnn_adv_infer64_8.dll", 
        "cudnn_adv_train64_8.dll", 
        "cudnn_cnn_infer64_8.dll", 
        "cudnn_cnn_train64_8.dll", 
        "cudnn_ops_infer64_8.dll", 
        "cudnn_ops_train64_8.dll", 
        "nvinfer.dll", 
        "nvinfer_plugin.dll", 
        "opencv_world346.dll"
    ]
    
else:
    raise RuntimeError(f"Unsupport platform {os_name}")

setup(
    name="pytrt",
    version="1.0",
    author="Wish",
    url="https://github.com/shouxieai/tensorRT_cpp",
    description="tensorRT CPP/Python",
    packages=find_packages(),
    package_data={
        "": cpp_library
    },
    zip_safe=False
)