cmake_minimum_required(VERSION 2.6)
project(pro)
add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)
set(HAS_PYTHON OFF)

# 取决于你的设备，这是编译cu文件的配置，一般可以不用设置，除非你调用了特定函数例如half新特性
# 如果你是不同显卡，请设置为显卡对应的号码
# set(CUDA_GEN_CODE "-gencode=arch=compute_75,code=sm_75")

# 如果你的opencv找不到，可以自己指定目录
set(OpenCV_DIR   "/data/datav/expstation/lean/opencv4.2.0/lib/cmake/opencv4/")

# 建议cuda10.2以上
set(CUDA_DIR     "/usr/local/cuda-10.2")

# 需要与cuda、tensorRT配对
set(CUDNN_DIR    "/data/sxai/lean/cudnn8.2.2.26")

# 注意支持8.0以上，以下不见得支持
set(TENSORRT_DIR "/data/sxai/lean/TensorRT-8.0.1.6")

# 因为protobuf，需要用特定版本，所以这里指定路径
set(PROTOBUF_DIR "/data/sxai/lean/protobuf3.11.4")

# 如果要支持python则设置python路径
set(PythonDIR "/data/datav/newbb/lean/anaconda3/envs/torch1.8")
set(PythonName "python3.9")


find_package(CUDA REQUIRED)
find_package(OpenCV)

include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/application
    ${PROJECT_SOURCE_DIR}/src/tensorRT
    ${PROJECT_SOURCE_DIR}/src/tensorRT/common
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_DIR}/include
    ${PROTOBUF_DIR}/include
    ${TENSORRT_DIR}/include
    ${CUDNN_DIR}/include
)

# 切记，protobuf的lib目录一定要比tensorRT目录前面，因为tensorRTlib下带有protobuf的so文件
# 这可能带来错误
link_directories(
    ${PROTOBUF_DIR}/lib
    ${TENSORRT_DIR}/lib
    ${CUDA_DIR}/lib64
    ${CUDNN_DIR}/lib
)

if("${HAS_PYTHON}" STREQUAL "ON")
    message("Usage Python ${PythonDIR}")
    include_directories(${PythonDIR}/include/${PythonName})
    link_directories(${PythonDIR}/lib)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAS_PYTHON")
endif()

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -pthread -w -g")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")
file(GLOB_RECURSE cpp_srcs ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE cuda_srcs ${PROJECT_SOURCE_DIR}/src/*.cu)
cuda_add_library(plugin_list SHARED ${cuda_srcs})

add_executable(pro ${cpp_srcs})
target_link_libraries(pro nvinfer nvinfer_plugin)
target_link_libraries(pro cuda cublas cudart cudnn)
target_link_libraries(pro protobuf pthread plugin_list)
target_link_libraries(pro ${OpenCV_LIBS})

if("${HAS_PYTHON}" STREQUAL "ON")
    set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/python/trtpy)
    add_library(trtpyc SHARED ${cpp_srcs})
    target_link_libraries(trtpyc nvinfer nvinfer_plugin)
    target_link_libraries(trtpyc cuda cublas cudart cudnn)
    target_link_libraries(trtpyc protobuf pthread plugin_list)
    target_link_libraries(trtpyc ${OpenCV_LIBS})
    target_link_libraries(trtpyc "${PythonName}")
    target_link_libraries(pro "${PythonName}")
endif()

add_custom_target(
    yolo
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro yolo
)

add_custom_target(
    alphapose 
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro alphapose
)

add_custom_target(
    retinaface
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro retinaface
)

add_custom_target(
    arcface 
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro arcface
)

add_custom_target(
    fall
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro fall_recognize
)

add_custom_target(
    pyscrfd
    DEPENDS trtpyc
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/python
    COMMAND python test_scrfd.py
)

add_custom_target(
    pyinstall
    DEPENDS trtpyc
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/python
    COMMAND python setup.py install
)