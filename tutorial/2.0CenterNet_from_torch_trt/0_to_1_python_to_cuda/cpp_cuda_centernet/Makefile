
cpp_srcs := $(shell find src -name "*.cpp")
cpp_objs := $(cpp_srcs:.cpp=.o)
cpp_objs := $(cpp_objs:src/%=objs/%)
cpp_mk   := $(cpp_objs:.o=.mk)

cu_srcs := $(shell find src -name "*.cu")
cu_objs := $(cu_srcs:.cu=.cuo)
cu_objs := $(cu_objs:src/%=objs/%)
cu_mk   := $(cu_objs:.cuo=.cumk)

# 配置你的库路径
lean_protobuf  := /datav/shared/wkx/tensorRT_cpp/lean/protobuf3.11.4
lean_tensor_rt := /datav/shared/wkx/tensorRT_cpp/lean/TensorRT-8.0.1.6
lean_cudnn     := /datav/shared/wkx/tensorRT_cpp/lean/cudnn8.2.2.26
lean_opencv    := /usr/local/opencv
lean_cuda      := /usr/local/cuda
lean_python    := /root/anaconda3
use_python     := true

# The Makefile is used for learning. If you want to run it, just modify the 2nd and 3rd line
# into {your_path_to_tensorRT_cpp}/src/tensorRT
# into {your_path_to_tensorRT_cpp}/src/tensorRT/common

include_paths := src        \
			src/tensorRT	\
			src/tensorRT/common  \
			$(lean_protobuf)/include \
			$(lean_opencv)/include/opencv4 \
			$(lean_tensor_rt)/include \
			$(lean_cuda)/include  \
			$(lean_cudnn)/include 

library_paths := $(lean_protobuf)/lib \
			$(lean_opencv)/lib    \
			$(lean_tensor_rt)/lib \
			$(lean_cuda)/lib  \
			/usr/local/cuda-10.2/targets/x86_64-linux/lib  \
			$(lean_cudnn)/lib \
			/usr/lib/x86_64-linux-gnu/

link_librarys := opencv_core opencv_imgproc opencv_videoio opencv_imgcodecs \
			nvinfer nvinfer_plugin \
			cuda cublas cudart cudnn \
			stdc++ protobuf dl


# HAS_PYTHON表示是否编译python支持
support_define    := 

ifeq ($(use_python), true) 
include_paths  += $(lean_python)/include/python3.8
library_paths  += $(lean_python)/lib
link_librarys  += python3.8
support_define += -DHAS_PYTHON
endif

paths     := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))

# 如果是其他显卡，请修改-gencode=arch=compute_75,code=sm_75为对应显卡的能力
cpp_compile_flags := -std=c++11 -fPIC -m64 -g -fopenmp -w -O0 $(support_define)
cu_compile_flags  := -std=c++11 -m64 -Xcompiler -fPIC -g -w -gencode=arch=compute_75,code=sm_75 -O0 $(support_define)
link_flags        := -pthread -fopenmp -Wl,-rpath='$$ORIGIN'

cpp_compile_flags += $(include_paths)
cu_compile_flags  += $(include_paths)
link_flags 		  += $(library_paths) $(link_librarys) $(paths)

ifneq ($(MAKECMDGOALS), clean)
-include $(cpp_mk) $(cu_mk)
endif

pro    : workspace/pro

workspace/pro : $(cpp_objs) $(cu_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@g++ $^ -o $@ $(link_flags)


objs/%.o : src/%.cpp
	@echo Compile CXX $<
	@mkdir -p $(dir $@)
	@g++ -c $< -o $@ $(cpp_compile_flags)

objs/%.cuo : src/%.cu
	@echo Compile CUDA $<
	@mkdir -p $(dir $@)
	@nvcc -c $< -o $@ $(cu_compile_flags)

objs/%.mk : src/%.cpp
	@echo Compile depends CXX $<
	@mkdir -p $(dir $@)
	@g++ -M $< -MF $@ -MT $(@:.mk=.o) $(cpp_compile_flags)
	
objs/%.cumk : src/%.cu
	@echo Compile depends CUDA $<
	@mkdir -p $(dir $@)
	@nvcc -M $< -MF $@ -MT $(@:.cumk=.o) $(cu_compile_flags)


centernet : workspace/pro
	@cd workspace && ./pro centernet

clean :
	@rm -rf objs workspace/pro python/pytrt/libpytrtc.so python/build python/dist python/pytrt.egg-info python/pytrt/__pycache__
	@rm -rf workspace/single_inference
	@rm -rf workspace/scrfd_result workspace/retinaface_result
	@rm -rf workspace/YoloV5_result workspace/YoloX_result
	@rm -rf workspace/face/library_draw workspace/face/result
	@rm -rf build

.PHONY : clean yolo alphapose fall debug