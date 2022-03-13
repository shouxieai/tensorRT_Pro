cc        := g++
nvcc      = ${lean_cuda}/bin/nvcc

# 如果是其他显卡，请修改-gencode=arch=compute_75,code=sm_75为对应显卡的能力
# 显卡对应的号码参考这里：https://developer.nvidia.com/zh-cn/cuda-gpus#compute
cuda_arch := -gencode=arch=compute_75,code=sm_75

cpp_srcs  := $(shell find src -name "*.cpp")
cpp_objs  := $(cpp_srcs:.cpp=.cpp.o)
cpp_objs  := $(cpp_objs:src/%=objs/%)
cpp_mk    := $(cpp_objs:.cpp.o=.cpp.mk)

cu_srcs  := $(shell find src -name "*.cu")
cu_objs  := $(cu_srcs:.cu=.cu.o)
cu_objs  := $(cu_objs:src/%=objs/%)
cu_mk    := $(cu_objs:.cu.o=.cu.mk)

lean_protobuf  := /datav/lean/protobuf3.11.4
lean_tensor_rt := /datav/lean/TensorRT-8.2.3.0-cuda11.4-cudnn8.2
lean_cudnn     := /datav/lean/cudnn8.2.4.15-cuda11.4
lean_opencv    := /datav/lean/opencv-4.2.0
lean_cuda      := /datav/lean/cuda-11.2
use_python     := false
python_root    := /datav/software/anaconda3
python_name    := python3.9

include_paths := src        \
			src/application \
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
			$(lean_cuda)/lib64  \
			$(lean_cudnn)/lib

link_librarys := opencv_core opencv_imgproc opencv_videoio opencv_imgcodecs \
			nvinfer nvinfer_plugin \
			cuda cublas cudart cudnn \
			stdc++ protobuf dl


# HAS_PYTHON表示是否编译python支持
support_define    := 

ifeq ($(use_python), true) 
include_paths  += $(python_root)/include/$(python_name)
library_paths  += $(python_root)/lib
link_librarys  += $(python_name)
support_define += -DHAS_PYTHON
endif

empty         :=
export_path   := $(subst $(empty) $(empty),:,$(library_paths))

run_paths     := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))

cpp_compile_flags := -std=c++11 -g -w -O0 -fPIC -pthread -fopenmp $(support_define)
cu_compile_flags  := -std=c++11 -g -w -O0 -Xcompiler "$(cpp_compile_flags)" $(cuda_arch) $(support_define)
link_flags        := -pthread -fopenmp -Wl,-rpath='$$ORIGIN'

cpp_compile_flags += $(include_paths)
cu_compile_flags  += $(include_paths)
link_flags        += $(library_paths) $(link_librarys) $(run_paths)

ifneq ($(MAKECMDGOALS), clean)
-include $(cpp_mk) $(cu_mk)
endif

pro    : workspace/pro
trtpyc : python/trtpy/libtrtpyc.so
expath : library_path.txt

library_path.txt : 
	@echo LD_LIBRARY_PATH=$(export_path):"$$"LD_LIBRARY_PATH > $@

workspace/pro : $(cpp_objs) $(cu_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@$(cc) $^ -o $@ $(link_flags)

python/trtpy/libtrtpyc.so : $(cpp_objs) $(cu_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@$(cc) -shared $^ -o $@ $(link_flags)

objs/%.cpp.o : src/%.cpp
	@echo Compile CXX $<
	@mkdir -p $(dir $@)
	@$(cc) -c $< -o $@ $(cpp_compile_flags)

objs/%.cu.o : src/%.cu
	@echo Compile CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -c $< -o $@ $(cu_compile_flags)

objs/%.cpp.mk : src/%.cpp
	@echo Compile depends CXX $<
	@mkdir -p $(dir $@)
	@$(cc) -M $< -MF $@ -MT $(@:.cpp.mk=.cpp.o) $(cpp_compile_flags)
	
objs/%.cu.mk : src/%.cu
	@echo Compile depends CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -M $< -MF $@ -MT $(@:.cu.mk=.cu.o) $(cu_compile_flags)

yolo : workspace/pro
	@cd workspace && ./pro yolo

dyolo : workspace/pro
	@cd workspace && ./pro dyolo

dunet : workspace/pro
	@cd workspace && ./pro dunet

dmae : workspace/pro
	@cd workspace && ./pro dmae

dclassifier : workspace/pro
	@cd workspace && ./pro dclassifier

yolo_fast : workspace/pro
	@cd workspace && ./pro yolo_fast

bert : workspace/pro
	@cd workspace && ./pro bert

alphapose : workspace/pro
	@cd workspace && ./pro alphapose

fall : workspace/pro
	@cd workspace && ./pro fall_recognize

retinaface : workspace/pro
	@cd workspace && ./pro retinaface

arcface    : workspace/pro
	@cd workspace && ./pro arcface

test_warpaffine    : workspace/pro
	@cd workspace && ./pro test_warpaffine

test_yolo_map    : workspace/pro
	@cd workspace && ./pro test_yolo_map

arcface_video    : workspace/pro
	@cd workspace && ./pro arcface_video

arcface_tracker    : workspace/pro
	@cd workspace && ./pro arcface_tracker

test_all : workspace/pro
	@cd workspace && ./pro test_all

scrfd : workspace/pro
	@cd workspace && ./pro scrfd

centernet : workspace/pro
	@cd workspace && ./pro centernet

dbface : workspace/pro
	@cd workspace && ./pro dbface

high_perf : workspace/pro
	@cd workspace && ./pro high_perf

lesson : workspace/pro
	@cd workspace && ./pro lesson

plugin : workspace/pro
	@cd workspace && ./pro plugin

pytorch : trtpyc
	@cd python && python test_torch.py

pyscrfd : trtpyc
	@cd python && python test_scrfd.py

pyretinaface : trtpyc
	@cd python && python test_retinaface.py

pycenternet : trtpyc
	@cd python && python test_centernet.py

pyyolov5 : trtpyc
	@cd python && python test_yolov5.py

pyyolox : trtpyc
	@cd python && python test_yolox.py

pyarcface : trtpyc
	@cd python && python test_arcface.py

pyinstall : trtpyc
	@cd python && python setup.py install

clean :
	@rm -rf objs workspace/pro python/trtpy/libtrtpyc.so python/build python/dist python/trtpy.egg-info python/trtpy/__pycache__
	@rm -rf workspace/single_inference
	@rm -rf workspace/scrfd_result workspace/retinaface_result
	@rm -rf workspace/YoloV5_result workspace/YoloX_result
	@rm -rf workspace/face/library_draw workspace/face/result
	@rm -rf build
	@rm -rf python/trtpy/libplugin_list.so
	@rm -rf library_path.txt

.PHONY : clean yolo alphapose fall debug

# 导出符号，使得运行时能够链接上
export LD_LIBRARY_PATH:=$(export_path):$(LD_LIBRARY_PATH)