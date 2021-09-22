
cpp_srcs := $(shell find src -name "*.cpp")
cpp_objs := $(cpp_srcs:.cpp=.o)
cpp_objs := $(cpp_objs:src/%=objs/%)
cpp_mk   := $(cpp_objs:.o=.mk)

cu_srcs := $(shell find src -name "*.cu")
cu_objs := $(cu_srcs:.cu=.cuo)
cu_objs := $(cu_objs:src/%=objs/%)
cu_mk   := $(cu_objs:.cuo=.cumk)

# 配置你的库路径
# 1. onnx-tensorrt（项目集成了，不需要配置，下面地址是下载位置）
#    https://github.com/onnx/onnx-tensorrt/tree/release/8.0
# 2. protobuf（请自行下载编译）
#    https://github.com/protocolbuffers/protobuf/tree/v3.11.4
# 3. cudnn8.2.2.26（请自行下载）
#    runtime的tar包，runtime中包含了lib、so文件
#    develop的tar包，develop中包含了include、h等文件
# 4. tensorRT-8.0.1.6-cuda10.2（请自行下载）
#    tensorRT下载GA版本（通用版、稳定版），EA（尝鲜版本）不要
# 5. cuda10.2，也可以是11.x看搭配（请自行下载安装）

lean_protobuf  := /data/sxai/lean/protobuf3.11.4
lean_tensor_rt := /data/sxai/lean/TensorRT-8.0.1.6-cuda10.2-cudnn8.2
lean_cudnn     := /data/sxai/lean/cudnn8.2.2.26
lean_opencv    := /data/sxai/lean/opencv4.2.0
lean_cuda      := /data/sxai/lean/cuda-10.2
use_python     := true
python_root    := /data/datav/newbb/lean/anaconda3/envs/torch1.8
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
			$(lean_cudnn)/lib \
			/datav/k12/lean/ffmpeg4.2/lib

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

paths     := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))

# 如果是其他显卡，请修改-gencode=arch=compute_75,code=sm_75为对应显卡的能力
# 显卡对应的号码参考这里：https://developer.nvidia.com/zh-cn/cuda-gpus#compute
# 如果是jetson nano，提示找不到-m64指令，请删掉 -m64选项。不影响结果
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
trtpyc : python/trtpy/libtrtpyc.so

workspace/pro : $(cpp_objs) $(cu_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@g++ $^ -o $@ $(link_flags)

python/trtpy/libtrtpyc.so : $(cpp_objs) $(cu_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@g++ -shared $^ -o $@ $(link_flags)

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

yolo : workspace/pro
	@cd workspace && ./pro yolo

yolo_fast : workspace/pro
	@cd workspace && ./pro yolo_fast

alphapose : workspace/pro
	@cd workspace && ./pro alphapose

fall : workspace/pro
	@cd workspace && ./pro fall_recognize

retinaface : workspace/pro
	@cd workspace && ./pro retinaface

arcface    : workspace/pro
	@cd workspace && ./pro arcface

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

pyinstall : trtpyc
	@cd python && python setup.py install

debug :
	@echo $(includes)

clean :
	@rm -rf objs workspace/pro python/trtpy/libtrtpyc.so python/build python/dist python/trtpy.egg-info python/trtpy/__pycache__
	@rm -rf workspace/single_inference
	@rm -rf workspace/scrfd_result workspace/retinaface_result
	@rm -rf workspace/YoloV5_result workspace/YoloX_result
	@rm -rf workspace/face/library_draw workspace/face/result
	@rm -rf build
	@rm -rf python/trtpy/libplugin_list.so

.PHONY : clean yolo alphapose fall debug