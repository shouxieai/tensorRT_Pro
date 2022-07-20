cc        := g++
nvcc      = ${lean_cuda}/bin/nvcc

cpp_srcs := $(shell find src -name "*.cpp")
cpp_objs := $(cpp_srcs:.cpp=.o)
cpp_objs := $(cpp_objs:src/%=objs/%)
cpp_mk   := $(cpp_objs:.o=.mk)

cu_srcs := $(shell find src -name "*.cu")
cu_objs := $(cu_srcs:.cu=.cuo)
cu_objs := $(cu_objs:src/%=objs/%)
cu_mk   := $(cu_objs:.cuo=.cumk)

# 配置你的库路径
# 1. cudnn8.2.2.26（请自行下载）
#    runtime的tar包，runtime中包含了lib、so文件
#    develop的tar包，develop中包含了include、h等文件
# 2. tensorRT-8.0.1.6-cuda10.2（请自行下载）
#    tensorRT下载GA版本（通用版、稳定版），EA（尝鲜版本）不要
# 3. cuda10.2，也可以是11.x看搭配（请自行下载安装）

# lean_tensor_rt := /data/sxai/lean/TensorRT-8.0.1.6-cuda10.2-cudnn8.2
# lean_cudnn     := /data/sxai/lean/cudnn8.2.2.26
# lean_opencv    := /data/sxai/lean/opencv4.2.0
# lean_cuda      := /data/sxai/lean/cuda-10.2
lean_tensor_rt := /datav/lean/TensorRT-8.2.3.0-cuda11.4-cudnn8.2
lean_cudnn     := /datav/lean/cudnn8.2.4.15-cuda11.4
lean_opencv    := /datav/lean/opencv-4.2.0
lean_cuda      := /datav/lean/cuda-11.2

# lean_tensor_rt := /data/sxai/lean/TensorRT-7.0.0.11
# lean_cudnn     := /data/sxai/lean/cudnn7.6.5.32-cuda10.2
# lean_opencv    := /data/sxai/lean/opencv4.2.0
# lean_cuda      := /data/sxai/lean/cuda-10.2

# lean_tensor_rt := /data/sxai/lean/TensorRT-7.2.1.6
# lean_cudnn     := /data/sxai/lean/cudnn8.2.2.26
# lean_opencv    := /data/sxai/lean/opencv4.2.0
# lean_cuda      := /data/sxai/lean/cuda-11.1

include_paths := src        \
			$(lean_opencv)/include/opencv4 \
			$(lean_tensor_rt)/include \
			$(lean_cuda)/include  \
			$(lean_cudnn)/include 

library_paths := $(lean_tensor_rt)/lib \
			$(lean_opencv)/lib    \
			$(lean_cuda)/lib64  \
			$(lean_cudnn)/lib

link_librarys := opencv_core opencv_imgproc opencv_videoio opencv_imgcodecs \
			nvinfer nvonnxparser \
			cuda cublas cudart cudnn \
			stdc++ dl

empty         :=
export_path   := $(subst $(empty) $(empty),:,$(library_paths))

paths     := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))

# 如果是其他显卡，请修改-gencode=arch=compute_75,code=sm_75为对应显卡的能力
# 显卡对应的号码参考这里：https://developer.nvidia.com/zh-cn/cuda-gpus#compute
# 如果是 jetson nano，提示找不到-m64指令，请删掉 -m64选项。不影响结果
cpp_compile_flags := -std=c++11 -fPIC -m64 -g -fopenmp -w -O0
cu_compile_flags  := -std=c++11 -m64 -Xcompiler -fPIC -g -w -O0
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

python/pytrt/libpytrtc.so : $(cpp_objs) $(cu_objs)
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
	@${nvcc} -c $< -o $@ $(cu_compile_flags)

objs/%.mk : src/%.cpp
	@echo Compile depends CXX $<
	@mkdir -p $(dir $@)
	@g++ -M $< -MF $@ -MT $(@:.mk=.o) $(cpp_compile_flags)
	
objs/%.cumk : src/%.cu
	@echo Compile depends CUDA $<
	@mkdir -p $(dir $@)
	@${nvcc} -M $< -MF $@ -MT $(@:.cumk=.cuo) $(cu_compile_flags)

run : workspace/pro
	@cd workspace && ./pro

debug :
	@echo $(export_path)

clean :
	@rm -rf objs workspace/pro build

.PHONY : clean yolo alphapose fall debug

# 导出符号，使得运行时能够链接上
export LD_LIBRARY_PATH:=$(export_path):$(LD_LIBRARY_PATH)