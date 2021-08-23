
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
lean_tensor_rt := /data/sxai/lean/TensorRT-8.0.1.6
lean_cudnn     := /data/sxai/lean/cudnn8.2.2.26
lean_opencv    := /data/sxai/lean/opencv4.2.0
lean_cuda      := /data/sxai/lean/cuda10.2

include_paths := src \
			src/core \
			src/tensorRT	\
			src/tensorRT/common  \
			$(lean_protobuf)/include \
			$(lean_opencv)/include/opencv4 \
			$(lean_tensor_rt)/include \
			$(lean_cuda)/include \
			$(lean_cudnn)/include 

library_paths := $(lean_protobuf)/lib \
			$(lean_opencv)/lib \
			$(lean_tensor_rt)/lib \
			$(lean_cuda)/lib \
			$(lean_cudnn)/lib

link_librarys := opencv_core opencv_imgproc opencv_videoio opencv_imgcodecs \
			nvinfer nvinfer_plugin nvparsers \
			cuda curand cublas cudart cudnn \
			stdc++ protobuf dl

run_paths     := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))

# 如果要支持FP16的插件推理（非插件不需要），请在编译选项上加-DHAS_CUDA_HALF，CPP和CU都加
# 这种特殊的宏可以在.vscode/c_cpp_properties.json文件中configurations下的defines中也加进去，使得看代码的时候
# 效果与编译一致
# support_define    := -DHAS_CUDA_HALF
support_define    := 
cpp_compile_flags := -std=c++11 -fPIC -m64 -g -fopenmp -w -O0 $(support_define)
cu_compile_flags  := -std=c++11 -m64 -Xcompiler -fPIC -g -w -gencode=arch=compute_75,code=sm_75 -O0 $(support_define)
link_flags        := -pthread -fopenmp

cpp_compile_flags += $(include_paths)
cu_compile_flags  += $(include_paths)
link_flags 		  += $(library_paths) $(link_librarys) $(run_paths)

ifneq ($(MAKECMDGOALS), clean)
-include $(cpp_mk) $(cu_mk)
endif

pro : workspace/pro

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

run_yolo : workspace/pro
	@cd workspace && ./pro yolo

run_alphapose : workspace/pro
	@cd workspace && ./pro alphapose

run_fall : workspace/pro
	@cd workspace && ./pro app_fall_recognize

debug :
	@echo $(includes)

clean :
	@rm -rf objs workspace/pro

.PHONY : clean run_yolo run_alphapose run_fall debug