cc        := g++
name      := libffhdd.so
workdir   := workspace
srcdir    := src
objdir    := objs
stdcpp    := c++11
cuda_home := /datav/lean/cuda-11.2
cudnn_home:= /datav/lean/cudnn8.2.4.15-cuda11.4
cuda_arch := 

# 定义cpp的路径查找和依赖项mk文件
cpp_srcs := $(shell find $(srcdir) -name "*.cpp")
cpp_objs := $(cpp_srcs:.cpp=.cpp.o)
cpp_objs := $(cpp_objs:$(srcdir)/%=$(objdir)/%)
cpp_mk   := $(cpp_objs:.cpp.o=.cpp.mk)

# 定义cu文件的路径查找和依赖项mk文件
cu_srcs := $(shell find $(srcdir) -name "*.cu")
cu_objs := $(cu_srcs:.cu=.cu.o)
cu_objs := $(cu_objs:$(srcdir)/%=$(objdir)/%)
cu_mk   := $(cu_objs:.cu.o=.cu.mk)

# 定义opencv和cuda需要用到的库文件
link_cuda      := cuda cudart nvcuvid nvidia-encode
link_ffmpeg    := avcodec avformat swresample swscale avutil
link_sys       := stdc++ dl python3.9
link_librarys  := $(link_cuda) $(link_sys) $(link_ffmpeg)

# 定义cuda和opencv的库路径
nvcc              := $(cuda_home)/bin/nvcc -ccbin=$(cc)
include_cuda      := $(cuda_home)/include
current_dir       := $(shell pwd)

lib_all        := $(cuda_home)/lib64 $(cudnn_home)/lib $(current_dir)/ffmpeg/lib /lib/x86_64-linux-gnu /datav/software/anaconda3/lib

# 定义头文件路径，请注意斜杠后边不能有空格
# 只需要写路径，不需要写-I
include_paths := src    \
    $(include_cuda)     \
	/datav/software/anaconda3/include/python3.9 \
	src/application     \
	ffmpeg/include      \
	cuvid-include        

# 定义库文件路径，只需要写路径，不需要写-L
library_paths := $(lib_all)

# 把library path给拼接为一个字符串，例如a b c => a:b:c
# 然后使得LD_LIBRARY_PATH=a:b:c
empty := 
library_path_export := $(subst $(empty) $(empty),:,$(library_paths))

# 把库路径和头文件路径拼接起来成一个，批量自动加-I、-L、-l
run_paths     := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))

# 如果是其他显卡，请修改-gencode=arch=compute_75,code=sm_75为对应显卡的能力
# 显卡对应的号码参考这里：https://developer.nvidia.com/zh-cn/cuda-gpus#compute
# 如果是 jetson nano，提示找不到-m64指令，请删掉 -m64选项。不影响结果
cpp_compile_flags := -std=$(stdcpp) -w -g -O0 -m64 -fPIC -fopenmp -pthread
cu_compile_flags  := -std=$(stdcpp) -w -g -O0 -m64 $(cuda_arch) -Xcompiler "$(cpp_compile_flags)"
link_flags        := -pthread -fopenmp -Wl,-rpath='$$ORIGIN'

cpp_compile_flags += $(include_paths)
cu_compile_flags  += $(include_paths)
link_flags        += $(library_paths) $(link_librarys) $(run_paths)

# 如果头文件修改了，这里的指令可以让他自动编译依赖的cpp或者cu文件
ifneq ($(MAKECMDGOALS), clean)
-include $(cpp_mk) $(cu_mk)
endif

$(name)   : $(workdir)/$(name)

all       : $(name)
run       : $(name)
	@cd $(workdir) && python test.py

runhdd    : $(name)
	@cd $(workdir) && python test_hard_decode_yolov5.py

pro       : $(workdir)/pro
runpro    : pro
	@cd $(workdir) && ./pro

$(workdir)/$(name) : $(cpp_objs) $(cu_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@$(cc) -shared $^ -o $@ $(link_flags)

$(workdir)/pro : $(cpp_objs) $(cu_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@$(cc) $^ -o $@ $(link_flags)

$(objdir)/%.cpp.o : $(srcdir)/%.cpp
	@echo Compile CXX $<
	@mkdir -p $(dir $@)
	@$(cc) -c $< -o $@ $(cpp_compile_flags)

$(objdir)/%.cu.o : $(srcdir)/%.cu
	@echo Compile CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -c $< -o $@ $(cu_compile_flags)

# 编译cpp依赖项，生成mk文件
$(objdir)/%.cpp.mk : $(srcdir)/%.cpp
	@echo Compile depends C++ $<
	@mkdir -p $(dir $@)
	@$(cc) -M $< -MF $@ -MT $(@:.cpp.mk=.cpp.o) $(cpp_compile_flags)
    
# 编译cu文件的依赖项，生成cumk文件
$(objdir)/%.cu.mk : $(srcdir)/%.cu
	@echo Compile depends CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -M $< -MF $@ -MT $(@:.cu.mk=.cu.o) $(cu_compile_flags)

# 定义清理指令
clean :
	@rm -rf $(objdir) $(workdir)/$(name) $(workdir)/pro $(workdir)/*.trtmodel $(workdir)/imgs

# 防止符号被当做文件
.PHONY : clean run $(name)

# 导出依赖库路径，使得能够运行起来
export LD_LIBRARY_PATH:=$(library_path_export)