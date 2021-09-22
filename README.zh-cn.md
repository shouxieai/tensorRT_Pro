阅读其他语言的README.md:[English](README.md), [简体中文](README.zh-cn.md).
## 最近的重要更新：
- 关于CenterNet 从pytorch到tensorRT的模型导出到推理的中英文教程已更新，在tutorial/2.0

## B站同步视频讲解
- B站视频讲解：https://www.bilibili.com/video/BV1Xw411f7FW
- 相关PPTX下载：http://zifuture.com:1556/fs/sxai/tensorRT.pptx
- tutorial 文件夹: 一个对入门者极其友好的框架概览和指南


## 高性能推理，TensorRT C++/Python库，工业级，便于使用

- C++接口，YoloX三行代码
```C++

// 创建推理引擎在0显卡上
//auto engine = Yolo::create_infer("yolov5m.fp32.trtmodel", Yolo::Type::V5, 0);
auto engine = Yolo::create_infer("yolox_m.fp32.trtmodel", Yolo::Type::X, 0);

// 加载图像
auto image = cv::imread("1.jpg");

// 推理并获取结果
auto box = engine->commit(image).get();  // 得到的是vector<Box>
```

- Python接口
```python
import trtpy

model     = models.resnet18(True).eval().to(device)
trt_model = tp.from_torch(model, input)
trt_out   = trt_model(input)
```


## 简介
1. 基于tensorRT8.0，C++/Python高级接口
2. 简化自定义插件的实现过程，封装序列化、反序列化
3. 简化fp32、fp16、int8编译过程，C++/Python部署，服务器/嵌入式使用
4. 高性能拿来就用的案例有RetinaFace、Scrfd、YoloV5、YoloX、Arcface、AlphaPose、DeepSORT(C++)


## YoloX和YoloV5系列所有模型性能测试

<details>
<summary>app_yolo.cpp速度测试</summary>

1. 输入分辨率(YoloV5P5、YoloX)=(640x640)，(YoloV5P6)=(1280x1280)
2. max batch size = 16
3. 图像预处理 + 推理 + 后处理
4. cuda10.2，cudnn8.2.2.26，TensorRT-8.0.1.6
5. RTX2080Ti
6. 测试次数，100次取平均，去掉warmup
7. 测试结果：[workspace/perf.result.std.log](workspace/perf.result.std.log)
8. 测试代码：[src/application/app_yolo.cpp](src/application/app_yolo.cpp)
9. 测试图像，6张。目录：workspace/inference
    - 分辨率分别为：810x1080，500x806，1024x684，550x676，1280x720，800x533
10. 测试方式，加载6张图后，以原图重复100次不停塞进去。让模型经历完整的图像的预处理，后处理

---

|模型名称|分辨率|模型类型|精度|耗时|帧率|
|---|---|---|---|---|---|
|yolox_x|640x640|YoloX|FP32|21.879 |45.71 |
|yolox_l|640x640|YoloX|FP32|12.308 |81.25 |
|yolox_m|640x640|YoloX|FP32|6.862 |145.72 |
|yolox_s|640x640|YoloX|FP32|3.088 |323.81 |
|yolox_x|640x640|YoloX|FP16|6.763 |147.86 |
|yolox_l|640x640|YoloX|FP16|3.933 |254.25 |
|yolox_m|640x640|YoloX|FP16|2.515 |397.55 |
|yolox_s|640x640|YoloX|FP16|1.362 |734.48 |
|yolox_x|640x640|YoloX|INT8|4.070 |245.68 |
|yolox_l|640x640|YoloX|INT8|2.444 |409.21 |
|yolox_m|640x640|YoloX|INT8|1.730 |577.98 |
|yolox_s|640x640|YoloX|INT8|1.060 |943.15 |
|yolov5x6|1280x1280|YoloV5_P6|FP32|68.022 |14.70 |
|yolov5l6|1280x1280|YoloV5_P6|FP32|37.931 |26.36 |
|yolov5m6|1280x1280|YoloV5_P6|FP32|20.127 |49.69 |
|yolov5s6|1280x1280|YoloV5_P6|FP32|8.715 |114.75 |
|yolov5x|640x640|YoloV5_P5|FP32|18.480 |54.11 |
|yolov5l|640x640|YoloV5_P5|FP32|10.110 |98.91 |
|yolov5m|640x640|YoloV5_P5|FP32|5.639 |177.33 |
|yolov5s|640x640|YoloV5_P5|FP32|2.578 |387.92 |
|yolov5x6|1280x1280|YoloV5_P6|FP16|20.877 |47.90 |
|yolov5l6|1280x1280|YoloV5_P6|FP16|10.960 |91.24 |
|yolov5m6|1280x1280|YoloV5_P6|FP16|7.236 |138.20 |
|yolov5s6|1280x1280|YoloV5_P6|FP16|3.851 |259.68 |
|yolov5x|640x640|YoloV5_P5|FP16|5.933 |168.55 |
|yolov5l|640x640|YoloV5_P5|FP16|3.450 |289.86 |
|yolov5m|640x640|YoloV5_P5|FP16|2.184 |457.90 |
|yolov5s|640x640|YoloV5_P5|FP16|1.307 |765.10 |
|yolov5x6|1280x1280|YoloV5_P6|INT8|12.207 |81.92 |
|yolov5l6|1280x1280|YoloV5_P6|INT8|7.221 |138.49 |
|yolov5m6|1280x1280|YoloV5_P6|INT8|5.248 |190.55 |
|yolov5s6|1280x1280|YoloV5_P6|INT8|3.149 |317.54 |
|yolov5x|640x640|YoloV5_P5|INT8|3.704 |269.97 |
|yolov5l|640x640|YoloV5_P5|INT8|2.255 |443.53 |
|yolov5m|640x640|YoloV5_P5|INT8|1.674 |597.40 |
|yolov5s|640x640|YoloV5_P5|INT8|1.143 |874.91 |

</details>

<details>
<summary>app_yolo_fast.cpp速度测试，速度只会无止境的追求快</summary>

- 相比上面，模型去头去尾，去掉了Focus和尾部的多余的transpose等节点，融合到了CUDA核函数中实现。其他都是一样的。没有精度区别，速度上提升大约0.5ms
- 测试结果：[workspace/perf.result.std.log](workspace/perf.result.std.log)
- 测试代码：[src/application/app_yolo_fast.cpp](src/application/app_yolo_fast.cpp)
- 可以自己参照下载后的onnx做修改，或者群里提要求讲一讲
- 这个工作的主要目的，是优化前后处理的时间，这在任何时候都是有用的。如果你用yolox、yolov5更小的系列，都可以考虑这东西


|模型名称|分辨率|模型类型|精度|耗时|帧率|
|---|---|---|---|---|---|
|yolox_x_fast|640x640|YoloX|FP32|21.598 |46.30 |
|yolox_l_fast|640x640|YoloX|FP32|12.199 |81.97 |
|yolox_m_fast|640x640|YoloX|FP32|6.819 |146.65 |
|yolox_s_fast|640x640|YoloX|FP32|2.979 |335.73 |
|yolox_x_fast|640x640|YoloX|FP16|6.764 |147.84 |
|yolox_l_fast|640x640|YoloX|FP16|3.866 |258.64 |
|yolox_m_fast|640x640|YoloX|FP16|2.386 |419.16 |
|yolox_s_fast|640x640|YoloX|FP16|1.259 |794.36 |
|yolox_x_fast|640x640|YoloX|INT8|3.918 |255.26 |
|yolox_l_fast|640x640|YoloX|INT8|2.292 |436.38 |
|yolox_m_fast|640x640|YoloX|INT8|1.589 |629.49 |
|yolox_s_fast|640x640|YoloX|INT8|0.954 |1048.47 |
|yolov5x6_fast|1280x1280|YoloV5_P6|FP32|67.075 |14.91 |
|yolov5l6_fast|1280x1280|YoloV5_P6|FP32|37.491 |26.67 |
|yolov5m6_fast|1280x1280|YoloV5_P6|FP32|19.422 |51.49 |
|yolov5s6_fast|1280x1280|YoloV5_P6|FP32|7.900 |126.57 |
|yolov5x_fast|640x640|YoloV5_P5|FP32|18.554 |53.90 |
|yolov5l_fast|640x640|YoloV5_P5|FP32|10.060 |99.41 |
|yolov5m_fast|640x640|YoloV5_P5|FP32|5.500 |181.82 |
|yolov5s_fast|640x640|YoloV5_P5|FP32|2.342 |427.07 |
|yolov5x6_fast|1280x1280|YoloV5_P6|FP16|20.538 |48.69 |
|yolov5l6_fast|1280x1280|YoloV5_P6|FP16|10.404 |96.12 |
|yolov5m6_fast|1280x1280|YoloV5_P6|FP16|6.577 |152.06 |
|yolov5s6_fast|1280x1280|YoloV5_P6|FP16|3.087 |323.99 |
|yolov5x_fast|640x640|YoloV5_P5|FP16|5.919 |168.95 |
|yolov5l_fast|640x640|YoloV5_P5|FP16|3.348 |298.69 |
|yolov5m_fast|640x640|YoloV5_P5|FP16|2.015 |496.34 |
|yolov5s_fast|640x640|YoloV5_P5|FP16|1.087 |919.63 |
|yolov5x6_fast|1280x1280|YoloV5_P6|INT8|11.236 |89.00 |
|yolov5l6_fast|1280x1280|YoloV5_P6|INT8|6.235 |160.38 |
|yolov5m6_fast|1280x1280|YoloV5_P6|INT8|4.311 |231.97 |
|yolov5s6_fast|1280x1280|YoloV5_P6|INT8|2.139 |467.45 |
|yolov5x_fast|640x640|YoloV5_P5|INT8|3.456 |289.37 |
|yolov5l_fast|640x640|YoloV5_P5|INT8|2.019 |495.41 |
|yolov5m_fast|640x640|YoloV5_P5|INT8|1.425 |701.71 |
|yolov5s_fast|640x640|YoloV5_P5|INT8|0.844 |1185.47 |


</details>

## 环境配置

<details>
<summary>Linux下配置</summary>

- 考虑方便，这里有编译好的依赖项
    - 下载地址：[lean-tensorRT8.0.1.6-protobuf3.11.4-cudnn8.2.2.tar.gz](http://zifuture.com:1556/fs/25.shared/lean-tensorRT8.0.1.6-protobuf3.11.4-cudnn8.2.2.tar.gz)
1. 推荐使用VSCode
2. 在Makefile/CMakeLists.txt中配置你的cudnn、cuda、tensorRT8.0、protobuf路径
3. 配置Makefile或者CMakeLists中的计算能力为你的显卡对应值
    - 例如`-gencode=arch=compute_75,code=sm_75`，例如3080Ti是86，则是：`-gencode=arch=compute_86,code=sm_86`
    - 计算能力根据型号参考这里查看：https://developer.nvidia.com/zh-cn/cuda-gpus#compute
4. 在.vscode/c_cpp_properties.json中配置你的库路径
5. CUDA版本：CUDA10.2
6. CUDNN版本：cudnn8.2.2.26，注意下载dev（h文件）和runtime（so文件）
7. tensorRT版本：tensorRT-8.0.1.6-cuda10.2，若要使用7.x，请看环节配置中的《TensorRT7.x支持》进行修改
8. protobuf版本（用于onnx解析器）：这里使用的是protobufv3.11.4
    - 如果采用其他版本，请参考该章节下面《适配Protobuf版本》
    - 下载地址：https://github.com/protocolbuffers/protobuf/tree/v3.11.4
    - 下载并编译，然后修改Makefile或者CMakeLists.txt的路径指向protobuf3.11.4
- CMake:
    - `mkdir build && cd build`
    - `cmake ..`
    - `make yolo -j8`
- Makefile:
    - `make yolo -j8`

</details>

<details>
<summary>Linux下Python编译</summary>

- 编译并安装:
    - Makefile方式：
        - 在Makefile中设置`use_python := true`启用python支持
    - CMakeLists.txt方式:
        - 在CMakeLists.txt中修改`set(HAS_PYTHON ON)`
    - 执行编译`make pyinstall -j8`
    - 编译后的文件，在`python/trtpy/libtrtpyc.so`

</details>


<details>
<summary>Windows下配置</summary>

1. 依赖请查看[lean/README.md](lean/README.md)
2. TensorRT.vcxproj文件中，修改`<Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 10.0.props" />`为你配置的CUDA路径
3. TensorRT.vcxproj文件中，修改`<Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 10.0.targets" />`为你配置的CUDA路径
4. TensorRT.vcxproj文件中，修改`<CodeGeneration>compute_61,sm_61</CodeGeneration>`为你显卡配备的计算能力
    - 根据型号参考这里：https://developer.nvidia.com/zh-cn/cuda-gpus#compute
5. 配置依赖或者下载依赖到lean中。配置VC++目录->包含目录和引用目录
6. 配置环境，调试->环境，设置PATH路径
7. 编译并运行案例，其中Debug为调试，Release为发布，Python为trtpyc模块

</details>

<details>
<summary>Windows下Python编译</summary>

1. 编译trtpyc.pyd，在visual studio中选择python进行编译
2. 复制dll，执行python/copy_dll_to_trtpy.bat
3. 在python目录下执行案例，python test_yolov5.py
- 如果需要进行安装，则在python目录下，切换到目标环境后，执行`python setup.py install`。（注意，执行了1、2两步后才行）
- 编译后的文件，在`python/trtpy/libtrtpyc.pyd`

</details>


<details>
<summary>适配Protobuf版本</summary>

- 修改onnx/make_pb.sh文件中protoc程序的路径`protoc=/data/sxai/lean/protobuf3.11.4/bin/protoc`，指向你自己版本的protoc

```bash
#切换终端目录到onnx下
cd onnx

#执行生成pb文件，并自动复制。使用make_pb.sh脚本
bash make_pb.sh
```

- CMake:
    - 修改CMakeLists.txt中`set(PROTOBUF_DIR "/data/sxai/lean/protobuf3.11.4")`为protoc相同的路径
```bash
mkdir build && cd build
cmake ..
make yolo -j64
```

- Makefile:
    - 修改Makefile中`lean_protobuf  := /data/sxai/lean/protobuf3.11.4`为protoc的相同路径
```bash
make yolo -j64
```


</details>


<details>
<summary>TensorRT7.x支持</summary>

- 默认支持的是8.x
- CMakeLists.txt/MakeFile中修改tensorRT的路径
- 执行`bash onnx_parser/use_tensorrt_7.x.sh`，修改解析器支持为7.x
- 正常进行编译运行即可

</details>


<details>
<summary>TensorRT8.x支持</summary>

- 默认支持的是8.x，不需要修改
- CMakeLists.txt/MakeFile中修改tensorRT的路径
- 执行`bash onnx_parser/use_tensorrt_8.x.sh`，修改解析器支持为8.x
- 正常进行编译运行即可

</details>

## 各项任务支持

<details>
<summary>YoloV5支持</summary>

- yolov5的onnx，你的pytorch版本>=1.7时，导出的onnx模型可以直接被当前框架所使用
- 你的pytorch版本低于1.7时，或者对于yolov5其他版本（2.0、3.0、4.0），可以对opset进行简单改动后直接被框架所支持
- 如果你想实现低版本pytorch的tensorRT推理、动态batchsize等更多更高级的问题，请打开我们[博客地址](http://zifuture.com:8090)后找到二维码进群交流
1. 下载yolov5
```bash
git clone git@github.com:ultralytics/yolov5.git
```

2. 修改代码，保证动态batchsize
```python
# yolov5/models/yolo.py第55行，forward函数 
# bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
# x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
# 修改为:

bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
bs = -1
ny = int(ny)
nx = int(nx)
x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

# yolov5/models/yolo.py第70行
#  z.append(y.view(bs, -1, self.no))
# 修改为：
z.append(y.view(bs, self.na * ny * nx, self.no))

# yolov5/export.py第52行
#torch.onnx.export(dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
#                                'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)  修改为
torch.onnx.export(dynamic_axes={'images': {0: 'batch'},  # shape(1,3,640,640)
                                'output': {0: 'batch'}  # shape(1,25200,85) 
```

3. 导出onnx模型
```bash
cd yolov5
python export.py --weights=yolov5s.pt --dynamic --include=onnx --opset=11
```
4. 复制模型并执行
```bash
cp yolov5/yolov5s.onnx tensorRT_cpp/workspace/
cd tensorRT_cpp
make yolo -j32
```

</details>


<details>
<summary>YoloX支持</summary>

- https://github.com/Megvii-BaseDetection/YOLOX
- 你可以选择直接make run，会从镜像地址下载onnx并推理运行看到效果。不需要自行导出
1. 下载YoloX
```bash
git clone git@github.com:Megvii-BaseDetection/YOLOX.git
cd YOLOX
```

2. 修改代码
- 这是保证int8能够顺利编译和性能提升的关键，否则提示`Missing scale and zero-point for tensor (Unnamed Layer* 686)`
- 这是保证模型推理正常顺利的关键，虽然部分情况不修改也可以执行
```Python
# yolox/models/yolo_head.py的206行forward函数，替换为下面代码
# self.hw = [x.shape[-2:] for x in outputs]
self.hw = [list(map(int, x.shape[-2:])) for x in outputs]


# yolox/models/yolo_head.py的208行forward函数，替换为下面代码
# [batch, n_anchors_all, 85]
# outputs = torch.cat(
#     [x.flatten(start_dim=2) for x in outputs], dim=2
# ).permute(0, 2, 1)
proc_view = lambda x: x.view(-1, int(x.size(1)), int(x.size(2) * x.size(3)))
outputs = torch.cat(
    [proc_view(x) for x in outputs], dim=2
).permute(0, 2, 1)


# yolox/models/yolo_head.py的253行decode_outputs函数，替换为下面代码
#outputs[..., :2] = (outputs[..., :2] + grids) * strides
#outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
#return outputs
xy = (outputs[..., :2] + grids) * strides
wh = torch.exp(outputs[..., 2:4]) * strides
return torch.cat((xy, wh, outputs[..., 4:]), dim=-1)


# tools/export_onnx.py的77行
model.head.decode_in_inference = True
```

3. 导出onnx模型
```bash

# 下载模型，或许你需要翻墙
# wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth

# 导出模型
python tools/export_onnx.py -c yolox_m.pth -f exps/default/yolox_m.py --output-name=yolox_m.onnx --dynamic --no-onnxsim
```

4. 执行程序
```bash
cp YOLOX/yolox_m.onnx tensorRT_cpp/workspace/
cd tensorRT_cpp
make yolo -j32
```

</details>


<details>
<summary>Retinaface支持</summary>


- https://github.com/biubug6/Pytorch_Retinaface
1. 下载Pytorch_Retinaface
```bash
git clone git@github.com:biubug6/Pytorch_Retinaface.git
cd Pytorch_Retinaface
```

2. 下载模型，请访问：https://github.com/biubug6/Pytorch_Retinaface#training 的training节点找到下载地址，解压到weights目录下，主要用到mobilenet0.25_Final.pth文件
3. 修改代码
```python
# models/retinaface.py第24行，
# return out.view(out.shape[0], -1, 2) 修改为
return out.view(-1, int(out.size(1) * out.size(2) * 2), 2)

# models/retinaface.py第35行，
# return out.view(out.shape[0], -1, 4) 修改为
return out.view(-1, int(out.size(1) * out.size(2) * 2), 4)

# models/retinaface.py第46行，
# return out.view(out.shape[0], -1, 10) 修改为
return out.view(-1, int(out.size(1) * out.size(2) * 2), 10)

# 以下是保证resize节点输出是按照scale而非shape，从而让动态大小和动态batch变为可能
# models/net.py第89行，
# up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest") 修改为
up3 = F.interpolate(output3, scale_factor=2, mode="nearest")

# models/net.py第93行，
# up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest") 修改为
up2 = F.interpolate(output2, scale_factor=2, mode="nearest")

# 以下代码是去掉softmax（某些时候有bug），同时合并输出为一个，简化解码部分代码
# models/retinaface.py第123行
# if self.phase == 'train':
#     output = (bbox_regressions, classifications, ldm_regressions)
# else:
#     output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
# return output
# 修改为
output = (bbox_regressions, classifications, ldm_regressions)
return torch.cat(output, dim=-1)

# 添加opset_version=11，使得算子按照预期导出
# torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False,
#     input_names=input_names, output_names=output_names)
torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False, opset_version=11,
    input_names=input_names, output_names=output_names)
```
4. 执行导出onnx
```bash
python convert_to_onnx.py
```

5. 执行
```bash
cp FaceDetector.onnx ../tensorRT_cpp/workspace/mb_retinaface.onnx
cd ../tensorRT_cpp
make retinaface -j64
```

</details>


<details>
<summary>DBFace支持</summary>

- https://github.com/dlunion/DBFace

```bash
make dbface -j64
```

</details>


<details>
<summary>Scrfd支持</summary>

- https://github.com/deepinsight/insightface/tree/master/detection/scrfd
- 具体导出Onnx的注意事项和方法，请加群沟通。等待后面更新

</details>


<details>
<summary>Arcface支持</summary>

- https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
```C++
auto arcface = Arcface::create_infer("arcface_iresnet50.fp32.trtmodel", 0);
auto feature = arcface->commit(make_tuple(face, landmarks)).get();
cout << feature << endl;  // 1x512
```
- 人脸识别案例中，`workspace/face/library`目录为注册入库人脸
- 人脸识别案例中，`workspace/face/recognize`目录为待识别的照片
- 结果储存在`workspace/face/result`和`workspace/face/library_draw`中

</details>


## 接口介绍

<details>
<summary>Python接口：从Pytorch模型导出Onnx和trtmodel</summary>

- 使用Python接口可以一句话导出Onnx和trtmodel，一次性调试发生的问题，解决问题。并储存onnx为后续部署使用
```python
import trtpy

model = models.resnet18(True).eval()
trtpy.from_torch(
    model, 
    dummy_input, 
    max_batch_size=16, 
    onnx_save_file="test.onnx", 
    engine_save_file="engine.trtmodel"
)
```

</details>

<details>
<summary>Python接口：TensorRT的推理</summary>

- YoloX的tensorRT推理
```python
import trtpy

yolo   = tp.Yolo(engine_file, type=tp.YoloType.X)
image  = cv2.imread("inference/car.jpg")
bboxes = yolo.commit(image).get()
```

- Pytorch的无缝对接
```python
import trtpy

model     = models.resnet18(True).eval().to(device)
trt_model = tp.from_torch(model, input)
trt_out   = trt_model(input)
```

</details>


<details>
<summary>C++接口：YoloX推理</summary>

```C++

// 创建推理引擎在0显卡上
auto engine = Yolo::create_infer("yolox_m.fp32.trtmodel"， Yolo::Type::X, 0);

// 加载图像
auto image = cv::imread("1.jpg");

// 推理并获取结果
auto box = engine->commit(image).get();
```

</details>


<details>
<summary>C++接口：编译模型FP32/FP16</summary>

```cpp
TRT::compile(
  TRT::Mode::FP32,   // 使用fp32模型编译
  3,                          // max batch size
  "plugin.onnx",              // onnx 文件
  "plugin.fp32.trtmodel",     // 保存的文件路径
  {}                         // 重新定制输入的shape
);
```
- 对于FP32编译，只需要提供onnx文件即可，可以允许重定义onnx输入节点的shape
- 对于动态或者静态batch的支持，仅仅只需要一个选项，这对于官方发布的解析器是不支持的

</details>

<details>
<summary>C++接口：编译INT8模型</summary>

- 众所周知，int8的推理效果比fp32稍微差一点（预计-5%的损失），但是速度确快很多很多，这里通过集成的编译方式，很容易实现int8的编译工作
```cpp
// 定义int8的标定数据处理函数，读取数据并交给tensor的函数
auto int8process = [](int current, int count, vector<string>& images, shared_ptr<TRT::Tensor>& tensor){
    for(int i = 0; i < images.size(); ++i){

    // 对于int8的编译需要进行标定，这里读取图像数据并通过set_norm_mat到tensor中
        auto image = cv::imread(images[i]);
        cv::resize(image, image, cv::Size(640, 640));
        float mean[] = {0, 0, 0};
        float std[]  = {1, 1, 1};
        tensor->set_norm_mat(i, image, mean, std);
    }
};


// 编译模型指定为INT8
auto model_file = "yolov5m.int8.trtmodel";
TRT::compile(
  TRT::Mode::INT8,            // 选择INT8
  3,                          // max batch size
  "yolov5m.onnx",             // onnx文件
  model_file,                 // 编译后保存的文件
  {},                         // 重定义输入的shape
  int8process,                // 指定int8标定数据的处理回调函数
  ".",                        // 指定int8标定图像数据的目录
  ""                          // 指定int8标定后的数据储存/读取路径
);
```
- 避免了官方标定流程分离的问题，复杂度太高，在这里直接集成为一个函数处理

</details>


<details>
<summary>C++接口：推理</summary>

- 对于模型推理，封装了Tensor类，实现推理的维护和数据交互，对于数据从GPU到CPU过程完全隐藏细节
- 封装了Engine类，实现模型推理和管理
```cpp
// 模型加载，得到一个共享指针，如果为空表示加载失败
auto engine = TRT::load_infer("yolov5m.fp32.trtmodel");

// 打印模型信息
engine->print();

// 加载图像
auto image = imread("demo.jpg");

// 获取模型的输入和输出tensor节点，可以根据名字或者索引获取具体第几个
auto input = engine->input(0);
auto output = engine->output(0);

// 把图像塞到input tensor中，这里是减去均值，并除以标准差
float mean[] = {0, 0, 0};
float std[]  = {1, 1, 1};
input->set_norm_mat(i, image, mean, std);

// 执行模型的推理，这里可以允许异步或者同步
engine->forward();

// 这里拿到的指针即是最终的结果指针，可以进行访问操作
float* output_ptr = output->cpu<float>();
// 这里对output_ptr进行处理即可得到结果
```

</details>


<details>
<summary>C++接口：插件</summary>

- 只需要定义必要的核函数和推理过程，完全隐藏细节，隐藏插件的序列化、反序列化、注入
- 可以简洁的实现FP32、FP16两种格式支持的插件。具体参见代码HSwish cu/hpp
```cpp
template<>
__global__ void HSwishKernel(float* input, float* output, int edge) {

    KernelPositionBlock;
    float x = input[position];
    float a = x + 3;
    a = a < 0 ? 0 : (a >= 6 ? 6 : a);
    output[position] = x * a / 6;
}

int HSwish::enqueue(const std::vector<GTensor>& inputs, std::vector<GTensor>& outputs, const std::vector<GTensor>& weights, void* workspace, cudaStream_t stream) {

    int count = inputs[0].count();
    auto grid = CUDATools::grid_dims(count);
    auto block = CUDATools::block_dims(count);
    HSwishKernel <<<grid, block, 0, stream >>> (inputs[0].ptr<float>(), outputs[0].ptr<float>(), count);
    return 0;
}


RegisterPlugin(HSwish);
```

</details>


## 关于
- 我们的博客地址：http://www.zifuture.com/
- 我们的B站地址： https://space.bilibili.com/1413433465
