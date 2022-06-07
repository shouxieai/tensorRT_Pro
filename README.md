*Read this in other languages: [English](README.md), [简体中文](README.zh-cn.md).*

## News: 
- 🔥 Docker Image has been released：https://hub.docker.com/r/hopef/tensorrt-pro
- ⚡tensorRT_Pro_comments_version(co-contributing version) is also provided for a better learning experience. Repo: https://github.com/Guanbin-Huang/tensorRT_Pro_comments
- 🔥 [Simple yolov5/yolox implemention is released. Simple and easy to use.](simple_yolo)
- 🔥 yolov5-1.0-6.0/master are supported.
- Tutorial notebooks download:
  - [WarpAffine.lesson.tar.gz](http://zifuture.com:1000/fs/25.shared/warpaffine.lesson.tar.gz)
  - [Offset.tar.gz](http://zifuture.com:1000/fs/25.shared/offset.tar.gz)
- Tutorial for exporting CenterNet from pytorch to tensorRT is released. 

## Tutorial Video

- <b>blibli</b> : https://www.bilibili.com/video/BV1Xw411f7FW (Now only in Chinese. English is comming)
- <b>slides</b> : http://zifuture.com:1556/fs/sxai/tensorRT.pptx (Now only in Chinese. English is comming)
- <b>tutorial folder</b>: a good intro for beginner to get a general idea of our framework.(Chinese/English)

## An Out-of-the-Box TensorRT-based Framework for High Performance Inference with C++/Python Support

- C++ Interface: 3 lines of code is all you need to run a YoloX

  ```C++
  // create inference engine on gpu-0
  //auto engine = Yolo::create_infer("yolov5m.fp32.trtmodel", Yolo::Type::V5, 0);
  auto engine = Yolo::create_infer("yolox_m.fp32.trtmodel", Yolo::Type::X, 0);
  
  // load image
  auto image = cv::imread("1.jpg");
  
  // do inference and get the result
  auto box = engine->commit(image).get();  // return vector<Box>
  ```

- Python Interface:
  ```python
  import trtpy
  
  model     = models.resnet18(True).eval().to(device)
  trt_model = tp.from_torch(model, input)
  trt_out   = trt_model(input)
  ```
  
  - simple yolo for python
  ```python
  import os
  import cv2
  import numpy as np
  import trtpy as tp

  engine_file = "yolov5s.fp32.trtmodel"
  if not os.path.exists(engine_file):
      tp.compile_onnx_to_file(1, tp.onnx_hub("yolov5s"), engine_file)

  yolo   = tp.Yolo(engine_file, type=tp.YoloType.V5)
  image  = cv2.imread("car.jpg")
  bboxes = yolo.commit(image).get()
  print(f"{len(bboxes)} objects")

  for box in bboxes:
      left, top, right, bottom = map(int, [box.left, box.top, box.right, box.bottom])
      cv2.rectangle(image, (left, top), (right, bottom), tp.random_color(box.class_label), 5)

  saveto = "yolov5.car.jpg"
  print(f"Save to {saveto}")

  cv2.imwrite(saveto, image)
  cv2.imshow("result", image)
  cv2.waitKey()
  ```

## INTRO

1. High level interface for C++/Python.
2. Simplify the implementation of custom plugin. And serialization and deserialization have been encapsulated for easier usage.
3. Simplify the compile of fp32, fp16 and int8 for facilitating the deployment with C++/Python in server or embeded device.
4. Models ready for use also with examples are RetinaFace, Scrfd, YoloV5, YoloX, Arcface, AlphaPose, CenterNet and DeepSORT(C++)

## YoloX and YoloV5-series Model Test Report

<details>
<summary>app_yolo.cpp speed testing</summary>
  
1. Resolution (YoloV5P5, YoloX) = (640x640),  (YoloV5P6) = (1280x1280)
2. max batch size = 16
3. preprocessing + inference + postprocessing
4. cuda10.2, cudnn8.2.2.26, TensorRT-8.0.1.6
5. RTX2080Ti
6. num of testing: take the average on the results of 100 times but excluding the first time for warmup 
7. Testing log: [workspace/perf.result.std.log (workspace/perf.result.std.log)
8. code for testing: [src/application/app_yolo.cpp](src/application/app_yolo.cpp)
9. images for testing: 6 images in workspace/inference 
    - with resolution 810x1080，500x806，1024x684，550x676，1280x720，800x533 respetively
10. Testing method: load 6 images. Then do the inference on the 6 images, which will be repeated for 100 times. Note that each image should be preprocessed and postprocessed.

---

| Model    | Resolution | Type      | Precision | Elapsed Time | FPS    |
| -------- | ---------- | --------- | --------- | ------------ | ------ |
| yolox_x  | 640x640    | YoloX     | FP32      | 21.879       | 45.71  |
| yolox_l  | 640x640    | YoloX     | FP32      | 12.308       | 81.25  |
| yolox_m  | 640x640    | YoloX     | FP32      | 6.862        | 145.72 |
| yolox_s  | 640x640    | YoloX     | FP32      | 3.088        | 323.81 |
| yolox_x  | 640x640    | YoloX     | FP16      | 6.763        | 147.86 |
| yolox_l  | 640x640    | YoloX     | FP16      | 3.933        | 254.25 |
| yolox_m  | 640x640    | YoloX     | FP16      | 2.515        | 397.55 |
| yolox_s  | 640x640    | YoloX     | FP16      | 1.362        | 734.48 |
| yolox_x  | 640x640    | YoloX     | INT8      | 4.070        | 245.68 |
| yolox_l  | 640x640    | YoloX     | INT8      | 2.444        | 409.21 |
| yolox_m  | 640x640    | YoloX     | INT8      | 1.730        | 577.98 |
| yolox_s  | 640x640    | YoloX     | INT8      | 1.060        | 943.15 |
| yolov5x6 | 1280x1280  | YoloV5_P6 | FP32      | 68.022       | 14.70  |
| yolov5l6 | 1280x1280  | YoloV5_P6 | FP32      | 37.931       | 26.36  |
| yolov5m6 | 1280x1280  | YoloV5_P6 | FP32      | 20.127       | 49.69  |
| yolov5s6 | 1280x1280  | YoloV5_P6 | FP32      | 8.715        | 114.75 |
| yolov5x  | 640x640    | YoloV5_P5 | FP32      | 18.480       | 54.11  |
| yolov5l  | 640x640    | YoloV5_P5 | FP32      | 10.110       | 98.91  |
| yolov5m  | 640x640    | YoloV5_P5 | FP32      | 5.639        | 177.33 |
| yolov5s  | 640x640    | YoloV5_P5 | FP32      | 2.578        | 387.92 |
| yolov5x6 | 1280x1280  | YoloV5_P6 | FP16      | 20.877       | 47.90  |
| yolov5l6 | 1280x1280  | YoloV5_P6 | FP16      | 10.960       | 91.24  |
| yolov5m6 | 1280x1280  | YoloV5_P6 | FP16      | 7.236        | 138.20 |
| yolov5s6 | 1280x1280  | YoloV5_P6 | FP16      | 3.851        | 259.68 |
| yolov5x  | 640x640    | YoloV5_P5 | FP16      | 5.933        | 168.55 |
| yolov5l  | 640x640    | YoloV5_P5 | FP16      | 3.450        | 289.86 |
| yolov5m  | 640x640    | YoloV5_P5 | FP16      | 2.184        | 457.90 |
| yolov5s  | 640x640    | YoloV5_P5 | FP16      | 1.307        | 765.10 |
| yolov5x6 | 1280x1280  | YoloV5_P6 | INT8      | 12.207       | 81.92  |
| yolov5l6 | 1280x1280  | YoloV5_P6 | INT8      | 7.221        | 138.49 |
| yolov5m6 | 1280x1280  | YoloV5_P6 | INT8      | 5.248        | 190.55 |
| yolov5s6 | 1280x1280  | YoloV5_P6 | INT8      | 3.149        | 317.54 |
| yolov5x  | 640x640    | YoloV5_P5 | INT8      | 3.704        | 269.97 |
| yolov5l  | 640x640    | YoloV5_P5 | INT8      | 2.255        | 443.53 |
| yolov5m  | 640x640    | YoloV5_P5 | INT8      | 1.674        | 597.40 |
| yolov5s  | 640x640    | YoloV5_P5 | INT8      | 1.143        | 874.91 |
</details>

<details>
<summary>app_yolo_fast.cpp speed testing. Never stop desiring for being faster</summary>
  
- <b>Highlight:</b>   0.5 ms faster without any loss in precision compared with the above. Specifically, we remove the Focus and some transpose nodes etc, and implement them in CUDA kenerl function. But the rest remains the same.
- <b>Test log:</b>   [workspace/perf.result.std.log](workspace/perf.result.std.log)
- <b>Code for testing:</b>   [src/application/app_yolo_fast.cpp](src/application/app_yolo_fast.cpp)
- <b>Tips:</b>   you can do the modification while refering to the downloaded onnx. Any questions are welcomed through any kinds of contact.
- <b>Conclusion:</b>   the main idea of this work is to optimize the pre-and-post processing. If you go for yolox, yolov5 small version, the optimization might help you.

|Model|Resolution|Type|Precision|Elapsed Time|FPS|
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

## Setup and Configuration
<details>
<summary>Linux</summary>
  
  
1. VSCode (highly recommended!)
2. Configure your path for cudnn, cuda, tensorRT8.0 and protobuf.
3. Configure the compute capability matched with your nvidia graphics card in Makefile/CMakeLists.txt
    - e.g.  `-gencode=arch=compute_75,code=sm_75`. If you are using 3080Ti, that should be `gencode=arch=compute_86,code=sm_86`
    - reference for the table for GPU Compute Capability:
  https://developer.nvidia.com/cuda-gpus#compute
4. Configure your library path in .vscode/c_cpp_properties.json
5. CUDA version: CUDA10.2
6. CUDNN version: cudnn8.2.2.26. Note that dev(.h file) and runtime(.so file) should be downloaded.
7. tensorRT version：tensorRT-8.0.1.6-cuda10.2
8. protobuf version（for onnx parser）：protobufv3.11.4
    - if other version, refer to the ........
    - link for download: https://github.com/protocolbuffers/protobuf/tree/v3.11.4
    - download, compile and replace the path in Makefile/CMakeLists.txt with new path to protobuf3.11.4
  - CMake:
    - `mkdir build && cd build`
    - `cmake ..`
    - `make yolo -j8`
  - Makefile:
    - `make yolo -j8`
  
</details>

<details>
<summary>Linux: Compile for Python</summary>

- compile and install
    - Makefile：
        - set `use_python := true` in Makefile
    - CMakeLists.txt:
        - `set(HAS_PYTHON ON)` in CMakeLists.txt
    - Type in `make pyinstall -j8`
    - Complied files are in `python/trtpy/libtrtpyc.so`

</details>
  
<details>
<summary>Windows</summary>

  
1. Please check the [lean/README.md](lean/README.md) for the detailed dependency
2. In TensorRT.vcxproj, replace the `<Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 10.0.props" />` with your own CUDA path
3. In TensorRT.vcxproj, replace the `<Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 10.0.targets" />` with your own CUDA path
4. In TensorRT.vcxproj, replace the `<CodeGeneration>compute_61,sm_61</CodeGeneration>` with your compute capability.
    - refer to the table in https://developer.nvidia.com/cuda-gpus#compute
  
5. Configure your dependency or download it to the foler /lean. Configure VC++ dir (include dir and refence)

6. Configure your env, debug->environment
7. Compile and run the example, where 3 options are available.

</details>

<details>
<summary>Windows: Compile for Python</summary>

  
1. Compile trtpyc.pyd. Choose python in visual studio to compile
2. Copy dll and execute 'python/copy_dll_to_trtpy.bat'
3. Execute the example in python dir by 'python test_yolov5.py'
  - if installation is needed, switch to target env(e.g. your conda env) then 'python setup.py install', which has to be followed by step 1 and step 2.
  - the compiled files are in `python/trtpy/libtrtpyc.pyd`

</details>
  
  
<details>
<summary>Other Protobuf Version</summary>
  
- in onnx/make_pb.sh, replace the path `protoc=/data/sxai/lean/protobuf3.11.4/bin/protoc` in protoc with the protoc of your own version

```bash
#cd the path in terminal to /onnx
cd onnx

#execuete the command to make pb files
bash make_pb.sh
```
  
- CMake:
    - replace the `set(PROTOBUF_DIR "/data/sxai/lean/protobuf3.11.4")` in CMakeLists.txt with the same path of your protoc.

```bash
mkdir build && cd build
cmake ..
make yolo -j64
```
- Makefile:
    - replace the path `lean_protobuf  := /data/sxai/lean/protobuf3.11.4` in Makefile with the same path of protoc

```bash
make yolo -j64
```

</details>
  

<details>
<summary>TensorRT 7.x support</summary>

- The default is tensorRT8.x
1. Replace onnx_parser_for_7.x/onnx_parser to src/tensorRT/onnx_parser
    - `bash onnx_parser/use_tensorrt_7.x.sh`
2. Configure Makefile/CMakeLists.txt path to TensorRT7.x
3. Execute `make yolo -j64`

</details>


<details>
<summary>TensorRT 8.x support</summary>

- The default is tensorRT8.x
1. Replace onnx_parser_for_8.x/onnx_parser to src/tensorRT/onnx_parser
    - `bash onnx_parser/use_tensorrt_8.x.sh`
2. Configure Makefile/CMakeLists.txt path to TensorRT8.x
3. Execute `make yolo -j64`

</details>
  
  
## Guide for Different Tasks/Model Support
<details>
<summary>YoloV5 Support</summary>
  
- if pytorch >= 1.7, and the model is 5.0+, the model is suppored by the framework 
- if pytorch < 1.7 or yolov5(2.0, 3.0 or 4.0), minor modification should be done in opset.
- if you want to achieve the inference with lower pytorch, dynamic batchsize and other advanced setting, please check our [blog](http://zifuture.com:8090) (now in Chinese) and scan the QRcode via Wechat to join us.


1. Download yolov5

```bash
git clone git@github.com:ultralytics/yolov5.git
```

2. Modify the code for dynamic batchsize
```python
# line 55 forward function in yolov5/models/yolo.py 
# bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
# x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
# modified into:

bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
bs = -1
ny = int(ny)
nx = int(nx)
x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

# line 70 in yolov5/models/yolo.py
#  z.append(y.view(bs, -1, self.no))
# modified into：
z.append(y.view(bs, self.na * ny * nx, self.no))

############# for yolov5-6.0 #####################
# line 65 in yolov5/models/yolo.py
# if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
#    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
# modified into:
if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

# disconnect for pytorch trace
anchor_grid = (self.anchors[i].clone() * self.stride[i]).view(1, -1, 1, 1, 2)

# line 70 in yolov5/models/yolo.py
# y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
# modified into:
y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid  # wh

# line 73 in yolov5/models/yolo.py
# wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
# modified into:
wh = (y[..., 2:4] * 2) ** 2 * anchor_grid  # wh
############# for yolov5-6.0 #####################


# line 52 in yolov5/export.py
# torch.onnx.export(dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
#                                'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)  修改为
# modified into:
torch.onnx.export(dynamic_axes={'images': {0: 'batch'},  # shape(1,3,640,640)
                                'output': {0: 'batch'}  # shape(1,25200,85) 
```
3. Export to onnx model
```bash
cd yolov5
python export.py --weights=yolov5s.pt --dynamic --include=onnx --opset=11
```
4. Copy the model and execute it
```bash
cp yolov5/yolov5s.onnx tensorRT_cpp/workspace/
cd tensorRT_cpp
make yolo -j32
```

</details>


<details>
<summary>YoloX Support</summary>
  
- download from: https://github.com/Megvii-BaseDetection/YOLOX
- If you don't want to export onnx by yourself, just make run in the repo of Megavii

1. Download YoloX
```bash
git clone git@github.com:Megvii-BaseDetection/YOLOX.git
cd YOLOX
```

2. Modify the code
The modification ensures a successful int8 compilation and inference, otherwise `Missing scale and zero-point for tensor (Unnamed Layer* 686)` will be raised.
  
```Python
# line 206 forward fuction in yolox/models/yolo_head.py. Replace the commented code with the uncommented code
# self.hw = [x.shape[-2:] for x in outputs] 
self.hw = [list(map(int, x.shape[-2:])) for x in outputs]


# line 208 forward function in yolox/models/yolo_head.py. Replace the commented code with the uncommented code
# [batch, n_anchors_all, 85]
# outputs = torch.cat(
#     [x.flatten(start_dim=2) for x in outputs], dim=2
# ).permute(0, 2, 1)
proc_view = lambda x: x.view(-1, int(x.size(1)), int(x.size(2) * x.size(3)))
outputs = torch.cat(
    [proc_view(x) for x in outputs], dim=2
).permute(0, 2, 1)


# line 253 decode_output function in yolox/models/yolo_head.py Replace the commented code with the uncommented code
#outputs[..., :2] = (outputs[..., :2] + grids) * strides
#outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
#return outputs
xy = (outputs[..., :2] + grids) * strides
wh = torch.exp(outputs[..., 2:4]) * strides
return torch.cat((xy, wh, outputs[..., 4:]), dim=-1)

# line 77 in tools/export_onnx.py
model.head.decode_in_inference = True
```

 
3. Export to onnx
```bash

# download model
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth

# export
export PYTHONPATH=$PYTHONPATH:.
python tools/export_onnx.py -c yolox_m.pth -f exps/default/yolox_m.py --output-name=yolox_m.onnx --dynamic --no-onnxsim
```

4. Execute the command
```bash
cp YOLOX/yolox_m.onnx tensorRT_cpp/workspace/
cd tensorRT_cpp
make yolo -j32
```

</details>


<details>
<summary>YoloV3 Support</summary>
  
- if pytorch >= 1.7, and the model is 5.0+, the model is suppored by the framework 
- if pytorch < 1.7 or yolov3, minor modification should be done in opset.
- if you want to achieve the inference with lower pytorch, dynamic batchsize and other advanced setting, please check our [blog](http://zifuture.com:8090) (now in Chinese) and scan the QRcode via Wechat to join us.


1. Download yolov3

```bash
git clone git@github.com:ultralytics/yolov3.git
```

2. Modify the code for dynamic batchsize
```python
# line 55 forward function in yolov3/models/yolo.py 
# bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
# x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
# modified into:

bs, _, ny, nx = map(int, x[i].shape)  # x(bs,255,20,20) to x(bs,3,20,20,85)
bs = -1
x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()


# line 70 in yolov3/models/yolo.py
#  z.append(y.view(bs, -1, self.no))
# modified into：
z.append(y.view(bs, self.na * ny * nx, self.no))

# line 62 in yolov3/models/yolo.py
# if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
#    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
# modified into:
if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
anchor_grid = (self.anchors[i].clone() * self.stride[i]).view(1, -1, 1, 1, 2)

# line 70 in yolov3/models/yolo.py
# y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
# modified into:
y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid  # wh

# line 73 in yolov3/models/yolo.py
# wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
# modified into:
wh = (y[..., 2:4] * 2) ** 2 * anchor_grid  # wh


# line 52 in yolov3/export.py
# torch.onnx.export(dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
#                                'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85) 
# modified into:
torch.onnx.export(dynamic_axes={'images': {0: 'batch'},  # shape(1,3,640,640)
                                'output': {0: 'batch'}  # shape(1,25200,85) 
```
3. Export to onnx model
```bash
cd yolov3
python export.py --weights=yolov3.pt --dynamic --include=onnx --opset=11
```
4. Copy the model and execute it
```bash
cp yolov3/yolov3.onnx tensorRT_cpp/workspace/
cd tensorRT_cpp

# change src/application/app_yolo.cpp: main
# test(Yolo::Type::V3, TRT::Mode::FP32, "yolov3");

make yolo -j32
```

</details>


<details>
<summary>UNet Support</summary>
  
- reference to : https://github.com/shouxieai/unet-pytorch

```
make dunet -j32
```

</details>


<details>
<summary>Retinaface Support</summary>

- https://github.com/biubug6/Pytorch_Retinaface

1. Download Pytorch_Retinaface Repo

```bash
git clone git@github.com:biubug6/Pytorch_Retinaface.git
cd Pytorch_Retinaface
```

2. Download model from the Training of README.md in https://github.com/biubug6/Pytorch_Retinaface#training .Then unzip it to the /weights . Here, we use mobilenet0.25_Final.pth

3. Modify the code

```python
# line 24 in models/retinaface.py
# return out.view(out.shape[0], -1, 2) is modified into 
return out.view(-1, int(out.size(1) * out.size(2) * 2), 2)

# line 35 in models/retinaface.py
# return out.view(out.shape[0], -1, 4) is modified into
return out.view(-1, int(out.size(1) * out.size(2) * 2), 4)

# line 46 in models/retinaface.py
# return out.view(out.shape[0], -1, 10) is modified into
return out.view(-1, int(out.size(1) * out.size(2) * 2), 10)

# The following modification ensures the output of resize node is based on scale rather than shape such that dynamic batch can be achieved.
# line 89 in models/net.py
# up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest") is modified into
up3 = F.interpolate(output3, scale_factor=2, mode="nearest")

# line 93 in models/net.py
# up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest") is modified into
up2 = F.interpolate(output2, scale_factor=2, mode="nearest")

# The following code removes softmax (bug sometimes happens). At the same time, concatenate the output to simplify the decoding.
# line 123 in models/retinaface.py
# if self.phase == 'train':
#     output = (bbox_regressions, classifications, ldm_regressions)
# else:
#     output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
# return output
# the above is modified into:
output = (bbox_regressions, classifications, ldm_regressions)
return torch.cat(output, dim=-1)

# set 'opset_version=11' to ensure a successful export
# torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False,
#     input_names=input_names, output_names=output_names)
# is modified into:
torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False, opset_version=11,
    input_names=input_names, output_names=output_names)




```
4. Export to onnx
```bash
python convert_to_onnx.py
```

5. Execute
```bash
cp FaceDetector.onnx ../tensorRT_cpp/workspace/mb_retinaface.onnx
cd ../tensorRT_cpp
make retinaface -j64
```

</details>


<details>
<summary>DBFace Support</summary>

- https://github.com/dlunion/DBFace

```bash
make dbface -j64
```

</details>

<details>
<summary>Scrfd Support</summary>

- https://github.com/deepinsight/insightface/tree/master/detection/scrfd
- The know-how about exporting to onnx is comming. Before it is released, come and join us to disucss. 

</details>



<details>
<summary>Arcface Support</summary>

- https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
```C++
auto arcface = Arcface::create_infer("arcface_iresnet50.fp32.trtmodel", 0);
auto feature = arcface->commit(make_tuple(face, landmarks)).get();
cout << feature << endl;  // 1x512
```
- In the example of Face Recognition, `workspace/face/library` is the set of faces registered.
- `workspace/face/recognize` is the set of face to be recognized.
- the result is saved in `workspace/face/result`和`workspace/face/library_draw`

</details>
  
<details>
<summary>CenterNet Support</summary>
  
check the great details in tutorial/2.0
</details>


<details>
<summary>Bert Support(Chinese Classification)</summary>

- https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch
- `make bert -j6`  

</details>


## the INTRO to Interface

<details>
<summary>Python Interface：Get onnx and trtmodel from pytorch model more easily</summary>

- Just one line of code to export onnx and trtmodel. And save them for usage in the future.
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
<summary>Python Interface：TensorRT Inference</summary>

- YoloX TensorRT Inference
```python
import trtpy

yolo   = tp.Yolo(engine_file, type=tp.YoloType.X)   # engine_file is the trtmodel file
image  = cv2.imread("inference/car.jpg")
bboxes = yolo.commit(image).get()
```

- Seamless Inference from Pytorch to TensorRT
```python
import trtpy

model     = models.resnet18(True).eval().to(device) # pt model
trt_model = tp.from_torch(model, input)
trt_out   = trt_model(input)
```

</details>


<details>
<summary>C++ Interface：YoloX Inference</summary>

```C++

// create infer engine on gpu 0
auto engine = Yolo::create_infer("yolox_m.fp32.trtmodel"， Yolo::Type::X, 0);

// load image
auto image = cv::imread("1.jpg");

// do inference and get the result
auto box = engine->commit(image).get();
```

</details>


<details>
<summary>C++ Interface：Compile Model in FP32/FP16</summary>

```cpp
TRT::compile(
  TRT::Mode::FP32,   // compile model in fp32
  3,                          // max batch size
  "plugin.onnx",              // onnx file
  "plugin.fp32.trtmodel",     // save path
  {}                         //  redefine the shape of input when needed
);
```
- For fp32 compilation, all you need is offering onnx file whose input shape is allowed to be redefined.
</details>


<details>
<summary>C++ Interface：Compile in int8</summary>

- The in8 inference performs slightly worse than fp32 in precision(about -5% drop down), but stunningly faster. In the framework, we offer int8 inference

```cpp
// define int8 calibration function to read data and handle it to tenor.
auto int8process = [](int current, int count, vector<string>& images, shared_ptr<TRT::Tensor>& tensor){
    for(int i = 0; i < images.size(); ++i){
    // int8 compilation requires calibration. We read image data and set_norm_mat. Then the data will be transfered into the tensor.
        auto image = cv::imread(images[i]);
        cv::resize(image, image, cv::Size(640, 640));
        float mean[] = {0, 0, 0};
        float std[]  = {1, 1, 1};
        tensor->set_norm_mat(i, image, mean, std);
    }
};


// Specify TRT::Mode as INT8
auto model_file = "yolov5m.int8.trtmodel";
TRT::compile(
  TRT::Mode::INT8,            // INT8
  3,                          // max batch size
  "yolov5m.onnx",             // onnx
  model_file,                 // saved filename
  {},                         // redefine the input shape
  int8process,                // the recall function for calibration
  ".",                        // the dir where the image data is used for calibration
  ""                          // the dir where the data generated from calibration is saved(a.k.a where to load the calibration data.)
);
```
- We integrate into only one int8process function to save otherwise a lot of issues that might happen in tensorRT official implementation. 

</details>


<details>
<summary>C++ Interface：Inference</summary>

- We introduce class Tensor for easier inference and data transfer between host to device. So that as a user, the details wouldn't be annoying.

- class Engine is another facilitator.

```cpp
// load model and get a shared_ptr. get nullptr if fail to load.
auto engine = TRT::load_infer("yolov5m.fp32.trtmodel");

// print model info
engine->print();

// load image
auto image = imread("demo.jpg");

// get the model input and output node, which can be accessed by name or index
auto input = engine->input(0);   // or auto input = engine->input("images");
auto output = engine->output(0); // or auto output = engine->output("output");

// put the image into input tensor by calling set_norm_mat()
float mean[] = {0, 0, 0};
float std[]  = {1, 1, 1};
input->set_norm_mat(i, image, mean, std);

// do the inference. Here sync(true) or async(false) is optional
engine->forward(); // engine->forward(true or false)

// get the outut_ptr, which can used to access the output
float* output_ptr = output->cpu<float>();
```

</details>


<details>
<summary>C++ Interface：Plugin</summary>

- You only need to define kernel function and inference process. The details of code(e.g the serialization, deserialization and injection of plugin etc) are under the hood.
- Easy to implement a new plugin in FP32 and FP16. Refer to HSwish.cu for details.
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


## About Us
- Our blog：http://www.zifuture.com/                        (Now only in Chinese. English is comming)
- Our video channel： https://space.bilibili.com/1413433465 (Now only in Chinese. English is comming)










