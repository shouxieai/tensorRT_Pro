## TensorRT8.0，C++封装库最新发布
1. 支持最新版tensorRT8.0，具有最新的解析器算子支持
2. 支持静态显性batch size，和动态非显性batch size，这是官方所不支持的
3. 支持自定义插件，简化插件的实现过程
4. 支持fp32、fp16、int8的编译
5. 优化代码结构，打印编译网络信息
6. 优化内存分配
7. yolov5的推理作为案例
8. c++类库，对编译和推理做了封装，对tensor做了封装，支持n维的tensor管理

## YoloV5-ONNX推理支持
- yolov5的onnx，你的pytorch版本>=1.7时，导出的onnx模型可以直接被当前框架所使用
- 你的pytorch版本低于1.7时，或者对于yolov5其他版本（2.0、3.0、4.0），可以对opset进行简单改动后直接被框架所支持
1. 下载yolov5
```bash
git clone git@github.com:ultralytics/yolov5.git
```
2. 导出onnx模型
```bash
cd yolov5
python export.py
```
3. 复制模型并执行
```bash
cp yolov5/yolov5s.onnx tensorRT_cpp/workspace/
make run -j32
```

- 下面是效果图workspace/2.draw.jpg
---
<img src="workspace/2.draw.jpg" width="300px" />

---

## 项目的配置
1. 推荐使用Linux、VSCode，当然也可以支持windows
2. 在Makefile中配置你的cudnn、cuda、tensorRT8.0、protobuf路径
3. 在.vscode/c_cpp_properties.json中配置你的库路径
3. CUDA版本：CUDA10.2
4. CUDNN版本：cudnn8.2.2.26，注意下载dev（h文件）和runtime（so文件）
5. tensorRT版本：tensorRT-8.0.1.6-cuda10.2
6. protobuf版本（用于onnx解析器）：这里使用的是protobufv3.11.4
    - 下载地址：https://github.com/protocolbuffers/protobuf/tree/v3.11.4


## 模型编译-FP32/16
```cpp
TRTBuilder::compile(
  TRTBuilder::TRTMode_FP32,   // 使用fp32模型编译
  {},                         // caffe时指定输出节点
  3,                          // max batch size
  "plugin.onnx",              // onnx 文件
  "plugin.fp32.trtmodel",     // 保存的文件路径
  {},                         // 重新定制输入的shape
  false                       // 是否动态batch size
);
```
- 对于FP32编译，只需要提供onnx文件即可，可以允许重定义onnx输入节点的shape
- 对于动态或者静态batch的支持，仅仅只需要一个选项，这对于官方发布的解析器是不支持的

## 模型编译-INT8
- 众所周知，int8的推理效果比fp32稍微差一点（预计-5%的损失），但是速度确快很多很多，这里通过集成的编译方式，很容易实现int8的编译工作
```cpp
// 定义int8的标定数据处理函数，读取数据并交给tensor的函数
auto int8process = [](int current, int count, vector<string>& images, shared_ptr<TRTInfer::Tensor>& tensor){
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
auto model_file = "yolov5s.int8.trtmodel";
TRTBuilder::compile(
  TRTBuilder::TRTMode_INT8,   // 选择INT8
  {},                         // 对于caffe的输出节点名称
  3,                          // max batch size
  "yolov5s.onnx",             // onnx文件
  model_file,                 // 编译后保存的文件
  {},                         // 重定义输入的shape
  false,                      // 是否为动态batch size
  int8process,                // 指定标定数据的处理回调函数
  ".",                        // 指定标定图像数据的目录
  ""                          // 指定标定后的数据储存/读取路径
);
```
- 避免了官方标定流程分离的问题，复杂度太高，在这里直接集成为一个函数处理

## 模型推理 
- 对于模型推理，封装了Tensor类，实现推理的维护和数据交互，对于数据从GPU到CPU过程完全隐藏细节
- 封装了Engine类，实现模型推理和管理
```cpp
// 模型加载，得到一个共享指针，如果为空表示加载失败
auto engine = TRTInfer::load_engine("yolov5s.fp32.trtmodel");

// 打印模型信息
engine->print();

// 加载图像
auto image = imread("demo.jpg");

// 获取模型的输入和输出tensor节点，可以根据名字或者索引获取第几个
auto input = engine->input(0);
auto output = engine->output(0);

// 把图像塞到input tensor中，这里是减去均值，除以标准差
float mean[] = {0, 0, 0};
float std[]  = {1, 1, 1};
input->set_norm_mat(i, image, mean, std);

// 执行模型的推理，这里可以允许异步或者同步
engine->forward();

// 这里拿到的指针即是最终的结果指针，可以进行访问操作
float* output_ptr = output->cpu<float>();
// 这里对output_ptr进行处理即可得到结果
```

## 一个插件的例子
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
    auto grid = cuda::grid_dims(count);
    auto block = cuda::block_dims(count);
    HSwishKernel <<<grid, block, 0, stream >>> (inputs[0].ptr<float>(), outputs[0].ptr<float>(), count);
    return 0;
}

RegisterPlugin(HSwish);
```

## 执行方式
1. 配置好Makefile中的依赖项路径
2. `make run -j64`即可

## 执行结果
```bash
[2021-07-22 14:37:11][info][_main.cpp:160]:===================== test fp32 ==================================
[2021-07-22 14:37:11][info][trt_builder.cpp:430]:Compile FP32 Onnx Model 'yolov5s.onnx'.
[2021-07-22 14:37:18][warn][trt_infer.cpp:27]:NVInfer WARNING: src/tensorRT/onnx_parser/ModelImporter.cpp:257: Change input batch size: images, final dimensions: (1, 3, 640, 640), origin dimensions: (5, 3, 640, 640)
[2021-07-22 14:37:18][info][trt_builder.cpp:548]:Input shape is 1 x 3 x 640 x 640
[2021-07-22 14:37:18][info][trt_builder.cpp:549]:Set max batch size = 3
[2021-07-22 14:37:18][info][trt_builder.cpp:550]:Set max workspace size = 1024.00 MB
[2021-07-22 14:37:18][info][trt_builder.cpp:551]:Dynamic batch dimension is true
[2021-07-22 14:37:18][info][trt_builder.cpp:554]:Network has 1 inputs:
[2021-07-22 14:37:18][info][trt_builder.cpp:560]:      0.[images] shape is 1 x 3 x 640 x 640
[2021-07-22 14:37:18][info][trt_builder.cpp:566]:Network has 3 outputs:
[2021-07-22 14:37:18][info][trt_builder.cpp:571]:      0.[470] shape is 1 x 255 x 80 x 80
[2021-07-22 14:37:18][info][trt_builder.cpp:571]:      1.[471] shape is 1 x 255 x 40 x 40
[2021-07-22 14:37:18][info][trt_builder.cpp:571]:      2.[472] shape is 1 x 255 x 20 x 20
[2021-07-22 14:37:18][verbo][trt_builder.cpp:575]:Network has 226 layers:
[2021-07-22 14:37:18][verbo][trt_builder.cpp:606]:  >>> 0.  Slice              1 x 3 x 640 x 640 -> 1 x 3 x 320 x 640 
[2021-07-22 14:37:18][verbo][trt_builder.cpp:606]:      1.  Slice              1 x 3 x 320 x 640 -> 1 x 3 x 320 x 320 
[2021-07-22 14:37:18][verbo][trt_builder.cpp:606]:  >>> 2.  Slice              1 x 3 x 640 x 640 -> 1 x 3 x 320 x 640 
[2021-07-22 14:37:18][verbo][trt_builder.cpp:606]:      3.  Slice              1 x 3 x 320 x 640 -> 1 x 3 x 320 x 320 
[2021-07-22 14:37:18][verbo][trt_builder.cpp:606]:  >>> 4.  Slice              1 x 3 x 640 x 640 -> 1 x 3 x 320 x 640 
[2021-07-22 14:37:18][verbo][trt_builder.cpp:606]:      5.  Slice              1 x 3 x 320 x 640 -> 1 x 3 x 320 x 320 
[2021-07-22 14:37:18][verbo][trt_builder.cpp:606]:  >>> 6.  Slice              1 x 3 x 640 x 640 -> 1 x 3 x 320 x 640 
[2021-07-22 14:37:18][verbo][trt_builder.cpp:606]:      7.  Slice              1 x 3 x 320 x 640 -> 1 x 3 x 320 x 320
[2021-07-22 14:37:18][verbo][trt_builder.cpp:606]:      222.LeakyRelu          1 x 768 x 20 x 20 -> 1 x 768 x 20 x 20 
[2021-07-22 14:37:18][verbo][trt_builder.cpp:606]:  *** 223.Convolution        1 x 192 x 80 x 80 -> 1 x 255 x 80 x 80 channel: 255, kernel: 1 x 1, padding: 0 x 0, stride: 1 x 1, dilation: 1 x 1, group: 1
[2021-07-22 14:37:18][verbo][trt_builder.cpp:606]:  *** 224.Convolution        1 x 384 x 40 x 40 -> 1 x 255 x 40 x 40 channel: 255, kernel: 1 x 1, padding: 0 x 0, stride: 1 x 1, dilation: 1 x 1, group: 1
[2021-07-22 14:37:18][verbo][trt_builder.cpp:606]:  *** 225.Convolution        1 x 768 x 20 x 20 -> 1 x 255 x 20 x 20 channel: 255, kernel: 1 x 1, padding: 0 x 0, stride: 1 x 1, dilation: 1 x 1, group: 1
[2021-07-22 14:37:18][info][trt_builder.cpp:615]:Building engine...
[2021-07-22 14:37:19][warn][trt_infer.cpp:27]:NVInfer WARNING: Detected invalid timing cache, setup a local cache instead
[2021-07-22 14:37:40][info][trt_builder.cpp:635]:Build done 22344 ms !
Engine 0x23dd7780 detail
        Max Batch Size: 3
        Dynamic Batch Dimension: true
        Inputs: 1
                0.images : shape {1 x 3 x 640 x 640}
        Outputs: 3
                0.470 : shape {1 x 255 x 80 x 80}
                1.471 : shape {1 x 255 x 40 x 40}
                2.472 : shape {1 x 255 x 20 x 20}
[2021-07-22 14:37:42][info][_main.cpp:77]:input.shape = 3 x 3 x 640 x 640
[2021-07-22 14:37:42][info][_main.cpp:96]:input->shape_string() = 3 x 3 x 640 x 640
[2021-07-22 14:37:42][info][_main.cpp:124]:outputs[0].size = 2
[2021-07-22 14:37:42][info][_main.cpp:124]:outputs[1].size = 5
[2021-07-22 14:37:42][info][_main.cpp:124]:outputs[2].size = 1
```
