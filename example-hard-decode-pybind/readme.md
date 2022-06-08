# 硬件解码配合TensorRT
1. 配置tensorRT一样的环境
2. 增加NVDEC和ffmpeg的配置
- `make run -j64`
    - 执行python的test.py并调用硬件解码
- `make runpro -j64`
    - 执行c++程序进行硬件解码
- `make runhdd -j64`
    - 执行python的test_hard_decode_yolov5.py进行tensorRT推理并对接硬件解码
3. 软解码和硬解码，分别消耗cpu和gpu资源。在多路，大分辨率下体现明显
4. 硬件解码和推理可以允许跨显卡
5. 理解并善于利用的时候，他才可能发挥最大的效果

# 使用
1. 为nvcuvid创建软链接，这个库随显卡驱动发布
    - `bash link-cuvid.sh`
2. ffmpeg使用提供好的，或者自行编译普通cpu版本即可，不需要配置ffmpeg的cuda支持
3. 在tensorRT_Pro目录下，编译python的支持，执行`make pyinstall -j64`，并安装pytrt
4. 执行python接口的硬件解码`make run -j64`
5. 执行cpp接口的硬件解码`make runpro -j64`
6. 执行tensorRT推理对接硬件解码`make runhdd -j64`

# 如果要在目录下执行
- 请先将编译依赖的library path添加到PATH中，然后python test.py即可