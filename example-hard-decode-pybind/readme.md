# 硬件解码配合TensorRT
- 配置tensorRT一样的环境
- 增加NVDEC和ffmpeg的配置
- `make run -j64`
    - 执行python的test.py并调用硬件解码
- `make runpro -j64`
    - 执行c++程序进行硬件解码
- `make runhdd -j64`
    - 执行python的test_hard_decode_yolov5.py进行tensorRT推理并对接硬件解码
- 软解码和硬解码，分别消耗cpu和gpu资源。在多路，大分辨率下体现明显
- 硬件解码和推理可以允许跨显卡
- 理解并善于利用的时候，他才可能发挥最大的效果

# 使用
1. 为nvcuvid创建软链接，这个库随显卡驱动发布
    - `bash link-cuvid.sh`
2. 执行python接口的硬件解码`make run -j64`
3. 执行cpp接口的硬件解码`make runpro -j64`
4. 执行tensorRT推理对接硬件解码`make runhdd -j64`

# 如果要在目录下执行
- 请先将编译依赖的library path添加到PATH中，然后python test.py即可