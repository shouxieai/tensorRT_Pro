# Simple YoloV5/YoloX
- 更加容易集成的YoloV5/YoloX实现
- 仅依赖官方的tensorRT，无第三方依赖，也没有复杂的封装
- 只有hpp和cu两个文件

# YoloV5
- 请在`export.py`中修改onnx导出时dynamic维度只有batch，去掉images部分的width和height，以及output部分的anchors，只保留batch: -1
- 然后执行跟主项目一样的修改方式
- 再执行导出onnx，并启用--dynamic选项，使得导出的onnx时动态batch。至此导出的onnx即可被这个项目接受
- 如果失败，请对比自动下载的onnx的头部和结尾部分是否一样，保证一样后才能推理正常
- YoloV5的导出命令：
```bash
python export.py --imgsz=640 --weights=yolov5s.pt --include=onnx --dynamic
```

# YoloX
- 请按照主项目一样的要求修改yolox代码
- 然后执行导出指令，加上dynamic选项，如下：
```bash
#这一句使得当前yolox能够找到
export PYTHONPATH=$PYTHONPATH:.
python tools/export_onnx.py --output-name=yolox_s.onnx --no-onnxsim --exp_file=exps/default/yolox_s.py --ckpt=yolox_s.pth --dynamic
```
- 如果失败，请对比自动下载的onnx是否与你导出的一致，不一样不能保证推理结果正常

# 运行
- 按照主项目配置Makefile或者CMakeLists.txt，然后`make run -j64`即可

# 关于Reisze报错
- 如果你是7.x，通常会提示onnx解析的resize无法解析报错，此时考虑用workspace/fix_trt7.0_resize.py把resize改为upsample即可正常运行