
import torch
import torchvision.models as models
import pytrt as tp
import numpy as np
import os

os.chdir("../workspace/")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 基于torch的tensor输入
input     = torch.full((5, 3, 224, 224), 0.3).to(device)
model     = models.resnet18(True).eval().to(device)
trt_model = tp.from_torch(model, input, 
    engine_save_file   = "torch.engine.trtmodel", 
    onnx_save_file     = "torch.onnx"
)

torch_out = model(input)
trt_out   = trt_model(input)

trt_model.save("torch.trtmodel")

abs_diff = (torch_out - trt_out).abs().max()
print(f"Torch and TRTModel abs diff is {abs_diff}")

print(f"trt_model.stream is {trt_model.stream}")
print(trt_model.input().shape)
trt_model.input().resize_single_dim(0, 1)
print(trt_model.input().shape)
trt_model.input().resize_single_dim(0, 5)

# 获取模型的input，并对输入进行赋值为0.5
trt_model.input().numpy[:] = 0.5

# 执行推理
trt_model.forward()

# 获取输出
trt_out = trt_model.output().numpy

#对torch进行推理
input[:] = 0.5
torch_out = model(input).cpu().data.numpy()

# 对比差距绝对值
abs_diff = np.abs(torch_out - trt_out).max()
print(f"Torch and TRTModel abs diff is {abs_diff}")