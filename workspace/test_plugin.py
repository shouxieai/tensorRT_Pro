import torch
import torch.nn.functional as F
import torch.nn as nn
import json

class HSwishImplementation(torch.autograd.Function):

    # 主要是这里，对于autograd.Function这种自定义实现的op，只需要添加静态方法symbolic即可，除了g以外的参数应与forward函数的除ctx以外完全一样
    # 这里演示了input->作为tensor输入，bias->作为参数输入，两者将会在tensorRT里面具有不同的处理方式
    # 对于附加属性（attributes），以 "名称_类型简写" 方式定义，类型简写，请参考：torch/onnx/symbolic_helper.py中_parse_arg函数的实现【from torch.onnx.symbolic_helper import _parse_arg】
    # 属性的定义会在对应节点生成attributes，并传给tensorRT的onnx解析器做处理
    @staticmethod
    def symbolic(g, input, bias):
        return g.op("Plugin", input, bias, name_s="HSwish", info_s=json.dumps({"alpha": 3.5, "beta": 2.88}))

    @staticmethod
    def forward(ctx, i, bias):
        ctx.save_for_backward(i)
        return i * F.relu6(i + 3) / 6
    
     # 这里省略了backward

class MemoryEfficientHSwish(nn.Module):
    def __init__(self):
        super(MemoryEfficientHSwish, self).__init__()
        
        # 这里我们假设有bias作为权重参数
        self.bias = nn.Parameter(torch.zeros((5, 3, 3, 1)))
        self.bias.data.fill_(3.15)

    def forward(self, x):
        # 我们假设丢一个bias进去
        return HSwishImplementation.apply(x, self.bias)

class FooModel(torch.nn.Module):
    def __init__(self):
        super(FooModel, self).__init__()
        self.hswish = MemoryEfficientHSwish()

    def forward(self, input1, input2):
        return input2 + self.hswish(input1)

dummy_input1 = torch.zeros((1, 3, 3, 3))
dummy_input2 = torch.zeros((1, 1, 3, 3))
model = FooModel()

# 这里演示了2个输入的情况，实际上你可以自己定义几个输入
torch.onnx.export(model, (dummy_input1, dummy_input2), 'plugin.onnx', verbose=True, opset_version=11)