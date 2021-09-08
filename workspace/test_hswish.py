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
        # 如果配合当前tensorRT框架，则必须名称为Plugin，参考：tensorRT/src/tensorRT/onnx_parser/builtin_op_importers.cpp的160行定义
        # 若你想自己命名，可以考虑做类似修改即可
        #
        # name_s表示，name是string类型的，对应于C++插件的名称，参考：tensorRT/src/tensorRT/onnxplugin/plugins/HSwish.cu的82行定义的名称
        # info_s表示，info是string类型的，通常我们可以利用json.dumps，传一个复杂的字符串结构，然后在CPP中json解码即可。参考：
        #             sxai/tensorRT/src/tensorRT/onnxplugin/plugins/HSwish.cu的39行
        return g.op("Plugin", input, bias, name_s="HSwish", info_s=json.dumps({"alpha": 3.5, "beta": 2.88}))

    # 这里的forward只是为了让onnx导出时可以执行，实际上写与不写意义不大，只需要返回同等的输出维度即可
    @staticmethod
    def forward(ctx, i, bias):
        ctx.save_for_backward(i)
        return i * F.relu6(i + 3) / 6
    
     # 这里省略了backward

class MemoryEfficientHSwish(nn.Module):
    def __init__(self):
        super(MemoryEfficientHSwish, self).__init__()
        
        # 这里我们假设有bias作为权重参数
        self.bias = nn.Parameter(torch.zeros((3, 3, 3, 3)))
        self.bias.data.fill_(3.15)

    def forward(self, x):
        # 我们假设丢一个bias进去
        return HSwishImplementation.apply(x, self.bias)

class FooModel(torch.nn.Module):
    def __init__(self):
        super(FooModel, self).__init__()
        self.hswish = MemoryEfficientHSwish()

    def forward(self, input1, input2):
        return F.relu(input2 * self.hswish(input1))

dummy_input1 = torch.zeros((1, 3, 3, 3))
dummy_input2 = torch.zeros((1, 3, 3, 3))
model = FooModel()

# 这里演示了2个输入的情况，实际上你可以自己定义几个输入
torch.onnx.export(
    model, 
    (dummy_input1, dummy_input2), 
    'hswish.plugin.onnx', 
    input_names=["input.0", "input.1"],
    output_names=["output.0"], 
    verbose=True, 
    opset_version=11,
    dynamic_axes={"input.0": {0:"batch"}, "input.1": {0:"batch"}, "output.0": {0:"batch"}},
    enable_onnx_checker=False
)
print("Done")