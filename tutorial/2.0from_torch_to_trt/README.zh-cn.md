*[English](README.md), [简体中文](README.zh-cn.md).*
*英文版为推荐版本*

# 介绍 
本教程将演示如何导出一个pytorch模型到trt模型。我们以CenterNet为例。因为DCNv2应用在CenterNet中，它演示了如何操作插件。
先预览两部分:
- 1. python的部分
- 2. c++的部分

# PYTHON部分
CenterNet repo 可从https://github.com/xingyizhou/CenterNet下载 但是由于版本兼容性问题，/CenterNet/src/lib/models/networks/DCNv2 要用这个 https://github.com/jinfagang/DCNv2_latest 来替换了。
考虑这是教程，在CenterNet中，我们只关心这两个文件(src/demo.py及其相关文件和src/export2onnx.py)。那些对CenterNet本身感兴趣的小伙伴可以参考官方的github repo。

现在让我们打开 export2onnx.py
我重写了一个导出脚本，而不是使用官方脚本，这样以后更容易适应我们的tensorRT_cpp框架。
通常当你得到一个pt模型时，你想要导出到onnx。建议采取以下步骤:
- 1. 将模型结构提取到一个新的脚本，以编写一个新的onnx脚本或使用官方的导出脚本。(我自己是重新写了一个)
- 2. 然后用torch.onnx.export而不是export_。在填参数之前，我们还需要修改我们的模型结构，以便更容易解码。(首先阅读3，然后回到这里)。我们用["images"]填充input_names，用["output"]填充output_names。注意，这两个名称是在tensorRT_cpp框架中预定义的。您也可以在脚本中保留其他输出，就像我在注释代码中做的，但这没有多大意义。还要记住设置enable_onnx_checker=False。

5_outputs.jpg和all_in_one_output.jpg是onnx的截图。

- 3. 打开CenterNet/src/lib/model/network/resnet_dcn.py 跳到大约280行，其中'def forward(self, x): '
    - /# !+表示加了的东西
    - 修改self.head = {"reg":2， "wh":2， "hm":80}
    - 并添加一个从hm head 分支出来的 pool_hm(<b>在这里操作的话就不用用c++/cuda来写pool了</b>)

    - 将所有head concat到一个输出，以便在c++ env中更容易解码
    - 我们也做了permute，把164(2+2+80+80)放在最后。
- 4. 似乎一切都准备好了。但如果你忘了修改CenterNet/src/lib/models/networks/DCNv2/dcn_v2.py，那就不行了。你需要注释Original version 并使用ONNX CONVERSION。ONNX CONVERSION是为了兼容tensorRT_cpp框架而编写的。
如果你已经读到了这里，做得很好。现在你可能会得到一个onnx文件。将其复制到tensorRT_cpp/工作区。那么你在python部分已经大功告成了。


# c++的部分
DCN插件已经由我们在src/tensorRT/onnxplugin/plugins/DCNv2中实现了。你不需要为此担心。但如果你想知道细节，请参阅P5 https://www.bilibili.com/video/BV1Xw411f7FW?p=2&spm_id_from=pageDriver (现在只有中文)。

然后是整个导出过程的高潮 ---- 编写CUDA内核进行预处理和后处理。但对于CenterNet来说，它与yolo有着相同的预处理。所以几乎不需要修改。
(我们还提供了详细的代码，让初学者能够体验整个过程，即使是那些对c++和cuda经验有限的人。)

#### CenterNet的文件和值得注意的点
centernet_decode.cu centernet.cpp centernet.hpp和app_centernet.cpp

大多数代码不需要修改，除了以下几点值得注意:
- mean和std遵循CenterNet的官方回购，与yolo不同
- 在yolo中使用了num_channels变量而不是num_classes。
- 指定下采样率
- 由于使用了Focal loss，导致confence_threshold低于yolo
- 在预处理时，默认值为0，与CenterNet官方回购相同。
- 应该更多地关注CenterNet的后期处理。

#### CenterNet 的后期处理
打开 tensorRT_cpp/src/application/app_centernet/centernet_decode.cu

如果你是初学者，可以选择'from_python_through_c++_to_cuda_impl'的学习路径(在0_to_1_python_to_cuda文件夹中)。如果没有，请继续阅读并熟悉centernet_decode.cu。结束了，就这么多。