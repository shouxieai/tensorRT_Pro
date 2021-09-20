
*Read this in other languages:[English](README.md), [简体中文](README.zh-cn.md).*

# INTRO
The tutorial is to demonstrate how to export a pytorch model to trt model. We use CenterNet as the example. Because DCNv2 is applied in CenterNet, which exemplify how to manipulate a plugin.

Two parts are previewed for a successful export.
- 1.python part
- 2.c++ part

# PYTHON PART

The CenterNet repo is downloaded from https://github.com/xingyizhou/CenterNet
But /CenterNet/src/lib/models/networks/DCNv2 is replaced by the one from https://github.com/jinfagang/DCNv2_latest due to version compatibility.

For the purpose of learning the export, in CenterNet, we only care about the two files(src/demo.py and its related files and src/export2onnx.py). The ones who are interested in CenterNet itself could refer to the paper or the official github repo.

Now let's open the export2onnx.py
I rewrite a export script instead of using the official one such that it's easier to adapt to our tensorRT_cpp framwork later.

Usually whenever you get a pt model, you want to export to a onnx. The following steps are recommened to follow:
- 1. extract the structure to a new script to write a new onnx script or use the official export script.(I did the former)

- 2. then use torch.onnx.export rather than export_. Before filling in args, we also need to modify our model structure for an eaisier decoding. We will be right back here for a second.(Read 3 firstly, then go back here). We fill in the input_names with ["images"] and output_names with ["output"]. Note that the two names are predefined in our tensorRT_cpp framework. You can also keep other outputs like the commented code in the script, but doesn't make much sense. Also remember to set enable_onnx_checker=False.

        - 5_outputs.jpg and all_in_one_output.jpg is the onnx screenshot.



- 3. open CenterNet/src/lib/models/networks/resnet_dcn.py . Skip to about 280 line where 'def forward(self, x): '

    - #!+ indicates the addition
    - modify self.heads = {"reg":2, "wh":2, "hm":80} 
    - and add a pool_hm head diverged from the hm head (<b>highlight</b>)
    - concat all output/heads into one output for an easier decoding in c++ env
    - we also do permutation to put the 164(2+2+80+80) in the end.

- 4. It seems everything is ready. But not yet if you forget to modify CenterNet/src/lib/models/networks/DCNv2/dcn_v2.py . Comment the original version and use the ONNX CONVERSION. The ONNX CONVERSION is written for being compatible to our tensorRT_cpp framework.


Well done if you are here. Now you might get a onnx file. You copy it to the tensorRT_cpp/workspace. Then you are well done in python part.



# C++ PART
DCN plugin CUDA implementation has already been implemented by us in src/tensorRT/onnxplugin/plugins/DCNv2.cu . You don't need to worry about it. But if you want to know the details, refer to P5 https://www.bilibili.com/video/BV1Xw411f7FW?p=2&spm_id_from=pageDriver (Now only in Chinese. English video tutorial is comming. Push us to release if you desire for it.)

Then it's the climax during the whole export time -- write CUDA kernel for preprocessing and postprocessing. But for CenterNet, it shares the same preprocessing with yolo. So no modification is needed. 

(We also offer a detailed code to enable the beginners to experience the whole process, even for those with limited c++ and cuda experience.)

## Files for CenterNet and Points worth Noting
centernet_decode.cu centernet.cpp centernet.hpp and app_centernet.cpp

Most of codes don't need to be modified except that the following are worth noting:
- mean and std follow the CenterNet official repo, different from the yolo
- num_channels variable is used instead of just num_classes in yolo.
- downsampling ratio should be specified
- confidence_threshold is lower than yolo due to the usage of Focal loss
- in preprocessing, the border value is 0 by default, which is the same as the CenterNet official repo.

More focus should be on post processing for CenterNet.


## Post processing for CenterNet

open tensorRT_cpp/src/application/app_centernet/centernet_decode.cu

If you are beginner, go for the path 'from_python_through_c++_to_cuda_impl'(in folder 0_to_1_python_to_cuda). If not, just keep reading and get familiar with the centernet_decode.cu . And now you are done well.





























