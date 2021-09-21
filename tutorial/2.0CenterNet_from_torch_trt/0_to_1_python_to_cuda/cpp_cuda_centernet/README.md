- Download model1 for tutorial: https://drive.google.com/file/d/1H5TI6aiWnleYfXjY3l2-VaxXr6evRfre/view?usp=sharing
- Download model2 for tutorial: https://drive.google.com/file/d/1QNsBDSlXyTtbO0VoZ4HPGnOb1mRHcciL/view?usp=sharing


cd to tensorRT_cpp/tutorial/2.0CenterNet_from_torch_trt/0_to_1_python_to_cuda/cpp_cuda_centernet, 

Copy tensorRT_cpp/src/TensorRT to /0_to_1_python_to_cuda/cpp_cuda_centernet/src/. Also remember to modify paths.

Then set the workspace as /cpp_cuda_centernet by 
opening folder in tensorRT_cpp/tutorial/2.0CenterNet_from_torch_trt/0_to_1_python_to_cuda/cpp_cuda_centernet in vscode.

In app_centernet.cpp, a pure c++ implementation and a pure cuda implementation are offered for tutorial. They are not neat because I prefer to keep the draft(the commented code) for guys to learn the API operations in our framework/c++/cuda env. Some codes are hardcoded for simplicity.It is suggested that after being familiar with python, c++ and cuda version, you are free to have a try on integrating CenterNet into our framework like what we've done in /tensorRT_cpp/src/application/app_centernet.

<hr/>

- 下载用于教程的模型1: https://pan.baidu.com/s/1Tj3EhxOOQgexPhuw74QtUA    提取码：75o3
- 下载用于教程的模型2: https://pan.baidu.com/s/1UEq2n0Kn5jd2n3ahU-sn3w    提取码：aal6

cd to tensorRT_cpp/tutorial/2.0CenterNet_from_torch_trt/0_to_1_python_to_cuda/cpp_cuda_centernet, 
Copy tensorRT_cpp/src/TensorRT to /0_to_1_python_to_cuda/cpp_cuda_centernet/src/. 记住修改成自己的路径

然后将工作区设置为/cpp_cuda_centernet:
在vscode中打开tensorRT_cpp/tutorial/2.0CenterNet_from_torch_trt/0_to_1_python_to_cuda/cpp_cuda_centernet文件夹。

在app_centernet.cpp中，我们提供了一个纯c++实现和一个纯cuda实现的教程。它们主要复现了一些思路，所以并非十分整洁，因为我们更希望保留草稿(例如：注释掉的代码)，以便小伙伴们在阅读草稿时可以感受到我们的框架/c++/cuda 环境中API操作的各种使用可能性。为了简单起见，有些代码是硬编码的。建议您在熟悉python、c++和cuda版本之后，可以自由地尝试将CenterNet集成到我们的框架中，就像我们在/tensorRT_cpp/src/application/app_centernet中所做的那样。





