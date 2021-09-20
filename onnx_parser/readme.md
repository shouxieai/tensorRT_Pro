# Onnx parser for 7.x/8.x
- Origin Code 7.x: https://github.com/onnx/onnx-tensorrt/releases/tag/release%2F7.2.1
- Origin Code 8.x: https://github.com/onnx/onnx-tensorrt/releases/tag/release%2F8.0

# TensorRT 7.x support
1. Replace onnx_parser_for_7.x/onnx_parser to src/tensorRT/onnx_parser
    - `rm -rf src/tensorRT/onnx_parser`
    - `cp -r onnx_parser/onnx_parser_7.x src/tensorRT/onnx_parser`
    - or execute `bash onnx_parser/use_tensorrt_7.x.sh`
2. Configure Makefile/CMakeLists.txt path to TensorRT7.x
3. Execute `make yolo -j64`

# TensorRT 8.x support
1. Replace onnx_parser_for_8.x/onnx_parser to src/tensorRT/onnx_parser
    - `rm -rf src/tensorRT/onnx_parser`
    - `cp -r onnx_parser/onnx_parser_8.x src/tensorRT/onnx_parser`
    - or execute `bash onnx_parser/use_tensorrt_8.x.sh`
2. Configure Makefile/CMakeLists.txt path to TensorRT8.x
3. Execute `make yolo -j64`

# Unsupported TensorRT for less 7.x version