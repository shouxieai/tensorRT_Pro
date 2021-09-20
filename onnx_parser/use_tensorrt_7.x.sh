#!/bin/bash

echo Remove src/tensorRT/onnx_parser
rm -rf src/tensorRT/onnx_parser

echo Copy [onnx_parser/onnx_parser_7.x] to [src/tensorRT/onnx_parser]
cp -r onnx_parser/onnx_parser_7.x src/tensorRT/onnx_parser

echo Configure your tensorRT path to 7.x
echo After that, you can execute the command 'make yolo -j64'