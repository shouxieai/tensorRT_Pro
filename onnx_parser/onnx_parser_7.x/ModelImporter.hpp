/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include "ImporterContext.hpp"
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"
#include "builtin_op_importers.hpp"
#include "onnx_utils.hpp"
#include "utils.hpp"

namespace onnx2trt
{

Status parseGraph(IImporterContext* ctx, const ::onnx::GraphProto& graph, bool deserializingINetwork = false, int* currentNode = nullptr);

class ModelImporter : public nvonnxparser::IParser
{
protected:
    string_map<NodeImporter> _op_importers;
    virtual Status importModel(::onnx::ModelProto const& model, uint32_t weight_count,
        onnxTensorDescriptorV1 const* weight_descriptors);

private:
    ImporterContext _importer_ctx;
    RefitMap_t mRefitMap;
    std::list<::onnx::ModelProto> _onnx_models; // Needed for ownership of weights
    int _current_node;
    std::vector<Status> _errors;
    std::vector<nvinfer1::Dims> _input_dims;

public:
    ModelImporter(nvinfer1::INetworkDefinition* network, nvinfer1::ILogger* logger, const std::vector<nvinfer1::Dims>& input_dims)
        : _op_importers(getBuiltinOpImporterMap())
        , _importer_ctx(network, logger, &mRefitMap)
        , _input_dims(input_dims)
    {
    }
    bool parseWithWeightDescriptors(void const* serialized_onnx_model, size_t serialized_onnx_model_size,
        uint32_t weight_count, onnxTensorDescriptorV1 const* weight_descriptors) override;
    bool parse(void const* serialized_onnx_model, size_t serialized_onnx_model_size) override;
    bool supportsModel(void const* serialized_onnx_model, size_t serialized_onnx_model_size,
        SubGraphCollection_t& sub_graph_collection) override;

    bool supportsOperator(const char* op_name) const override;
    void destroy() override
    {
        delete this;
    }
    // virtual void registerOpImporter(std::string op,
    //                                NodeImporter const &node_importer) override {
    //  // Note: This allows existing importers to be replaced
    //  _op_importers[op] = node_importer;
    //}
    // virtual Status const &setInput(const char *name,
    //                               nvinfer1::ITensor *input) override;
    // virtual Status const& setOutput(const char* name, nvinfer1::ITensor** output) override;
    int getNbErrors() const override
    {
        return _errors.size();
    }
    nvonnxparser::IParserError const* getError(int index) const override
    {
        assert(0 <= index && index < (int) _errors.size());
        return &_errors[index];
    }
    void clearErrors() override
    {
        _errors.clear();
    }
    virtual int getRefitMap(const char** weightNames, const char** layerNames, nvinfer1::WeightsRole* roles) override
    {
        int count = 0;
        for (const auto& entry: mRefitMap)
        {
            if (weightNames != nullptr)
            {
                weightNames[count] = entry.first.c_str();
            }
            if (layerNames != nullptr)
            {
                layerNames[count] = entry.second.first.c_str();
            }
            if (roles != nullptr)
            {
                roles[count] = entry.second.second;
            }
            ++count;
        }
        return mRefitMap.size();
    }
    //...LG: Move the implementation to .cpp
    bool parseFromFile(const char* onnxModelFile, int verbosity) override;
    bool parseFromData(const void* onnx_data, size_t size, int verbosity) override;
};

} // namespace onnx2trt
