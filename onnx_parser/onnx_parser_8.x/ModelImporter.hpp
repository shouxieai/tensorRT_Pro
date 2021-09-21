/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ImporterContext.hpp"
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"
#include "builtin_op_importers.hpp"
#include "utils.hpp"

namespace onnx2trt
{

Status parseGraph(IImporterContext* ctx, const ::onnx::GraphProto& graph, bool deserializingINetwork = false, int* currentNode = nullptr);

class ModelImporter : public nvonnxparser::IParser
{
protected:
    string_map<NodeImporter> _op_importers;
    virtual Status importModel(::onnx::ModelProto const& model);

private:
    ImporterContext _importer_ctx;
    std::list<::onnx::ModelProto> _onnx_models; // Needed for ownership of weights
    int _current_node;
    std::vector<Status> _errors;
    std::vector<nvinfer1::Dims> _input_dims;

public:
    ModelImporter(nvinfer1::INetworkDefinition* network, nvinfer1::ILogger* logger, const std::vector<nvinfer1::Dims>& input_dims)
        : _op_importers(getBuiltinOpImporterMap())
        , _importer_ctx(network, logger)
        , _input_dims(input_dims)
    {
    }
    bool parseWithWeightDescriptors(void const* serialized_onnx_model, size_t serialized_onnx_model_size) override;
    bool parse(void const* serialized_onnx_model, size_t serialized_onnx_model_size, const char* model_path = nullptr) override;
    bool supportsModel(void const* serialized_onnx_model, size_t serialized_onnx_model_size,
        SubGraphCollection_t& sub_graph_collection, const char* model_path = nullptr) override;

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

    //...LG: Move the implementation to .cpp
    bool parseFromFile(const char* onnxModelFile, int verbosity) override;
    bool parseFromData(const void* onnx_data, size_t size, int verbosity) override;
};

} // namespace onnx2trt
