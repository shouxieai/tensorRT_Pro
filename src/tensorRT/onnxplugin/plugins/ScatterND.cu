
#include "onnxplugin/onnxplugin.hpp"
#include <common/cuda_tools.hpp>
#include <cublas_v2.h>
#include <cuda_fp16.h>

using namespace ONNXPlugin;


//this scatter kernel works on a 2d table writing rows 
//index is 1-D array
//updates is 2-D array
//output is 2-D array
//output[index[i]] = updates[i]
__global__ void scatterKernel(
    char* output,
    const char* updates,
    const int* indices,
    int pitch,
    int rowSize)
{
    int idx = indices[blockIdx.x];
    char* pDst = (char*)output + idx * pitch;
    const char* pSrc = updates + blockIdx.x * rowSize; 
    memcpy(pDst, pSrc, rowSize);
}

// Transform nd index to 1 - d index
__global__ void transformIdxKernel(
    int* output,
    const int* transformCoeff, // these are actually the output pitches of the respective dimensions
    const int* indices,
    int sliceRank)
{
    const int* idx = indices + sliceRank * blockIdx.x;
    int transformedIdx = 0;
    for (int i = 0; i < sliceRank; i++)
    {
        transformedIdx += idx[i] * transformCoeff[i];
    }
    output[blockIdx.x] = transformedIdx;
}


void MyScatterNDInference( 
    cudaStream_t stream,
    int* transformCoeff,
    int nOutputDims,
    int sliceRank,        
    int nRows,
    int rowSize,
    int copySize,
    int sizeOfElementInBytes,         
    const void* index,
    const void* updates,
    const void* data,
    void* output,
    void* workspace)
{
    const int* _index = (const int*)(index);
    const char* _updates = (const char*)(updates);
    char* _output = (char*)(output);
    int* wo = (int*)(workspace);
    int* transformedIdx = wo + sizeof(int)*nOutputDims;
    int* deviceTransformCoeff = wo;
    cudaMemcpy(workspace, transformCoeff, sizeof(int)*nOutputDims,cudaMemcpyHostToDevice );
    transformIdxKernel<<<nRows, 1, 0, stream>>>(transformedIdx, deviceTransformCoeff, _index, sliceRank);
    cudaMemcpy(output, data, copySize, cudaMemcpyDeviceToDevice);
    //assuming output pitch = rowSize i.e no padding
    scatterKernel<<<nRows, 1, 0, stream>>>(_output, _updates, transformedIdx, rowSize*4, rowSize*4);
}

class MyScatterND : public TRTPlugin {
public:
	SetupPlugin(MyScatterND);

	static constexpr  int indexTensorIdx = 1;
    static constexpr  int updateTensorIdx = 2;
    static constexpr  int dataTensorIdx = 0;

	int32_t calculateNumSlices(nvinfer1::Dims indexTensorDims) const noexcept
	{
		int32_t nSlices = 1;
		for (int i = 0; i < indexTensorDims.nbDims-1; i++)
		{
			nSlices *= indexTensorDims.d[i];
		}
		return nSlices;
	}

	size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs, const nvinfer1::PluginTensorDesc* outputs,int32_t nbOutputs) const noexcept
	{
		int32_t nSlices = calculateNumSlices(inputs[indexTensorIdx].dims);
		//transformCoeffs + transformed indices
		return outputs[0].dims.MAX_DIMS * sizeof(int32_t) + nSlices * sizeof(int32_t);
	}

	void calculateTransformCoeff(const nvinfer1::Dims& dataTensorDims, int indexRank, int32_t* transformCoeff) const noexcept
	{    
		std::vector<int32_t> pitches;    
		for (int32_t i = indexRank - 1, nIndx = 1; i >= 0 ; i--)
		{
			pitches.push_back(nIndx);
			nIndx *= dataTensorDims.d[i];        
		}

		std::reverse(pitches.begin(), pitches.end()); //last dimension pitch is always one (assuming linear mem)

		std::copy(pitches.begin(), pitches.end(), transformCoeff);
	}

	bool supportsFormatCombination(int32_t pos, const nvinfer1::PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
	{
		assert(pos < 4);
		assert(nbInputs == 3);
		assert(nbOutputs == 1);
		const nvinfer1::PluginTensorDesc& desc = inOut[pos];
		bool ret = false;
		switch (pos)
		{
		case dataTensorIdx:
		case updateTensorIdx:
			ret = ((desc.type == nvinfer1::DataType::kFLOAT || desc.type == nvinfer1::DataType::kINT32)
				&& desc.format == nvinfer1::TensorFormat::kLINEAR);
			break;
		case indexTensorIdx:
			ret = (desc.type == nvinfer1::DataType::kINT32 && desc.format == nvinfer1::TensorFormat::kLINEAR);
			break;
		case 3:
			ret = ((desc.type == nvinfer1::DataType::kFLOAT || desc.type == nvinfer1::DataType::kINT32) && desc.format == nvinfer1::TensorFormat::kLINEAR);
			break;
		}
		return ret;
	}

	int32_t calculateCopySize(const nvinfer1::Dims& dataDims) const noexcept
	{
		int32_t copySize = 1;
		for (int i = 0; i < dataDims.nbDims; i++)
		{
			copySize *= dataDims.d[i];    
		}
		copySize *= sizeof(float);
		return copySize;
	}

	virtual void config_finish() override{

	}

	virtual nvinfer1::DimsExprs getOutputDimensions(
        	int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override{

		return inputs[0];
	}
	virtual int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
            const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override{

		int32_t transformCoeff[outputDesc[0].dims.MAX_DIMS];
		std::memset(transformCoeff, 0, sizeof(int32_t)*outputDesc[0].dims.MAX_DIMS);
		nvinfer1::Dims IndexDims = inputDesc[indexTensorIdx].dims;
		
		nvinfer1::Dims dataDims = inputDesc[dataTensorIdx].dims;

		int32_t indexRank = IndexDims.d[IndexDims.nbDims-1];
		assert(indexRank <= dataDims.nbDims);

		int32_t nSlices = calculateNumSlices(IndexDims);
		int32_t rowSize = 1;
		int32_t copySize = calculateCopySize(dataDims);
		int32_t elementSizeInBytes = 1;
		switch (inputDesc->type)
		{
		case nvinfer1::DataType::kFLOAT:
		case nvinfer1::DataType::kINT32:
			elementSizeInBytes = 4;
			break;
		case nvinfer1::DataType::kHALF:
			elementSizeInBytes = 2;
			break;
		case nvinfer1::DataType::kINT8:
		case nvinfer1::DataType::kBOOL:
			elementSizeInBytes = 1;
			break;
		}
		
		for (int i = indexRank; i < dataDims.nbDims; i++)
		{
			rowSize *= dataDims.d[i];
		}
		
		calculateTransformCoeff(dataDims, indexRank, transformCoeff);

		MyScatterNDInference(stream, transformCoeff, 
		dataDims.nbDims, 
		indexRank, 
		nSlices, 
		rowSize,  
		copySize, 
		elementSizeInBytes,  
		inputs[indexTensorIdx],
		inputs[updateTensorIdx],
		inputs[dataTensorIdx],
		outputs[0],
		workspace );    
		return 0;
	}
	
	int enqueue(const std::vector<GTensor>& inputs, std::vector<GTensor>& outputs, const std::vector<GTensor>& weights, void* workspace, cudaStream_t stream) override{
		return 0;
	}
};
RegisterPlugin(MyScatterND);
