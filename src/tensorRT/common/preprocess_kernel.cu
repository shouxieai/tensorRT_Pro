
#include "preprocess_kernel.cuh"

namespace CUDAKernel{

	Norm Norm::mean_std(const float mean[3], const float std[3], float alpha, ChannelType channel_type){

		Norm out;
		out.type  = NormType::MeanStd;
		out.alpha = alpha;
		out.channel_type = channel_type;
		memcpy(out.mean, mean, sizeof(out.mean));
		memcpy(out.std,  std,  sizeof(out.std));
		return out;
	}

	Norm Norm::alpha_beta(float alpha, float beta, ChannelType channel_type){

		Norm out;
		out.type = NormType::AlphaBeta;
		out.alpha = alpha;
		out.beta = beta;
		out.channel_type = channel_type;
		return out;
	}

	Norm Norm::None(){
		return Norm();
	}	

	#define INTER_RESIZE_COEF_BITS 11
	#define INTER_RESIZE_COEF_SCALE (1 << INTER_RESIZE_COEF_BITS)
	#define CAST_BITS (INTER_RESIZE_COEF_BITS << 1)
	template<typename _T>
	static __inline__ __device__ _T limit(_T value, _T low, _T high){
		return value < low ? low : (value > high ? high : value);
	}

	static __inline__ __device__ int resize_cast(int value){
		return (value + (1 << (CAST_BITS - 1))) >> CAST_BITS;
	}

	// same to opencv
	// reference: https://github.com/opencv/opencv/blob/24fcb7f8131f707717a9f1871b17d95e7cf519ee/modules/imgproc/src/resize.cpp
	// reference: https://github.com/openppl-public/ppl.cv/blob/04ef4ca48262601b99f1bb918dcd005311f331da/src/ppl/cv/cuda/resize.cu
	/*
	  可以考虑用同样实现的resize函数进行训练，python代码在：tools/test_resize.py
	*/
	__global__ void resize_bilinear_and_normalize_kernel(
		uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height, 
		float sx, float sy, Norm norm, int edge
	){
		int position = blockDim.x * blockIdx.x + threadIdx.x;
		if (position >= edge) return;

		int dx      = position % dst_width;
		int dy      = position / dst_width;
		float src_x = (dx + 0.5f) * sx - 0.5f;
		float src_y = (dy + 0.5f) * sy - 0.5f;
		float c0, c1, c2;

		int y_low = floorf(src_y);
		int x_low = floorf(src_x);
		int y_high = limit(y_low + 1, 0, src_height - 1);
		int x_high = limit(x_low + 1, 0, src_width - 1);
		y_low = limit(y_low, 0, src_height - 1);
		x_low = limit(x_low, 0, src_width - 1);

		int ly    = rint((src_y - y_low) * INTER_RESIZE_COEF_SCALE);
		int lx    = rint((src_x - x_low) * INTER_RESIZE_COEF_SCALE);
		int hy    = INTER_RESIZE_COEF_SCALE - ly;
		int hx    = INTER_RESIZE_COEF_SCALE - lx;
		int w1    = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
		float* pdst = dst + dy * dst_width + dx * 3;
		uint8_t* v1 = src + y_low * src_line_size + x_low * 3;
		uint8_t* v2 = src + y_low * src_line_size + x_high * 3;
		uint8_t* v3 = src + y_high * src_line_size + x_low * 3;
		uint8_t* v4 = src + y_high * src_line_size + x_high * 3;

		c0 = resize_cast(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0]);
		c1 = resize_cast(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1]);
		c2 = resize_cast(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2]);

		if(norm.channel_type == ChannelType::Invert){
			float t = c2;
			c2 = c0;  c0 = t;
		}

		if(norm.type == NormType::MeanStd){
			c0 = (c0 * norm.alpha - norm.mean[0]) / norm.std[0];
			c1 = (c1 * norm.alpha - norm.mean[1]) / norm.std[1];
			c2 = (c2 * norm.alpha - norm.mean[2]) / norm.std[2];
		}else if(norm.type == NormType::AlphaBeta){
			c0 = c0 * norm.alpha + norm.beta;
			c1 = c1 * norm.alpha + norm.beta;
			c2 = c2 * norm.alpha + norm.beta;
		}

		int area = dst_width * dst_height;
		float* pdst_c0 = dst + dy * dst_width + dx;
		float* pdst_c1 = pdst_c0 + area;
		float* pdst_c2 = pdst_c1 + area;
		*pdst_c0 = c0;
		*pdst_c1 = c1;
		*pdst_c2 = c2;
	}

	__global__ void warp_affine_bilinear_and_normalize_plane_kernel(uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height, 
		uint8_t const_value_st, float* warp_affine_matrix_2_3, Norm norm, int edge){

		int position = blockDim.x * blockIdx.x + threadIdx.x;
		if (position >= edge) return;

		float m_x1 = warp_affine_matrix_2_3[0];
		float m_y1 = warp_affine_matrix_2_3[1];
		float m_z1 = warp_affine_matrix_2_3[2];
		float m_x2 = warp_affine_matrix_2_3[3];
		float m_y2 = warp_affine_matrix_2_3[4];
		float m_z2 = warp_affine_matrix_2_3[5];

		int dx      = position % dst_width;
		int dy      = position / dst_width;
		float src_x = m_x1 * dx + m_y1 * dy + m_z1;
		float src_y = m_x2 * dx + m_y2 * dy + m_z2;
		float c0, c1, c2;

		if(src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height){
			// out of range
			c0 = const_value_st;
			c1 = const_value_st;
			c2 = const_value_st;
		}else{
			int y_low = floorf(src_y);
			int x_low = floorf(src_x);
			int y_high = y_low + 1;
			int x_high = x_low + 1;

			uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
			float ly    = src_y - y_low;
			float lx    = src_x - x_low;
			float hy    = 1 - ly;
			float hx    = 1 - lx;
			float w1    = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
			uint8_t* v1 = const_value;
			uint8_t* v2 = const_value;
			uint8_t* v3 = const_value;
			uint8_t* v4 = const_value;
			if(y_low >= 0){
				if (x_low >= 0)
					v1 = src + y_low * src_line_size + x_low * 3;

				if (x_high < src_width)
					v2 = src + y_low * src_line_size + x_high * 3;
			}
			
			if(y_high < src_height){
				if (x_low >= 0)
					v3 = src + y_high * src_line_size + x_low * 3;

				if (x_high < src_width)
					v4 = src + y_high * src_line_size + x_high * 3;
			}
			
			// same to opencv
			c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
			c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
			c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
		}

		if(norm.channel_type == ChannelType::Invert){
			float t = c2;
			c2 = c0;  c0 = t;
		}

		if(norm.type == NormType::MeanStd){
			c0 = (c0 * norm.alpha - norm.mean[0]) / norm.std[0];
			c1 = (c1 * norm.alpha - norm.mean[1]) / norm.std[1];
			c2 = (c2 * norm.alpha - norm.mean[2]) / norm.std[2];
		}else if(norm.type == NormType::AlphaBeta){
			c0 = c0 * norm.alpha + norm.beta;
			c1 = c1 * norm.alpha + norm.beta;
			c2 = c2 * norm.alpha + norm.beta;
		}

		int area = dst_width * dst_height;
		float* pdst_c0 = dst + dy * dst_width + dx;
		float* pdst_c1 = pdst_c0 + area;
		float* pdst_c2 = pdst_c1 + area;
		*pdst_c0 = c0;
		*pdst_c1 = c1;
		*pdst_c2 = c2;
	}


	__global__ void warp_affine_bilinear_and_normalize_focus_kernel(uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height, 
		uint8_t const_value_st, float* warp_affine_matrix_1_3, Norm norm, int edge){

		int position = blockDim.x * blockIdx.x + threadIdx.x;
		if (position >= edge) return;

		float m_k   = *warp_affine_matrix_1_3++;
		float m_b0  = *warp_affine_matrix_1_3++;
		float m_b1  = *warp_affine_matrix_1_3++;

		int dx      = position % dst_width;
		int dy      = position / dst_width;
		float src_x = m_k * dx + m_b0;
		float src_y = m_k * dy + m_b1;
		float c0, c1, c2;

		if(src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height){
			// out of range
			c0 = const_value_st;
			c1 = const_value_st;
			c2 = const_value_st;
		}else{
			int y_low = floorf(src_y);
			int x_low = floorf(src_x);
			int y_high = y_low + 1;
			int x_high = x_low + 1;

			uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
			float ly    = src_y - y_low;
			float lx    = src_x - x_low;
			float hy    = 1 - ly;
			float hx    = 1 - lx;
			float w1    = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
			uint8_t* v1 = const_value;
			uint8_t* v2 = const_value;
			uint8_t* v3 = const_value;
			uint8_t* v4 = const_value;
			if(y_low >= 0){
				if (x_low >= 0)
					v1 = src + y_low * src_line_size + x_low * 3;

				if (x_high < src_width)
					v2 = src + y_low * src_line_size + x_high * 3;
			}
			
			if(y_high < src_height){
				if (x_low >= 0)
					v3 = src + y_high * src_line_size + x_low * 3;

				if (x_high < src_width)
					v4 = src + y_high * src_line_size + x_high * 3;
			}

			// same to opencv
			c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
			c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
			c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
		}

		if(norm.channel_type == ChannelType::Invert){
			float t = c2;
			c2 = c0;  c0 = t;
		}

		if(norm.type == NormType::MeanStd){
			c0 = (c0 * norm.alpha - norm.mean[0]) / norm.std[0];
			c1 = (c1 * norm.alpha - norm.mean[1]) / norm.std[1];
			c2 = (c2 * norm.alpha - norm.mean[2]) / norm.std[2];
		}else if(norm.type == NormType::AlphaBeta){
			c0 = c0 * norm.alpha + norm.beta;
			c1 = c1 * norm.alpha + norm.beta;
			c2 = c2 * norm.alpha + norm.beta;
		}

		int after_focus_width  = dst_width / 2;
		int after_focus_height = dst_height / 2;
		int fdx = dx / 2;
		int fdy = dy / 2;
		int fc  = ((dx % 2) << 1) | (dy % 2);

		/**
		 *   x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]
		 *    4                     fc
		 *    3                     [0, 1, 2]
		 *    after_focus_height    fdy
		 *    after_focus_width     fdx
		 *    左乘右加
		 **/

		float* pdst_c0 = dst + ((fc * 3 + 0) * after_focus_height + fdy) * after_focus_width + fdx;
		float* pdst_c1 = dst + ((fc * 3 + 1) * after_focus_height + fdy) * after_focus_width + fdx;
		float* pdst_c2 = dst + ((fc * 3 + 2) * after_focus_height + fdy) * after_focus_width + fdx;

		*pdst_c0 = c0;
		*pdst_c1 = c1;
		*pdst_c2 = c2;
	}

	__global__ void normalize_feature_kernel(float* feature_array, int num_feature, int feature_length, int edge){

		/*
		&   1 gz         bi.z   0
		*   1 gy         bi.y   0
        *   N NF         bi.x   ~
		*   1 1          ti.z   0
		*   F FL / 32    ti.y   ~
		*   Q 32         ti.x   ~
		*/

		int position = (blockIdx.x * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
        if (position >= edge) return;

		extern __shared__ float l2_norm[];

		int irow    = position / feature_length;
		int icol    = position % feature_length;

		if(icol == 0)
			l2_norm[irow] = 0;
		
		__syncthreads();

		float value = feature_array[position];
		atomicAdd(l2_norm + irow, value * value);

		__syncthreads();
		if(icol == 0)
			l2_norm[irow] = sqrt(l2_norm[irow]);

		__syncthreads();
		feature_array[position] = value / l2_norm[irow];
	}

    static __device__ uint8_t cast(float value){
        return value < 0 ? 0 : (value > 255 ? 255 : value);
    }

    static __global__ void convert_nv12_to_bgr_kernel(const uint8_t* y, const uint8_t* uv, int width, int height, int linesize, uint8_t* dst_bgr, int edge){

        int position = blockDim.x * blockIdx.x + threadIdx.x;
        if (position >= edge) return;

        int ox = position % width;
        int oy = position / width;
        const uint8_t& yvalue = y[oy * linesize + ox];
        int offset_uv = (oy >> 1) * linesize + (ox & 0xFFFFFFFE);
        const uint8_t& u = uv[offset_uv + 0];
        const uint8_t& v = uv[offset_uv + 1];
		dst_bgr[position * 3 + 0] = 1.164f * (yvalue - 16.0f) + 2.018f * (u - 128.0f);
		dst_bgr[position * 3 + 1] = 1.164f * (yvalue - 16.0f) - 0.813f * (v - 128.0f) - 0.391f * (u - 128.0f);
		dst_bgr[position * 3 + 2] = 1.164f * (yvalue - 16.0f) + 1.596f * (v - 128.0f);
    }


	/////////////////////////////////////////////////////////////////////////
	void convert_nv12_to_bgr_invoke(
		const uint8_t* y, const uint8_t* uv, int width, int height, int linesize, uint8_t* dst, cudaStream_t stream){
			
		int total = width * height;
		dim3 grid = CUDATools::grid_dims(total);
		dim3 block = CUDATools::block_dims(total);

		checkCudaKernel(convert_nv12_to_bgr_kernel<<<grid, block, 0, stream>>>(
			y, uv, width, height, linesize,
			dst, total
		));
	}

	void warp_affine_bilinear_and_normalize_plane(
		uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height,
		float* matrix_2_3, uint8_t const_value, const Norm& norm,
		cudaStream_t stream) {
		
		int jobs   = dst_width * dst_height;
		auto grid  = CUDATools::grid_dims(jobs);
		auto block = CUDATools::block_dims(jobs);
		
		checkCudaKernel(warp_affine_bilinear_and_normalize_plane_kernel << <grid, block, 0, stream >> > (
			src, src_line_size,
			src_width, src_height, dst,
			dst_width, dst_height, const_value, matrix_2_3, norm, jobs
		));
	}

	
	void warp_affine_bilinear_and_normalize_focus(
        uint8_t* src, int src_line_size, int src_width, int src_height, 
        float* dst  , int dst_width, int dst_height,
        float* matrix_1_3, uint8_t const_value, const Norm& norm,
        cudaStream_t stream){

		int jobs   = dst_width * dst_height;
		auto grid  = CUDATools::grid_dims(jobs);
		auto block = CUDATools::block_dims(jobs);
		
		checkCudaKernel(warp_affine_bilinear_and_normalize_focus_kernel << <grid, block, 0, stream >> > (
			src, src_line_size,
			src_width, src_height, dst,
			dst_width, dst_height, const_value, matrix_1_3, norm, jobs
		));
	}

	void resize_bilinear_and_normalize(
		uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height,
		const Norm& norm,
		cudaStream_t stream) {
		
		int jobs   = dst_width * dst_height;
		auto grid  = CUDATools::grid_dims(jobs);
		auto block = CUDATools::block_dims(jobs);
		
		checkCudaKernel(resize_bilinear_and_normalize_kernel << <grid, block, 0, stream >> > (
			src, src_line_size,
			src_width, src_height, dst,
			dst_width, dst_height, src_width/(float)dst_width, src_height/(float)dst_height, norm, jobs
		));
	}
	
	void norm_feature(
        float* feature_array, int num_feature, int feature_length,
        cudaStream_t stream
    ){
		Assert(feature_length % 32 == 0);

		int jobs   = num_feature * feature_length;
		auto grid  = dim3(num_feature);
		auto block = dim3(feature_length / 32, 32);
		checkCudaKernel(normalize_feature_kernel << <grid, block, num_feature * sizeof(float), stream >> > (
			feature_array, num_feature, feature_length, jobs
		));	
	}
};
