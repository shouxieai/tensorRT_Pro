#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include "centernet.hpp"
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

void warpaffine_and_normalize_opencv_cpp(Mat& src, int src_w, int src_h, Mat& M_2x3 , Mat& dst, int dst_w, int dst_h);
void modify_shape(const Mat& image, shared_ptr<TRT::Tensor>& input);
void decode(Mat image, shared_ptr<TRT::Infer> engine, float conf_T);


void resize_into(const Mat& image, shared_ptr<TRT::Tensor>& input){
    
    for (int y = 0; y < image.cols; ++y){
        for (int x = 0; x < image.rows; ++x){
            auto pixel = image.at<Vec3f>(y,x);
            input->at<float>(0,0,y,x) = pixel[0];
            input->at<float>(0,1,y,x) = pixel[1];
            input->at<float>(0,2,y,x) = pixel[2];
            
        } // assign a pixel value of the image to a given location of the input
    }
}

void warpaffine_and_normalize_opencv_cpp(Mat& src, int src_w, int src_h, Mat& M_2x3, Mat& dst, int dst_w, int dst_h){
    // 1. some key info        
    float scale = std::min((float)dst_h/(float)src_h, (float)dst_w/(float)src_w);

    float trans_w = -src_w * 0.5 * scale + dst_w * 0.5;
    float trans_h = -src_h * 0.5 * scale + dst_h * 0.5;

    Scalar mean(0.408, 0.447, 0.470);
    Scalar std (0.289, 0.274, 0.278);

    float affine_mat_data[6] = {scale, 0, trans_w, 0,  scale, trans_h};
    M_2x3 = cv::Mat(2,3, CV_32F, affine_mat_data).clone();

    // 2. warpaffine
    cv::warpAffine(src, dst, M_2x3, 
            Size(dst_w, dst_h), 
            cv::INTER_LINEAR, 
            cv::BORDER_CONSTANT, 
            cv::Scalar::all(0));
    
    
    cv::imwrite("affined_result_cpp.jpg", dst);
    dst.convertTo(dst, CV_32F, 1/255.0f);
    dst = (dst - mean) / std;
}

void decode(Mat image, shared_ptr<TRT::Infer> engine, float conf_T, Mat& M_2x3){
    // image is the src_image

    Mat inv_M;
    cv::invertAffineTransform(M_2x3, inv_M);
    float* inv_M_ptr = inv_M.ptr<float>(0);
    cout << M_2x3 << endl;
    
    auto show = image.clone();
    auto src_w = image.cols;
    auto src_h = image.rows;

    // 1. three heads of the output
    auto hm = engine->output(0);    // see it as a 1d array [1x(128x128x80)]
    auto pool_hm = engine->output(1); // 1x(80x128x128)
    auto wh = engine->output(2);    // 1x(2x128x128)
    auto regxy = engine->output(3); // 1x(2x128x128)
                                    // col->row->channel

    auto hm_ptr = hm->cpu<float>();
    auto pool_ptr = pool_hm->cpu<float>();
    auto regxy_ptr = regxy ->cpu<float>();
    auto wh_ptr = wh->cpu<float>();

    auto desigmoid = [](float y){
        return - log(1/y - 1);
    };
    
    float desigmoid_conf_T = desigmoid(0.3);
    int x; int y; int c;
    int stride = 4;

    for (int i = 0; i < hm->numel(); ++i,++hm_ptr,++pool_ptr){
        if ((*hm_ptr == *pool_ptr) && (*hm_ptr >= desigmoid_conf_T) ){ 
            // printf("%d \n",i);
            x = i % hm->width();
            y = (i / hm->width()) % hm->height();
            c = i / (hm->count(2));

            float dx = regxy->at<float>(0,0,y,x);
            float dy = regxy->at<float>(0,1,y,x);

            float w = stride * wh->at<float>(0,0,y,x);
            float h = stride * wh->at<float>(0,1,y,x);

            float input_x = (x+dx)*stride; 
            float input_y = (y+dy)*stride;

            float _left = input_x - w/2;
            float _right = input_x + w/2;
            float _top = input_y - h/2;
            float _bottom = input_y + h/2;

            int left = inv_M_ptr[0] * _left + inv_M_ptr[1] * _top + inv_M_ptr[2];
            int true_top  = inv_M_ptr[3] * _left + inv_M_ptr[4] * _top + inv_M_ptr[5];

            int right = inv_M_ptr[0] * _right + inv_M_ptr[1] * _bottom + inv_M_ptr[2];
            int bottom = inv_M_ptr[3] * _right + inv_M_ptr[4] * _bottom + inv_M_ptr[5];
            
            printf("%d,%d,%d,%d \n", left, true_top, right, bottom);
            
            cv::rectangle(image, Point(left, true_top), Point(right, bottom), Scalar(0,255,0),2);

        }
    }
    cv::imwrite("final_result_cpp.jpg",image);
    
}









// void warp_affine_bilinear_and_normalize_kernel(uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height, 
//      uint8_t const_value_st, float* warp_affine_matrix_2_3, Norm norm, int edge){

//      int position = blockDim.x * blockIdx.x + threadIdx.x;
//      if (position >= edge) return;

//      float m_x1 = warp_affine_matrix_2_3[0];
//      float m_y1 = warp_affine_matrix_2_3[1];
//      float m_z1 = warp_affine_matrix_2_3[2];
//      float m_x2 = warp_affine_matrix_2_3[3];
//      float m_y2 = warp_affine_matrix_2_3[4];
//      float m_z2 = warp_affine_matrix_2_3[5];

//      int dx      = position % dst_width;
//      int dy      = position / dst_width;
//      float src_x = (m_x1 * dx + m_y1 * dy + m_z1) + 0.5f;
//      float src_y = (m_x2 * dx + m_y2 * dy + m_z2) + 0.5f;
//      float c0, c1, c2;

//      if(src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height){
//          // out of range
//          c0 = const_value_st;
//          c1 = const_value_st;
//          c2 = const_value_st;
//      }else{
//          int y_low = floor(src_y);
//          int x_low = floor(src_x);
//          int y_high = y_low + 1;
//          int x_high = x_low + 1;

//          uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
//          float ly    = src_y - y_low;
//          float lx    = src_x - x_low;
//          float hy    = 1 - ly;
//          float hx    = 1 - lx;
//          float w1    = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
//          float* pdst = dst + dy * dst_width + dx * 3;
//          uint8_t* v1 = const_value;
//          uint8_t* v2 = const_value;
//          uint8_t* v3 = const_value;
//          uint8_t* v4 = const_value;
//          if(y_low >= 0){
//              if (x_low >= 0)
//                  v1 = src + y_low * src_line_size + x_low * 3;

//              if (x_high < src_width)
//                  v2 = src + y_low * src_line_size + x_high * 3;
//          }
            
//          if(y_high < src_height){
//              if (x_low >= 0)
//                  v3 = src + y_high * src_line_size + x_low * 3;

//              if (x_high < src_width)
//                  v4 = src + y_high * src_line_size + x_high * 3;
//          }

//          c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f;
//          c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f;
//          c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f;
//      }

//      if(norm.channel_type == ChannelType::Invert){
//          float t = c2;
//          c2 = c0;  c0 = t;
//      }

//      if(norm.type == NormType::MeanStd){
//          c0 = (c0 * norm.alpha - norm.mean[0]) / norm.std[0];
//          c1 = (c1 * norm.alpha - norm.mean[1]) / norm.std[1];
//          c2 = (c2 * norm.alpha - norm.mean[2]) / norm.std[2];
//      }else if(norm.type == NormType::AlphaBeta){
//          c0 = c0 * norm.alpha + norm.beta;
//          c1 = c1 * norm.alpha + norm.beta;
//          c2 = c2 * norm.alpha + norm.beta;
//      }

//      int area = dst_width * dst_height;
//      float* pdst_c0 = dst + dy * dst_width + dx;
//      float* pdst_c1 = pdst_c0 + area;
//      float* pdst_c2 = pdst_c1 + area;
//      *pdst_c0 = c0;
//      *pdst_c1 = c1;
//      *pdst_c2 = c2;
//  }
