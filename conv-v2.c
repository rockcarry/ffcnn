#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "utils.h"
#include "conv.h"

static void im2col(float *data_im, int channels, int height, int width,
                   int kernel_h, int kernel_w, int pad, int stride,
                   float *data_col)
{
    int output_h = (height + 2 * pad - kernel_h) / stride + 1;
    int output_w = (width  + 2 * pad - kernel_w) / stride + 1;
    int channels_col = channels * kernel_h * kernel_w;

    for (int c = 0; c < channels_col; c++) {
        int w_offset =  c % kernel_w;
        int h_offset = (c / kernel_w) % kernel_h;
        int c_im = c / (kernel_h * kernel_w);

        for (int h = 0; h < output_h; h++) {
            for (int w = 0; w < output_w; w++) {
                int im_row = h_offset + h * stride - pad;
                int im_col = w_offset + w * stride - pad;

                if (im_row >= 0 && im_row < height && im_col >= 0 && im_col < width) {
                    data_col[(c * output_h + h) * output_w + w] = data_im[(c_im * height + im_row) * width + im_col];
                } else {
                    data_col[(c * output_h + h) * output_w + w] = 0;
                }
            }
        }
    }
}

void groupconv(float *datai, float *dataf, float *datao,
               int iw, int ih, int ic, int ig, int ipad, int istride,
               int fs, int fn, int ow, int oh, int oc, int activation,
               float **gc_buffer, int *gc_bufsize)
{
    int gc_ic  = ic / ig;
    int gc_oc  = oc / ig;
    int gc_fn  = fn / ig;
    int walign = ALIGN(fs * fs * gc_ic, 4);
    int ftsize = walign + 4;

    // 计算 im2col 后矩阵的大小
    int col_height = gc_ic * fs * fs;
    int col_width  = ow * oh;
    int col_size   = col_height * (col_width + gc_oc);

    // 分配内存用于存储 im2col 结果
    if (*gc_bufsize < col_size) {
        *gc_bufsize = col_size;
        free(*gc_buffer); *gc_buffer = malloc(*gc_bufsize * sizeof(float));
        if (*gc_buffer == NULL) { printf("failed to allocate memory for cnntempbuf !"); return; }
    }

    for (int g = 0; g < ig; g++) {
        // 对当前组的输入数据进行 im2col 转换
        im2col(datai, gc_ic, ih, iw, fs, fs, ipad, istride, *gc_buffer);

        // 重组滤波器权重
        float *weight_matrix = *gc_buffer + col_height * col_width;;
        for (int c = 0; c < gc_oc; c++) {
            for (int i = 0; i < gc_ic; i++) {
                for (int j = 0; j < fs; j++) {
                    for (int k = 0; k < fs; k++) {
                        int weight_idx = c * col_height + i * fs * fs + j * fs + k;
                        int filter_idx = c * ftsize + i * fs * fs + k + j * fs;
                        weight_matrix[weight_idx] = dataf[filter_idx];
                    }
                }
            }
        }

        // 执行矩阵乘法: output = col_matrix * weight_matrix^T
        for (int n = 0; n < col_width; n++) {
            for (int c = 0; c < gc_oc; c++) {
                float sum = 0;
                for (int h = 0; h < col_height; h++) {
                    sum += (*gc_buffer)[h * col_width + n] * weight_matrix[c * col_height + h];
                }
                // 添加偏置并应用激活函数
                int bias_idx = c * ftsize + walign;
                datao[c * ow * oh + n] = activate(sum * dataf[bias_idx] + dataf[bias_idx + 1], activation);
            }
        }

        // 移动到下一组
        datai += iw * ih * gc_ic;
        datao += ow * oh * gc_oc;
        dataf += ftsize * gc_fn;
    }
}
