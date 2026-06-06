#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "utils.h"
#include "conv.h"

// im2row implementation
static void im2row(float *img, int w, int h, int c, int pad, int fs, int stride, int walign, int y, float *buf)
{
    float *src = img + (y - pad) * w - pad, *dst = buf;
    int    i, j, k, x;
    for (i = 0; i < c; i++) {
        for (j = 0; j < fs; j++) {
            float *tmp = dst;
            for (x = 0; x < w; x += stride) {
                for (k = 0; k < fs; k++) dst[k] = (unsigned)(x - pad + k) < (unsigned)w && (unsigned)(y - pad + j) < (unsigned)h ? src[k] : 0;
                src += stride, dst += walign;
            }
            src += w - x, dst = tmp + fs;
        }
        src += w * (h - fs);
    }
}

// Generic convolution - optimized for single-core
// Key optimizations:
//   1. Loop order: y -> x -> c(4 at once) -> ftsize
//   2. 4 output channels processed together (better ILP and vectorization)
//   3. Use restrict to help compiler optimize
static void convolution_generic(float *restrict datai, float *restrict dataf, float *restrict datao,
                        int iw, int ih, int ic, int ipad, int istride,
                        int fs, int ftsize, int ow, int oh, int oc, int activation,
                        float *restrict gc_buffer)
{
    for (int y = 0; y < oh; y++) {
        im2row(datai, iw, ih, ic, ipad, fs, istride, ftsize, y * istride, gc_buffer);
        for (int x = 0; x < ow; x++) {
            for (int c = 0; c < oc; c++) {
                float sum = 0;
                for (int i = 0; i < ftsize; i++) sum += dataf[c * (ftsize + 4) + i] * gc_buffer[x * ftsize + i];
                datao[c * ow * oh + y * ow + x] = activate(sum * dataf[c * (ftsize + 4) + ftsize + 0] + dataf[c * (ftsize + 4) + ftsize + 1], activation);
            }
        }
    }
}

// Specialized convolution for pad=0, fs=1, stride=1 (1x1 convolution)
// No im2row needed - directly compute output = input * weights
// Memory layout: datai[ic][y][x], datao[oc][y][x], dataf[oc][ic]
static void convolution_pad0_fs1_stride1(float *restrict datai, float *restrict dataf, float *restrict datao,
                        int iw, int ih, int ic, int ipad, int istride,
                        int fs, int ftsize, int ow, int oh, int oc, int activation,
                        float *restrict gc_buffer)
{
    // Process output in blocks of 4 channels for better ILP
    for (int y = 0; y < oh; y++) {
        for (int x = 0; x < ow; x++) {
            int oc_aligned = oc & ~3;
            int c;

            for (c = 0; c < oc_aligned; c += 4) {
                float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
                float *wt0 = dataf + (c  ) * (ftsize + 4);
                float *wt1 = dataf + (c+1) * (ftsize + 4);
                float *wt2 = dataf + (c+2) * (ftsize + 4);
                float *wt3 = dataf + (c+3) * (ftsize + 4);

                // Accumulate over input channels
                for (int ci = 0; ci < ic; ci++) {
                    float val = datai[(ci * ih + y) * iw + x];
                    sum0 += wt0[ci] * val;
                    sum1 += wt1[ci] * val;
                    sum2 += wt2[ci] * val;
                    sum3 += wt3[ci] * val;
                }

                int out_idx = y * ow + x;
                datao[(c  ) * ow * oh + out_idx] = activate(sum0 * wt0[ftsize + 0] + wt0[ftsize + 1], activation);
                datao[(c+1) * ow * oh + out_idx] = activate(sum1 * wt1[ftsize + 0] + wt1[ftsize + 1], activation);
                datao[(c+2) * ow * oh + out_idx] = activate(sum2 * wt2[ftsize + 0] + wt2[ftsize + 1], activation);
                datao[(c+3) * ow * oh + out_idx] = activate(sum3 * wt3[ftsize + 0] + wt3[ftsize + 1], activation);
            }

            // Remaining output channels
            for (; c < oc; c++) {
                float sum = 0;
                float *wt = dataf + c * (ftsize + 4);
                for (int ci = 0; ci < ic; ci++) {
                    sum += wt[ci] * datai[(ci * ih + y) * iw + x];
                }
                int out_idx = y * ow + x;
                datao[c * ow * oh + out_idx] = activate(sum * wt[ftsize + 0] + wt[ftsize + 1], activation);
            }
        }
    }
}

// datai - input tensor iw * ih * ic divide to ig groups, with ipad & istride
// dataf - cnn filter, fs * fs size, fn number
// datao - output tensor ow * oh size
void groupconv(float *datai, float *dataf, float *datao,
               int iw, int ih, int ic, int ig, int ipad, int istride,
               int fs, int fn, int ow, int oh, int oc, int activation,
               float **gc_buffer, int *gc_bufsize)
{
    int gc_ic  = ic / ig; // group conv ic
    int gc_oc  = oc / ig; // group conv oc
    int gc_fn  = fn / ig; // group conv fn
    int ftsize = ALIGN(fs * fs * gc_ic, 4);

    // Single-core: only need buffer for one y (ftsize * ow)
    if (*gc_bufsize < ftsize * ow) {
        *gc_bufsize = ftsize * ow;
        free(*gc_buffer);
        *gc_buffer = (float*)malloc(*gc_bufsize * sizeof(float));
        if (*gc_buffer == NULL) { printf("failed to allocate memory for cnntempbuf !\n"); return; }
    }

    for (int g = 0; g < ig; g++) {
        if (ipad == 0 && fs == 1 && istride == 1) {
            convolution_pad0_fs1_stride1(datai, dataf, datao, iw, ih, gc_ic, ipad, istride, fs, ftsize, ow, oh, gc_oc, activation, *gc_buffer);
        } else {
            convolution_generic(datai, dataf, datao, iw, ih, gc_ic, ipad, istride, fs, ftsize, ow, oh, gc_oc, activation, *gc_buffer);
        }
        datai += (long)iw * ih * gc_ic;
        datao += (long)ow * oh * gc_oc;
        dataf += (long)(ftsize + 4) * gc_fn;
    }
}
