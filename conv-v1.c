#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "utils.h"
#include "conv.h"

static void im2row(float *img, int w, int h, int c, int pad, int fs, int stride, int ftsize, int y, float *buf)
{
    float *src = img + (y - pad) * w - pad, *dst = buf;
    int    i, j, k, x;
    for (i = 0; i < c; i++) {
        for (j = 0; j < fs; j++) {
            float *tmp = dst;
            for (x = 0; x < w; x += stride) {
                for (k = 0; k < fs; k++) dst[k] = (unsigned)(x - pad + k) < (unsigned)w && (unsigned)(y - pad + j) < (unsigned)h ? src[k] : 0;
                src += stride, dst += ftsize;
            }
            src += w - x, dst = tmp + fs;
        }
        src += w * (h - fs);
    }
}

static void convolution(float *datai, float *dataf, float *datao,
                        int iw, int ih, int ic, int ipad, int istride,
                        int fs, int ftsize, int ow, int oh, int oc, int activation,
                        float *gc_buffer)
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

    if (*gc_bufsize < ftsize * ow) {
        *gc_bufsize = ftsize * ow;
        free(*gc_buffer); *gc_buffer = malloc(*gc_bufsize * sizeof(float));
        if (*gc_buffer == NULL) { printf("failed to allocate memory for cnntempbuf !"); return; }
    }

    for (int g = 0; g < ig; g++) {
        convolution(datai, dataf, datao, iw, ih, gc_ic, ipad, istride, fs, ftsize, ow, oh, gc_oc, activation, *gc_buffer);
        datai += iw * ih * gc_ic;
        datao += ow * oh * gc_oc;
        dataf += (ftsize + 4) * gc_fn;
    }
}
