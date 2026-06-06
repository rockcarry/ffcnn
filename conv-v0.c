#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "utils.h"
#include "conv.h"

static void convolution(float *datai, float *dataf, float *datao,
                        int iw, int ih, int ic, int ipad, int istride,
                        int fs, int ftsize, int ow, int oh, int oc, int activation,
                        float *gc_buffer)
{
    for (int y = 0; y < oh; y++) {
        for (int x = 0; x < ow; x++) {
            for (int c = 0; c < oc; c++) {
                float sum = 0;
                for (int i = 0; i < ic; i++) {
                    for (int j = 0; j < fs; j++) {
                        for (int k = 0; k < fs; k++) {
                            int ix = x * istride - ipad + k;
                            int iy = y * istride - ipad + j;
                            if ((unsigned)ix < (unsigned)iw && (unsigned)iy < (unsigned)ih) {
                                sum += datai[i * iw * ih + ix + iy * iw] * dataf[c * (ftsize + 4) + i * fs * fs + k + j * fs];
                            }
                        }
                    }
                }
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

    for (int g = 0; g < ig; g++) {
        convolution(datai, dataf, datao, iw, ih, gc_ic, ipad, istride, fs, ftsize, ow, oh, gc_oc, activation, *gc_buffer);
        datai += iw * ih * gc_ic;
        datao += ow * oh * gc_oc;
        dataf += (ftsize + 4) * gc_fn;
    }
}
