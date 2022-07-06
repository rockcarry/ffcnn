#include <stdlib.h>
#include <stdio.h>
#include "conv.h"

void layer_groupconv_forward(NET *net, LAYER *ilayer, LAYER *olayer)
{
    int    walign, ftsize, x, y, g, c, ic, oc, fn, i, j, k;
    float *datai, *datao, *dataf, sum;

    datai  = ilayer->data;
    datao  = olayer->data;
    dataf  = ilayer->filter;
    ic     = ilayer->c / ilayer->groups;
    oc     = olayer->c / ilayer->groups;
    fn     = ilayer->fn/ ilayer->groups;
    walign = ALIGN(ilayer->fs * ilayer->fs * ic, 4);
    ftsize = walign + 4;

    for (g=0; g<ilayer->groups; g++) {
        for (y=0; y<olayer->h; y++) {
            for (x=0; x<olayer->w; x++) {
                for (c=0; c<oc; c++) {
                    for (sum=0,i=0; i<ic; i++) {
                        for (j=0; j<ilayer->fs; j++) {
                            for (k=0; k<ilayer->fs; k++) {
                                int ix = x * ilayer->stride - ilayer->pad + k;
                                int iy = y * ilayer->stride - ilayer->pad + j;
                                if ((unsigned)ix < (unsigned)ilayer->w && (unsigned)iy < (unsigned)ilayer->h) {
                                    sum += datai[i * ilayer->w * ilayer->h + ix + iy * ilayer->w] * dataf[c * ftsize + i * ilayer->fs * ilayer->fs + k + j * ilayer->fs];
                                }
                            }
                        }
                    }
                    datao[c * olayer->w * olayer->h + y * olayer->w + x] = activate(sum * dataf[c * ftsize + walign + 0] + dataf[c * ftsize + walign + 1], ilayer->activation);
                }
            }
        }
        datai += ilayer->w * ilayer->h * ic;
        datao += olayer->w * olayer->h * oc;
        dataf += ftsize * fn;
    }
}
