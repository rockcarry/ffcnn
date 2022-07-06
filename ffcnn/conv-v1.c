#include <stdlib.h>
#include <stdio.h>
#include "conv.h"

static void im2row(float *img, int w, int h, int c, int pad, int fs, int stride, int walign, int y, float *buf)
{
    float *src = img + (y - pad) * w - pad, *dst = buf;
    int    i, j, k, x;
    for (i=0; i<c; i++) {
        for (j=0; j<fs; j++) {
            float *tmp = dst;
            for (x=0; x<w; x+=stride) {
                for (k=0; k<fs; k++) dst[k] = (unsigned)(x - pad + k) < (unsigned)w && (unsigned)(y - pad + j) < (unsigned)h ? src[k] : 0;
                src += stride, dst += walign;
            }
            src += w - x, dst = tmp + fs;
        }
        src += w * (h - fs);
    }
}

void layer_groupconv_forward(NET *net, LAYER *ilayer, LAYER *olayer)
{
    int    walign, ftsize, x, y, g, c, ic, oc, fn, i;
    float *datai, *datao, *dataf, sum;

    datai  = ilayer->data;
    datao  = olayer->data;
    dataf  = ilayer->filter;
    ic     = ilayer->c / ilayer->groups;
    oc     = olayer->c / ilayer->groups;
    fn     = ilayer->fn/ ilayer->groups;
    walign = ALIGN(ilayer->fs * ilayer->fs * ic, 4);
    ftsize = walign + 4;
    if (net->cnnbufsize < walign * olayer->w) {
        net->cnnbufsize = walign * olayer->w;
        free(net->cnntempbuf); net->cnntempbuf = malloc(net->cnnbufsize * sizeof(float));
        if (net->cnntempbuf == NULL) { printf("failed to allocate memory for cnntempbuf !"); return; }
    }

    for (g=0; g<ilayer->groups; g++) {
        for (y=0; y<olayer->h; y++) {
            im2row(datai, ilayer->w, ilayer->h, ic, ilayer->pad, ilayer->fs, ilayer->stride, walign, y * ilayer->stride, net->cnntempbuf);
            for (x=0; x<olayer->w; x++) {
                for (c=0; c<oc; c++) {
                    for (sum=0,i=0; i<walign; i++) sum += dataf[c * ftsize + i] * net->cnntempbuf[x * walign + i];
                    datao[c * olayer->w * olayer->h + y * olayer->w + x] = activate(sum * dataf[c * ftsize + walign + 0] + dataf[c * ftsize + walign + 1], ilayer->activation);
                }
            }
        }
        datai += ilayer->w * ilayer->h * ic;
        datao += olayer->w * olayer->h * oc;
        dataf += ftsize * fn;
    }
}
