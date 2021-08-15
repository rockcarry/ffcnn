#include <stdlib.h>
#include <stdio.h>
#include <pmmintrin.h>
#include "ffcnn.h"

void layer_groupconv_forward_sse3(NET *net, LAYER *ilayer, LAYER *olayer)
{
    MATRIX mati = ilayer->matrix;
    MATRIX mato = olayer->matrix;
    FILTER fltr = ilayer->filter;
    int    mwi  = mati.width + mati.pad * 2;
    int    mhi  = mati.height+ mati.pad * 2;
    int    mwo  = mato.width + mato.pad * 2;
    int    mho  = mato.height+ mato.pad * 2;
    int    walign, ftsize, x, y, i, j;
    float  sum[4];

    mato.data     += mato.pad * mwo + mato.pad;
    mato.channels /= fltr.groups;
    mati.channels /= fltr.groups;
    fltr.n        /= fltr.groups;
    walign         = ALIGN(fltr.size * fltr.size * fltr.channels, 4);
    ftsize         = walign + 4;
    if (net->cnnbufsize < walign) {
        net->cnnbufsize = walign;
        net->cnntempbuf = realloc(net->cnntempbuf, net->cnnbufsize * sizeof(float));
        if (net->cnntempbuf == NULL) { printf("failed to allocate memory for cnntempbuf !"); return; }
    }

    do {
        for (y=0; y<mato.height; y++) {
            for (x=0; x<mato.width; x++) {
                im2row(&mati, fltr.size, net->cnntempbuf); mati.data += fltr.stride;
                for (i=0; i<mato.channels; i++) {
                    __m128 a, b = _mm_setzero_ps();
                    for (j=0; j<walign; j+=4) {
                        a = _mm_mul_ps(_mm_loadu_ps(&fltr.data[i * ftsize + j]), _mm_loadu_ps(&net->cnntempbuf[j]));
                        b = _mm_add_ps(b, a);
                    }
                    b = _mm_hadd_ps(b, b);
                    b = _mm_hadd_ps(b, b);
                    _mm_storeu_ps(sum, b);
                    if (fltr.batchnorm) *sum = (*sum - fltr.data[i * ftsize + walign + 1]) * fltr.data[i * ftsize + walign + 2];
                    mato.data[i * mwo * mho + y * mwo + x] = activate(*sum + fltr.data[i * ftsize + walign + 0], fltr.activate);
                }
            }
            mati.data += (mwi - mato.width) * fltr.stride;
        }
        mati.data += mwi * (mhi * mati.channels - mato.height * fltr.stride);
        mato.data += mwo * mho * mato.channels;
        fltr.data += ftsize * fltr.n;
    } while (--fltr.groups);
}
