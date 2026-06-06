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

// 1x1 convolution - process all channels at once, outside group loop
// Memory layout: datai[ic][y][x], datao[oc][y][x], dataf[oc][ic]
static void convolution_pad0_fs1_stride1_all(float *restrict datai, float *restrict dataf, float *restrict datao,
                        int iw, int ih, int ic, int oc, int activation)
{
    const int n = iw * ih;

    // Main loop: process 4 output channels at a time
    int c = 0;
    for (; c + 3 < oc; c += 4) {
        const float *wt0 = dataf + (c  ) * (ic + 4);
        const float *wt1 = dataf + (c+1) * (ic + 4);
        const float *wt2 = dataf + (c+2) * (ic + 4);
        const float *wt3 = dataf + (c+3) * (ic + 4);
        float s0 = wt0[ic + 0], b0 = wt0[ic + 1];
        float s1 = wt1[ic + 0], b1 = wt1[ic + 1];
        float s2 = wt2[ic + 0], b2 = wt2[ic + 1];
        float s3 = wt3[ic + 0], b3 = wt3[ic + 1];

        for (int pix = 0; pix < n; pix++) {
            float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
            for (int ci = 0; ci < ic; ci++) {
                float val = datai[ci * n + pix];
                sum0 += wt0[ci] * val;
                sum1 += wt1[ci] * val;
                sum2 += wt2[ci] * val;
                sum3 += wt3[ci] * val;
            }
            datao[(c  ) * n + pix] = activate(sum0 * s0 + b0, activation);
            datao[(c+1) * n + pix] = activate(sum1 * s1 + b1, activation);
            datao[(c+2) * n + pix] = activate(sum2 * s2 + b2, activation);
            datao[(c+3) * n + pix] = activate(sum3 * s3 + b3, activation);
        }
    }

    // Remaining output channels (1, 2, or 3)
    for (; c < oc; c++) {
        const float *wt = dataf + c * (ic + 4);
        float sc = wt[ic + 0], bi = wt[ic + 1];
        for (int pix = 0; pix < n; pix++) {
            float sum = 0;
            for (int ci = 0; ci < ic; ci++) {
                sum += wt[ci] * datai[ci * n + pix];
            }
            datao[c * n + pix] = activate(sum * sc + bi, activation);
        }
    }
}

// Specialized convolution for pad=1, fs=3, stride=1, depthwise (gc_ic == 1)
// No im2row needed - directly compute output from input neighborhood
// Process all channels at once to avoid groupconv loop overhead
static void convolution_pad1_fs3_stride1_depthwise(float *restrict datai, float *restrict dataf, float *restrict datao,
                        int iw, int ih, int ic, int ftsize, int activation)
{
    const int ow = iw, oh = ih;
    for (int c = 0; c < ic; c++) {
        const float *in  = datai + c * ih * iw;
        float       *out = datao + c * oh * ow;
        const float *wt  = dataf + c * (ftsize + 4);
        float w0 = wt[0], w1 = wt[1], w2 = wt[2];
        float w3 = wt[3], w4 = wt[4], w5 = wt[5];
        float w6 = wt[6], w7 = wt[7], w8 = wt[8];
        float scale = wt[ftsize + 0];
        float bias  = wt[ftsize + 1];

        // Top row (y=0): no row above
        {
            const float *r0 = in;
            const float *r1 = in + iw;
            float *o = out;
            if (ow == 1) {
                float s = w4*r0[0] + w7*r1[0];
                o[0] = activate(s * scale + bias, activation);
            } else {
                // x=0
                float s = w4*r0[0] + w5*r0[1] + w7*r1[0] + w8*r1[1];
                o[0] = activate(s * scale + bias, activation);
                // x=1..ow-2 (branch-free)
                for (int x = 1; x < ow - 1; x++) {
                    s = w3*r0[x-1] + w4*r0[x] + w5*r0[x+1]
                      + w6*r1[x-1] + w7*r1[x] + w8*r1[x+1];
                    o[x] = activate(s * scale + bias, activation);
                }
                // x=ow-1
                int x = ow - 1;
                s = w3*r0[x-1] + w4*r0[x] + w6*r1[x-1] + w7*r1[x];
                o[x] = activate(s * scale + bias, activation);
            }
        }

        // Middle rows
        for (int y = 1; y < oh - 1; y++) {
            const float *rm1 = in + (y - 1) * iw;
            const float *r0  = in + y * iw;
            const float *rp1 = in + (y + 1) * iw;
            float *o = out + y * ow;

            if (ow == 1) {
                float s = w1*rm1[0] + w4*r0[0] + w7*rp1[0];
                o[0] = activate(s * scale + bias, activation);
                continue;
            }

            // x=0
            float s = w1*rm1[0] + w2*rm1[1]
                    + w4*r0 [0] + w5*r0 [1]
                    + w7*rp1[0] + w8*rp1[1];
            o[0] = activate(s * scale + bias, activation);

            if (ow == 2) {
                int x = 1;
                s = w0*rm1[x-1] + w1*rm1[x]
                  + w3*r0 [x-1] + w4*r0 [x]
                  + w6*rp1[x-1] + w7*rp1[x];
                o[x] = activate(s * scale + bias, activation);
                continue;
            }

            // x=1
            {
                int x = 1;
                s = w0*rm1[x-1] + w1*rm1[x] + w2*rm1[x+1]
                  + w3*r0 [x-1] + w4*r0 [x] + w5*r0 [x+1]
                  + w6*rp1[x-1] + w7*rp1[x] + w8*rp1[x+1];
                o[x] = activate(s * scale + bias, activation);
            }

            // Center: x=2..ow-3, process 2 pixels at a time for better ILP
            int x = 2;
            for (; x + 1 < ow - 1; x += 2) {
                float s0 = w0*rm1[x-1] + w1*rm1[x] + w2*rm1[x+1]
                         + w3*r0 [x-1] + w4*r0 [x] + w5*r0 [x+1]
                         + w6*rp1[x-1] + w7*rp1[x] + w8*rp1[x+1];
                float s1 = w0*rm1[x]   + w1*rm1[x+1] + w2*rm1[x+2]
                         + w3*r0 [x]   + w4*r0 [x+1] + w5*r0 [x+2]
                         + w6*rp1[x]   + w7*rp1[x+1] + w8*rp1[x+2];
                o[x]   = activate(s0 * scale + bias, activation);
                o[x+1] = activate(s1 * scale + bias, activation);
            }

            // Remaining center pixel(s)
            for (; x < ow - 1; x++) {
                s = w0*rm1[x-1] + w1*rm1[x] + w2*rm1[x+1]
                  + w3*r0 [x-1] + w4*r0 [x] + w5*r0 [x+1]
                  + w6*rp1[x-1] + w7*rp1[x] + w8*rp1[x+1];
                o[x] = activate(s * scale + bias, activation);
            }

            // x=ow-1
            {
                int xe = ow - 1;
                s = w0*rm1[xe-1] + w1*rm1[xe]
                  + w3*r0 [xe-1] + w4*r0 [xe]
                  + w6*rp1[xe-1] + w7*rp1[xe];
                o[xe] = activate(s * scale + bias, activation);
            }
        }

        // Bottom row (y=oh-1): no row below
        if (oh > 1) {
            int y = oh - 1;
            const float *rm1 = in + (y - 1) * iw;
            const float *r0  = in + y * iw;
            float *o = out + y * ow;
            if (ow == 1) {
                float s = w1*rm1[0] + w4*r0[0];
                o[0] = activate(s * scale + bias, activation);
            } else {
                // x=0
                float s = w1*rm1[0] + w2*rm1[1] + w4*r0[0] + w5*r0[1];
                o[0] = activate(s * scale + bias, activation);
                // x=1..ow-2
                for (int x = 1; x < ow - 1; x++) {
                    s = w0*rm1[x-1] + w1*rm1[x] + w2*rm1[x+1]
                      + w3*r0 [x-1] + w4*r0 [x] + w5*r0 [x+1];
                    o[x] = activate(s * scale + bias, activation);
                }
                // x=ow-1
                int x = ow - 1;
                s = w0*rm1[x-1] + w1*rm1[x] + w3*r0[x-1] + w4*r0[x];
                o[x] = activate(s * scale + bias, activation);
            }
        }
    }
}

// Specialized convolution for pad=1, fs=3, stride=2, depthwise (gc_ic == 1)
// No im2row needed - directly compute output from input neighborhood
static void convolution_pad1_fs3_stride2_depthwise(float *restrict datai, float *restrict dataf, float *restrict datao,
                        int iw, int ih, int ic, int ftsize, int activation)
{
    int ow = (iw + 1) / 2;
    int oh = (ih + 1) / 2;

    for (int c = 0; c < ic; c++) {
        const float *in  = datai + c * ih * iw;
        float       *out = datao + c * oh * ow;
        const float *wt  = dataf + c * (ftsize + 4);
        float w0 = wt[0], w1 = wt[1], w2 = wt[2];
        float w3 = wt[3], w4 = wt[4], w5 = wt[5];
        float w6 = wt[6], w7 = wt[7], w8 = wt[8];
        float scale = wt[ftsize + 0];
        float bias  = wt[ftsize + 1];

        for (int oy = 0; oy < oh; oy++) {
            int iy = oy * 2 - 1;
            float *o = out + oy * ow;

            int has_t = (iy >= 0);
            int has_b = (iy + 2 < ih);

            const float *rt = has_t ? in + iy * iw : NULL;
            const float *rm = in + (iy + 1) * iw;
            const float *rb = has_b ? in + (iy + 2) * iw : NULL;

            for (int ox = 0; ox < ow; ox++) {
                int ix = ox * 2 - 1;
                int has_l = (ix >= 0);
                int has_r = (ix + 2 < iw);

                float s = 0;

                if (has_t) {
                    if (has_l)  s += w0 * rt[ix];
                    s += w1 * rt[ix + 1];
                    if (has_r)  s += w2 * rt[ix + 2];
                }

                if (has_l)  s += w3 * rm[ix];
                s += w4 * rm[ix + 1];
                if (has_r)  s += w5 * rm[ix + 2];

                if (has_b) {
                    if (has_l)  s += w6 * rb[ix];
                    s += w7 * rb[ix + 1];
                    if (has_r)  s += w8 * rb[ix + 2];
                }

                o[ox] = activate(s * scale + bias, activation);
            }
        }
    }
}

// Specialized convolution for pad=2, fs=5, stride=1, depthwise (gc_ic == 1)
// No im2row needed - directly compute output from input neighborhood
static void convolution_pad2_fs5_stride1_depthwise(float *restrict datai, float *restrict dataf, float *restrict datao,
                        int iw, int ih, int ic, int ftsize, int activation)
{
    const int ow = iw, oh = ih;
    for (int c = 0; c < ic; c++) {
        const float *in  = datai + c * ih * iw;
        float       *out = datao + c * oh * ow;
        const float *wt  = dataf + c * (ftsize + 4);
        float scale = wt[ftsize + 0];
        float bias  = wt[ftsize + 1];

        // Helper macro for a single 5x5 dot product with valid rows and variable x range
        // rs..re are relative row offsets (-2..+2), xc is center x, x0/x1 are inclusive bounds
        #define F5X5_DOT(rs, re, xc, x0, x1)                                    \
            do {                                                                \
                float _s = 0;                                                   \
                for (int _r = (rs); _r <= (re); _r++) {                         \
                    const float *_row = in + ((yc) + _r) * iw;                  \
                    int _wt_base = (_r + 2) * 5;                                \
                    for (int _x = (x0); _x <= (x1); _x++) {                     \
                        _s += _row[_x] * wt[_wt_base + ((_x) - (xc) + 2)];      \
                    }                                                           \
                }                                                               \
                outrow[xc] = activate(_s * scale + bias, activation);           \
            } while (0)

        // Row y=0: only rows 0,1,2 available (relative -2,-1 out of bounds)
        {
            float *outrow = out;
            const float *r0 = in + 0 * iw;
            const float *r1 = in + 1 * iw;
            const float *r2 = in + 2 * iw;
            for (int x = 0; x < ow; x++) {
                int x0 = (x > 1) ? x - 2 : 0;
                int x1 = (x < ow - 2) ? x + 2 : ow - 1;
                float s = 0;
                for (int k = x0; k <= x1; k++) {
                    int widx = (k - x + 2);
                    s += r0[k] * wt[10 + widx];
                    s += r1[k] * wt[15 + widx];
                    s += r2[k] * wt[20 + widx];
                }
                outrow[x] = activate(s * scale + bias, activation);
            }
        }

        // Row y=1: rows 0,1,2,3 available (relative -2 out of bounds)
        if (oh > 1) {
            int yc = 1;
            float *outrow = out + yc * ow;
            const float *r0 = in + 0 * iw;
            const float *r1 = in + 1 * iw;
            const float *r2 = in + 2 * iw;
            const float *r3 = in + 3 * iw;
            for (int x = 0; x < ow; x++) {
                int x0 = (x > 1) ? x - 2 : 0;
                int x1 = (x < ow - 2) ? x + 2 : ow - 1;
                float s = 0;
                for (int k = x0; k <= x1; k++) {
                    int widx = (k - x + 2);
                    s += r0[k] * wt[5  + widx];
                    s += r1[k] * wt[10 + widx];
                    s += r2[k] * wt[15 + widx];
                    s += r3[k] * wt[20 + widx];
                }
                outrow[x] = activate(s * scale + bias, activation);
            }
        }

        // Middle rows: all 5 rows available
        for (int y = 2; y < oh - 2; y++) {
            float *outrow = out + y * ow;
            const float *rm2 = in + (y - 2) * iw;
            const float *rm1 = in + (y - 1) * iw;
            const float *r0  = in + y * iw;
            const float *rp1 = in + (y + 1) * iw;
            const float *rp2 = in + (y + 2) * iw;

            // x = 0 (left edge, no x-2, x-1)
            {
                float s = rm2[0]*wt[2] + rm2[1]*wt[3] + rm2[2]*wt[4]
                        + rm1[0]*wt[7] + rm1[1]*wt[8] + rm1[2]*wt[9]
                        + r0 [0]*wt[12]+ r0 [1]*wt[13]+ r0 [2]*wt[14]
                        + rp1[0]*wt[17]+ rp1[1]*wt[18]+ rp1[2]*wt[19]
                        + rp2[0]*wt[22]+ rp2[1]*wt[23]+ rp2[2]*wt[24];
                outrow[0] = activate(s * scale + bias, activation);
            }

            // x = 1 (one left neighbor missing)
            if (ow > 1) {
                float s = rm2[0]*wt[1] + rm2[1]*wt[2] + rm2[2]*wt[3] + rm2[3]*wt[4]
                        + rm1[0]*wt[6] + rm1[1]*wt[7] + rm1[2]*wt[8] + rm1[3]*wt[9]
                        + r0 [0]*wt[11]+ r0 [1]*wt[12]+ r0 [2]*wt[13]+ r0 [3]*wt[14]
                        + rp1[0]*wt[16]+ rp1[1]*wt[17]+ rp1[2]*wt[18]+ rp1[3]*wt[19]
                        + rp2[0]*wt[21]+ rp2[1]*wt[22]+ rp2[2]*wt[23]+ rp2[3]*wt[24];
                outrow[1] = activate(s * scale + bias, activation);
            }

            // Center columns (no boundary checks)
            for (int x = 2; x < ow - 2; x++) {
                float s = rm2[x-2]*wt[0] + rm2[x-1]*wt[1] + rm2[x]*wt[2] + rm2[x+1]*wt[3] + rm2[x+2]*wt[4]
                        + rm1[x-2]*wt[5] + rm1[x-1]*wt[6] + rm1[x]*wt[7] + rm1[x+1]*wt[8] + rm1[x+2]*wt[9]
                        + r0 [x-2]*wt[10]+ r0 [x-1]*wt[11]+ r0 [x]*wt[12]+ r0 [x+1]*wt[13]+ r0 [x+2]*wt[14]
                        + rp1[x-2]*wt[15]+ rp1[x-1]*wt[16]+ rp1[x]*wt[17]+ rp1[x+1]*wt[18]+ rp1[x+2]*wt[19]
                        + rp2[x-2]*wt[20]+ rp2[x-1]*wt[21]+ rp2[x]*wt[22]+ rp2[x+1]*wt[23]+ rp2[x+2]*wt[24];
                outrow[x] = activate(s * scale + bias, activation);
            }

            // x = ow-2 (one right neighbor missing)
            if (ow > 3) {
                int x = ow - 2;
                float s = rm2[x-2]*wt[0] + rm2[x-1]*wt[1] + rm2[x]*wt[2] + rm2[x+1]*wt[3]
                        + rm1[x-2]*wt[5] + rm1[x-1]*wt[6] + rm1[x]*wt[7] + rm1[x+1]*wt[8]
                        + r0 [x-2]*wt[10]+ r0 [x-1]*wt[11]+ r0 [x]*wt[12]+ r0 [x+1]*wt[13]
                        + rp1[x-2]*wt[15]+ rp1[x-1]*wt[16]+ rp1[x]*wt[17]+ rp1[x+1]*wt[18]
                        + rp2[x-2]*wt[20]+ rp2[x-1]*wt[21]+ rp2[x]*wt[22]+ rp2[x+1]*wt[23];
                outrow[x] = activate(s * scale + bias, activation);
            }

            // x = ow-1 (right edge, no x+1, x+2)
            if (ow > 2) {
                int x = ow - 1;
                float s = rm2[x-2]*wt[0] + rm2[x-1]*wt[1] + rm2[x]*wt[2]
                        + rm1[x-2]*wt[5] + rm1[x-1]*wt[6] + rm1[x]*wt[7]
                        + r0 [x-2]*wt[10]+ r0 [x-1]*wt[11]+ r0 [x]*wt[12]
                        + rp1[x-2]*wt[15]+ rp1[x-1]*wt[16]+ rp1[x]*wt[17]
                        + rp2[x-2]*wt[20]+ rp2[x-1]*wt[21]+ rp2[x]*wt[22];
                outrow[x] = activate(s * scale + bias, activation);
            }
        }

        // Row y=oh-2: rows oh-4..oh-1 available (relative +2 out of bounds)
        if (oh > 2) {
            int yc = oh - 2;
            float *outrow = out + yc * ow;
            const float *rm1 = in + (yc - 1) * iw;
            const float *r0  = in + yc * iw;
            const float *rp1 = in + (yc + 1) * iw;
            for (int x = 0; x < ow; x++) {
                int x0 = (x > 1) ? x - 2 : 0;
                int x1 = (x < ow - 2) ? x + 2 : ow - 1;
                float s = 0;
                for (int k = x0; k <= x1; k++) {
                    int widx = (k - x + 2);
                    s += rm1[k] * wt[5  + widx];
                    s += r0 [k] * wt[10 + widx];
                    s += rp1[k] * wt[15 + widx];
                }
                outrow[x] = activate(s * scale + bias, activation);
            }
        }

        // Row y=oh-1: rows oh-3..oh-1 available (relative +1,+2 out of bounds)
        if (oh > 1) {
            int yc = oh - 1;
            float *outrow = out + yc * ow;
            const float *rm2 = in + (yc - 2) * iw;
            const float *rm1 = in + (yc - 1) * iw;
            const float *r0  = in + yc * iw;
            for (int x = 0; x < ow; x++) {
                int x0 = (x > 1) ? x - 2 : 0;
                int x1 = (x < ow - 2) ? x + 2 : ow - 1;
                float s = 0;
                for (int k = x0; k <= x1; k++) {
                    int widx = (k - x + 2);
                    s += rm2[k] * wt[0 + widx];
                    s += rm1[k] * wt[5 + widx];
                    s += r0 [k] * wt[10 + widx];
                }
                outrow[x] = activate(s * scale + bias, activation);
            }
        }
    }
    #undef F5X5_DOT
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

    // Fast path for 1x1 stride=1: no buffer, no per-group loop
    if (ipad == 0 && fs == 1 && istride == 1 && ig == 1) {
        convolution_pad0_fs1_stride1_all(datai, dataf, datao, iw, ih, ic, oc, activation);
        return;
    }

    // Fast path for depthwise 3x3 stride=1: no im2row buffer, no per-group loop
    if (ipad == 1 && fs == 3 && istride == 1 && gc_ic == 1) {
        convolution_pad1_fs3_stride1_depthwise(datai, dataf, datao, iw, ih, ic, ftsize, activation);
        return;
    }

    // Fast path for depthwise 3x3 stride=2: no im2row buffer, no per-group loop
    if (ipad == 1 && fs == 3 && istride == 2 && gc_ic == 1) {
        convolution_pad1_fs3_stride2_depthwise(datai, dataf, datao, iw, ih, ic, ftsize, activation);
        return;
    }

    // Fast path for depthwise 5x5 stride=1: no im2row buffer, no per-group loop
    if (ipad == 2 && fs == 5 && istride == 1 && gc_ic == 1) {
        convolution_pad2_fs5_stride1_depthwise(datai, dataf, datao, iw, ih, ic, ftsize, activation);
        return;
    }

    // Single-core: only need buffer for one y (ftsize * ow)
    if (*gc_bufsize < ftsize * ow) {
        *gc_bufsize = ftsize * ow;
        free(*gc_buffer);
        *gc_buffer = (float*)malloc(*gc_bufsize * sizeof(float));
        if (*gc_buffer == NULL) { printf("failed to allocate memory for cnntempbuf !\n"); return; }
    }

    for (int g = 0; g < ig; g++) {
        convolution_generic(datai, dataf, datao, iw, ih, gc_ic, ipad, istride, fs, ftsize, ow, oh, gc_oc, activation, *gc_buffer);
        datai += (long)iw * ih * gc_ic;
        datao += (long)ow * oh * gc_oc;
        dataf += (long)(ftsize + 4) * gc_fn;
    }
}
