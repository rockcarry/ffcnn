#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "ffcnn.h"

void matrix_multiply(MATRIX *mr, MATRIX *m1, MATRIX *m2)
{
    int    r, c, n;
    float *d1, *d2, *dr;
    d1 = m1->data;
    d2 = m2->data;
    dr = mr->data;
    for (r=0; r<mr->rows; r++) {
        for (c=0; c<mr->cols; c++) {
            for (dr[c]=0,n=0; n<m1->cols; n++) {
                dr[c] += d1[n] * d2[c+m2->cols*n];
            }
        }
        d1 += m1->cols;
        dr += mr->cols;
    }
}

void matrix_add(MATRIX *mr, MATRIX *m1, MATRIX *m2)
{
    int n = mr->rows * mr->cols, i;
    for (i=0; i<n; i++) mr->data[i] = m1->data[i] + m2->data[i];
}

void matrix_sub(MATRIX *mr, MATRIX *m1, MATRIX *m2)
{
    int n = mr->rows * mr->cols, i;
    for (i=0; i<n; i++) mr->data[i] = m1->data[i] - m2->data[i];
}

void matrix_scale(MATRIX *mr, MATRIX *m1, float s)
{
    int n = mr->rows * mr->cols, i;
    for (i=0; i<n; i++) mr->data[i] = m1->data[i] * s;
}

void matrix_upsample(MATRIX *mr, MATRIX *m1, int stride)
{
    int i, j;
    for (j=0; j<mr->rows; j++) {
        for (i=0; i<mr->cols; i++) {
            m1->data[j * mr->cols + i] = m1->data[(j / stride) * m1->cols + i / stride];
        }
    }
}

static float filter_conv(MATRIX *m, int x, int y, FILTER *f)
{
    float val = 0;
    int   i, j;
    for (j=0; j<f->rows; j++) {
        for (i=0; i<f->cols; i++) {
            val += m->data[(y + j) * m->cols + x + i] * f->data[j * f->cols + i];
        }
    }
    return val;
}

static float filter_avg(MATRIX *m, int x, int y, int w, int h)
{
    float val = 0;
    int   i, j;
    for (j=0; j<h; j++) {
        for (i=0; i<w; i++) {
            val += m->data[(y + j) * m->cols + x + i];
        }
    }
    return val / (w * h);
}

static float filter_max(MATRIX *m, int x, int y, int w, int h)
{
    float val = m->data[y * m->cols + x];
    int   i, j;
    for (j=0; j<h; j++) {
        for (i=0; i<w; i++) {
            if (val < m->data[(y + j) * m->cols + x + i]) val = m->data[(y + j) * m->cols + x + i];
        }
    }
    return val;
}

float filter(MATRIX *m, int x, int y, FILTER *f)
{
    switch (f->type) {
    case FILTER_TYPE_CONV: filter_conv(m, x, y, f);
    case FILTER_TYPE_AVG : filter_avg (m, x, y, f->cols, f->rows);
    case FILTER_TYPE_MAX : filter_max (m, x, y, f->cols, f->rows);
    default: return 0;
    }
}

float activate(float x, int type)
{
    switch (type) {
    case ACTIVATE_TYPE_RELU  : return x > 0 ? x : 0;
    case ACTIVATE_TYPE_LEAKY : return x > 0 ? x : 0.1f * x;
    default: return x;
    }
}

static void layer_filter_forward(LAYER *ilayer, LAYER *olayer)
{
    MATRIX *im, *om; FILTER *flt;
    int     in, on, iw, ih, fw, fh, fs, ix, iy, ox, oy, i, j, px, py;
    float   val;
    in = ilayer->matrix_num;
    on = olayer->matrix_num;
    iw = ilayer->matrix_list[0].cols;
    ih = ilayer->matrix_list[0].rows;
    fw = ilayer->filter_list[0].cols;
    fh = ilayer->filter_list[0].cols;
    fs = ilayer->stride;
    px = olayer->pad ? olayer->filter_list[0].cols / 2 : 0;
    py = olayer->pad ? olayer->filter_list[0].rows / 2 : 0;
    for (iy=0,oy=py; iy+fh<ih; iy+=fs,oy++) {
        for (ix=0,ox=px; ix+fw<iw; ix+=fs,ox++) {
            for (j=0; j<on; j++) {
                om = ilayer->matrix_list + j;
                for (i=0; i<in; i++) {
                    im  = ilayer->matrix_list + i;
                    flt = ilayer->filter_list + j * in + i;
                    val = filter(im, ix, iy, flt);
                    if (!i) om->data[oy * om->cols + ox] = val + ilayer->fbias_list[j];
                    else    om->data[oy * om->cols + ox]+= val;
                    if (i == in - 1) om->data[oy * om->cols + ox] = activate(om->data[oy * om->cols + ox], ilayer->activate);
                }
            }
        }
    }
}

static void layer_upsample_forward(LAYER *ilayer, LAYER *olayer)
{
    int i; for (i=0; i<ilayer->matrix_num; i++) matrix_upsample(olayer->matrix_list + i, ilayer->matrix_list + i, ilayer->stride);
}

static void layer_shortcut_forward(LAYER *ilayer, LAYER *olayer)
{
    LAYER *slayer = ilayer + ilayer->lshortcut;
    int  i, j, n = olayer->matrix_list[0].cols * olayer->matrix_list[0].rows;
    for (i=0; i<olayer->matrix_num; i++) {
        float *po = olayer->matrix_list[i].data;
        float *pi = ilayer->matrix_list[i].data;
        float *ps = slayer->matrix_list[i].data;
        for (j=0; j<n; j++) {
            *po = *pi + *ps;
            *po = activate(*po, ilayer->activate);
             pi++, ps++, po++;
        }
    }
}

static void layer_route_forward(LAYER *ilayer, LAYER *olayer)
{
    int  i, j, k = 0;
    for (i=0; i<ilayer->route_num; i++) {
        LAYER *rlayer = ilayer + ilayer->route_list[i];
        for (j=0; j<rlayer->matrix_num; j++) {
            memcpy(olayer->matrix_list[k].data, rlayer->matrix_list[j].data, olayer->matrix_list[k].cols * olayer->matrix_list[k].rows * sizeof(float)); k++;
        }
    }
}

void layer_forward(LAYER *ilayer, LAYER *olayer)
{
    switch (ilayer->type) {
    case LAYER_TYPE_CONV   :
    case LAYER_TYPE_MAXPOOL:
    case LAYER_TYPE_AVGPOOL:
        layer_filter_forward(ilayer, olayer); break;
    case LAYER_TYPE_UPSAMPLE: layer_upsample_forward(ilayer, olayer); break;
    case LAYER_TYPE_SHORTCUT: layer_shortcut_forward(ilayer, olayer); break;
    case LAYER_TYPE_ROUTE   : layer_route_forward   (ilayer, olayer); break;
    }
}

int main(void)
{
    return 0;
}
