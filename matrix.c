#include <stdlib.h>
#include <stdio.h>
#include "matrix.h"

MATRIX* matrix_create(int rows, int cols)
{
    MATRIX *matrix = malloc(sizeof(MATRIX) + rows * cols * sizeof(float));
    if (!matrix) {
        printf("matrix_create: failed to allocate memory !\n");
        return NULL;
    }
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->data = (double*)((uint8_t*)matrix + sizeof(MATRIX));
    return matrix;
}

void matrix_destroy(MATRIX *m) { free(m); }

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
    int n = m1->rows * m1->cols, i;
    for (i=0; i<n; i++) m1->data[i] += m2->data[i];
}

void matrix_sub(MATRIX *mr, MATRIX *m1, MATRIX *m2)
{
    int n = m1->rows * m1->cols, i;
    for (i=0; i<n; i++) m1->data[i] -= m2->data[i];
}

void matrix_scale(MATRIX *m1, float s)
{
    int n = m1->rows * m1->cols, i;
    for (i=0; i<n; i++) m1->data[i] *= s;
}

void matrix_upsample(MATRIX *m1, MATRIX *m2, int stride)
{
    int i, j;
    for (j=0; j<m1->rows; j++) {
        for (i=0; i<m1->cols; i++) {
            m1->data[j * m1->cols + i] = m2->data[(j / stride) * m2->cols + i / stride];
        }
    }
}

float filter_conv(MATRIX *m, int x, int y, MATRIX *f)
{
    float val = 0;
    int   i, j;
    for (j=0; j<f->rows; j++) {
        for (i=0; i<f->cols; i++) {
            val += m->data[(y + j) * m->cols + x + i)] * f->data[j * f->cols + i];
        }
    }
    return val;
}

float filter_avg(MATRIX *m, int x, int y, int w, int h)
{
    float val = 0;
    int   i, j;
    for (j=0; j<h; j++) {
        for (i=0; i<w; i++) {
            val += m->data[(y + j) * m->cols + x + i)];
        }
    }
    return val / (w * h);
}

float filter_max(MATRIX *m, int x, int y, int w, int h)
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

