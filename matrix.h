#ifndef _MATRIX_H_
#define _MATRIX_H_

typedef struct {
    int   rows, cols;
    float data;
} MATRIX;

MATRIX* matrix_create  (int rows, int cols);
void    matrix_destroy (MATRIX *m);
void    matrix_multiply(MATRIX *mr, MATRIX *m1, MATRIX *m2);
void    matrix_add     (MATRIX *m1, MATRIX *m2);
void    matrix_sub     (MATRIX *m1, MATRIX *m2);
void    matrix_scale   (MATRIX *m1, float s);
void    matrix_upsample(MATRIX *m1, MATRIX *m2, int stride);
float   filter_conv(MATRIX *m, int x, int y, MATRIX *f);
float   filter_avg (MATRIX *m, int x, int y, int w, int h);
float   filter_max (MATRIX *m, int x, int y, int w, int h);

#endif
