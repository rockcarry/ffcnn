#ifndef _MATRIX_H_
#define _MATRIX_H_

#define MATRIX_COMMON_MEMBERS \
    int    rows, cols; \
    float *data;

typedef struct {
    MATRIX_COMMON_MEMBERS
} MATRIX;

MATRIX* matrix_create  (int rows, int cols);
void    matrix_destroy (MATRIX *m);
void    matrix_multiply(MATRIX *mr, MATRIX *m1, MATRIX *m2);
void    matrix_add     (MATRIX *m1, MATRIX *m2);
void    matrix_sub     (MATRIX *m1, MATRIX *m2);
void    matrix_scale   (MATRIX *m1, float s);
void    matrix_upsample(MATRIX *m1, MATRIX *m2, int stride);

enum {
    FILTER_TYPE_CONV,
    FILTER_TYPE_AVG ,
    FILTER_TYPE_MAX ,
};

typedef struct {
    MATRIX_COMMON_MEMBERS
    int    type;
    float  bias;
} FILTER;
float filter(MATRIX *m, int x, int y, FILTER *f);

enum {
    ACTIVATE_TYPE_LINEAR,
    ACTIVATE_TYPE_RELU  ,
    ACTIVATE_TYPE_LEAKY ,
};
float activate(float x, int type);

enum {
    LAYER_TYPE_CONV    ,
    LAYER_TYPE_SHORTCUT,
    LAYER_TYPE_MAXPOOL ,
    LAYER_TYPE_AVGPOOL ,
    LAYER_TYPE_ROUTE   ,
    LAYER_TYPE_UPSAMPLE,
    LAYER_TYPE_YOLO    ,
};

typedef struct {
    int     type;
    MATRIX *matrix_list;
    int     matrix_num ;
    FILTER *filter_list;
    int     filter_num ;
} LAYER;
void layer_forward(LAYER *layercur, LAYER *layernext);

typedef struct {
    LAYER *layer_list;
    int    layer_num ;
} NET;
NET* net_load   (char *file1, char *file2);
void net_free   (NET  *net );
void net_forward(NET  *net );

#endif
