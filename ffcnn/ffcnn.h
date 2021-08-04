#ifndef _FFCNN_H_
#define _FFCNN_H_

typedef struct {
    int    width, height, channels;
    float *data;
} MATRIX;

enum {
    FILTER_TYPE_CONV,
    FILTER_TYPE_AVG ,
    FILTER_TYPE_MAX ,
};
typedef struct {
    int    type, width, height, channels, n;
    float *data, bias;
} FILTER;

enum {
    ACTIVATE_TYPE_LINEAR,
    ACTIVATE_TYPE_RELU  ,
    ACTIVATE_TYPE_LEAKY ,
};
float activate(float x, int type);

enum {
    LAYER_TYPE_CONV    ,
    LAYER_TYPE_AVGPOOL ,
    LAYER_TYPE_MAXPOOL ,
    LAYER_TYPE_UPSAMPLE,
    LAYER_TYPE_DROPOUT ,
    LAYER_TYPE_SHORTCUT,
    LAYER_TYPE_ROUTE   ,
    LAYER_TYPE_YOLO    ,
};
typedef struct {
    int     type;
    int     refcnt;
    MATRIX  matrix;
    FILTER  filter;
    int     stride, pad, groups;
    int     batchnorm, activate;
    int     depend_list[4];
    int     depend_num;
} LAYER;
void layer_forward(LAYER *ilaye, LAYER *olayer);

typedef struct {
    LAYER *layer_list;
    int    layer_num;
    int    weight_size;
    float *weight_buf;
} NET;
NET* net_load   (char *file1, char *file2);
void net_free   (NET *net);
void net_forward(NET *net);
void net_dump   (NET *net);

#endif
