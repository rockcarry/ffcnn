#ifndef _FFCNN_H_
#define _FFCNN_H_

typedef struct {
    int    width, height, channels, padw, padh;
    float *data;
} MATRIX;

typedef struct {
    int    width, height, channels, n;
    float *data, *bias, *scale, *rolling_mean, *rolling_variance;
} FILTER;

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

    int     class_num;
    int     anchor_num;
    int     anchor_list[9][2];
    float   scale_x_y;
    float   ignore_thres;
} LAYER;

typedef struct {
    LAYER *layer_list;
    int    layer_num;
    int    weight_size;
    float *weight_buf;
} NET;

NET* net_load   (char *file1, char *file2);
void net_free   (NET *net);
void net_input  (NET *net, unsigned char *bgr, int w, int h, float *mean, float *norm);
void net_forward(NET *net);
void net_dump   (NET *net);

#endif
