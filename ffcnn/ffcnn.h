#ifndef _FFCNN_H_
#define _FFCNN_H_

enum {
    LAYER_TYPE_CONV    ,
    LAYER_TYPE_AVGPOOL ,
    LAYER_TYPE_MAXPOOL ,
    LAYER_TYPE_UPSAMPLE,
    LAYER_TYPE_DROPOUT ,
    LAYER_TYPE_SHORTCUT,
    LAYER_TYPE_ROUTE   ,
    LAYER_TYPE_YOLO    ,
    LAYER_TYPE_TOTOAL  ,
};

typedef struct {
    int     type, refcnt;
    float  *data, *filter;
    int     w, h, c, pad, stride, fn, fs, groups;
    int     batchnorm, activation;
    int     depend_list[4];
    int     depend_num;

    int     class_num;
    int     anchor_list[3][2];
    float   ignore_thres, scale_x_y;
} LAYER;

typedef struct {
    int   type;
    float score, x1, y1, x2, y2;
} BBOX;

typedef struct {
    LAYER *layer_list;
    int    layer_num;
    BBOX  *bbox_list;
    int    bbox_num;
    int    bbox_max;
    int    s1, s2;
    int    weight_size;
    float *weight_buf;
    float *cnntempbuf;
    int    cnnbufsize;
    int    timeused[LAYER_TYPE_TOTOAL];
} NET;

NET* net_load   (char *file1, char *file2);
void net_free   (NET *net);
void net_input  (NET *net, unsigned char *bgr, int w, int h, float *mean, float *norm);
void net_forward(NET *net);
void net_dump   (NET *net);

#endif
