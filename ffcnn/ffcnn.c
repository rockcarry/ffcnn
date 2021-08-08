#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include "ffcnn.h"

#ifdef _MSC_VER
#pragma warning(disable:4996)
#define snprintf _snprintf
typedef int                int32_t;
typedef unsigned long long uint64_t;
#else
#include <stdint.h>
#endif

enum {
    ACTIVATE_TYPE_LINEAR ,
    ACTIVATE_TYPE_RELU   ,
    ACTIVATE_TYPE_LEAKY  ,
    ACTIVATE_TYPE_SIGMOID,
};

static float activate(float x, int type)
{
    switch (type) {
    case ACTIVATE_TYPE_RELU   : return x > 0 ? x : 0;
    case ACTIVATE_TYPE_LEAKY  : return x > 0 ? x : 0.1f * x;
    case ACTIVATE_TYPE_SIGMOID: return 1.0f / (1.0f + (float)exp(-x));
    default: return x;
    }
}

static float fast_inverse_sqrt(float x)
{
    union { float f; int32_t i; } fui;
    float halfx = 0.5f * x;
    fui.f = x;
    fui.i = 0x5F3759DF - (fui.i >> 1);
    x     = fui.f;
    x     = x * (1.5f - halfx * x * x);
    return x;
}

static void matrix_fill_pad(MATRIX *mat, float val)
{
    float *data = mat->data;
    int    pad  = mat->pad;
    int    mw   = mat->width + pad * 2;
    int    mh   = mat->height+ pad * 2;
    int    i, x, y;
    for (i=0; i<mat->channels; i++) {
        for (y=0; y<pad; y++) {
            for (x=0; x<mw ; x++) data[y * mw + x] = data[(mh - 1 - y) * mw + x] = val;
        }
        for (y=pad; y<mh-pad; y++) {
            for (x=0; x<pad; x++) data[y * mw + x] = data[y * mw + (mw - 1 - x)] = val;
        }
        data += mw * mh;
    }
}

static float filter_conv(float *mat, int mw, int x, int y, float *flt, int fw, int fh)
{
    float val = 0; int i, j;
    mat += y * mw + x;
    for (j=0; j<fh; j++) {
        for (i=0; i<fw; i++) {
            val += mat[j * mw + i] * flt[j * fw + i];
        }
    }
    return val;
}

static float filter_avgmax(float *mat, int mw, int x, int y, int fw, int fh, int flag)
{
    float val = 0, max; int i, j;
    mat += y * mw + x;
    max  = mat[0];
    for (j=0; j<fh; j++) {
        for (i=0; i<fw; i++) {
            if (flag) {
                if (max < mat[j * mw + i]) max = mat[j * mw + i];
            } else {
                val += mat[j * mw + i];
            }
        }
    }
    return flag ? max : val / (fw * fh);
}

static void layer_convolution_forward(LAYER *ilayer, LAYER *olayer)
{
    int  n, i, ix, iy, ox, oy, fw, fh, fs, mwi, mwo;
    float *datai, *datao, *dataf;
    fw  = ilayer->filter.width;
    fh  = ilayer->filter.height;
    fs  = ilayer->stride;
    mwi = ilayer->matrix.width + ilayer->matrix.pad * 2;
    mwo = olayer->matrix.width + olayer->matrix.pad * 2;

    datai = ilayer->matrix.data;
    for (i=0; i<ilayer->matrix.channels; i++) {
        for (iy=0,oy=0; iy<ilayer->matrix.width; iy+=fs,oy++) {
            for (ix=0,ox=0; ix<ilayer->matrix.width; ix+=fs,ox++) {
                datao = olayer->matrix.data + olayer->matrix.pad * mwo + olayer->matrix.pad;
                dataf = ilayer->filter.data + i * fw * fh;
                for (n=0; n<olayer->matrix.channels; n++) {
                    float val = filter_conv(datai, mwi, ix, iy, dataf, fw, fh);
                    if (!i) datao[oy * mwo + ox] = val;
                    else    datao[oy * mwo + ox]+= val;
                    if (i == ilayer->matrix.channels - 1) {
                        if (ilayer->batchnorm) {
                            datao[oy * mwo + ox] = (datao[oy * mwo + ox] - ilayer->filter.rolling_mean[n]) * fast_inverse_sqrt(ilayer->filter.rolling_variance[n] + 0.00001f);
                            datao[oy * mwo + ox]*= ilayer->filter.scale[n];
                        }
                        datao[oy * mwo + ox] = activate(datao[oy * mwo + ox] + ilayer->filter.bias[n], ilayer->activate);
                    }
                    datao += (olayer->matrix.height + olayer->matrix.pad * 2) * mwo;
                    dataf += fw * fh * ilayer->filter.channels;
                }
            }
        }
        datai += (ilayer->matrix.height + ilayer->matrix.pad * 2) * mwi;
    }
}

static void layer_groupconv_forward(LAYER *ilayer, LAYER *olayer)
{
    LAYER tilayer, tolayer; int i;
    tilayer.activate         = ilayer->activate ;
    tilayer.batchnorm        = ilayer->batchnorm;
    tilayer.filter           = ilayer->filter;
    tilayer.matrix           = ilayer->matrix;
    tilayer.stride           = ilayer->stride;
    tilayer.matrix.channels /= ilayer->groups;
    tilayer.filter.n        /= ilayer->groups;
    tolayer.matrix           = olayer->matrix;
    tolayer.matrix.channels /= ilayer->groups;
    for (i=0; i<ilayer->groups; i++) {
        layer_convolution_forward(&tilayer, &tolayer);
        tolayer.matrix.data +=(tolayer.matrix.width + tolayer.matrix.pad * 2) * (tolayer.matrix.height + tolayer.matrix.pad * 2) * tolayer.matrix.channels;
        tilayer.matrix.data +=(tilayer.matrix.width + tilayer.matrix.pad * 2) * (tilayer.matrix.height + tilayer.matrix.pad * 2) * tilayer.matrix.channels;
        tilayer.filter.data += tilayer.filter.width * tilayer.filter.height * tilayer.filter.channels * tilayer.filter.n;
        tilayer.filter.bias += tilayer.filter.n;
        tilayer.filter.scale+= tilayer.filter.n;
        tilayer.filter.rolling_mean     += tilayer.filter.n;
        tilayer.filter.rolling_variance += tilayer.filter.n;
    }
}

static void layer_avgmaxpool_forward(LAYER *ilayer, LAYER *olayer, int flag)
{
    int  n, ix, iy, ox, oy, fw, fh, fs, mwi, mwo;
    float *datai, *datao;
    fw    = ilayer->filter.width;
    fh    = ilayer->filter.height;
    fs    = ilayer->stride;
    mwi   = ilayer->matrix.width + ilayer->matrix.pad * 2;
    mwo   = olayer->matrix.width + olayer->matrix.pad * 2;
    datai = ilayer->matrix.data;
    datao = olayer->matrix.data + olayer->matrix.pad * mwo + olayer->matrix.pad;
    for (n=0; n<olayer->matrix.channels; n++) {
        for (iy=0,oy=0; iy<ilayer->matrix.width; iy+=fs,oy++) {
            for (ix=0,ox=0; ix<ilayer->matrix.width; ix+=fs,ox++) {
                datao[oy * mwo + ox] = activate(filter_avgmax(datai, mwi, ix, iy, fw, fh, flag), ilayer->activate);
            }
        }
        datai += (ilayer->matrix.height + ilayer->matrix.pad * 2) * mwi;
        datao += (olayer->matrix.height + olayer->matrix.pad * 2) * mwo;
    }
}

static void layer_upsample_forward(LAYER *ilayer, LAYER *olayer)
{
    int    mwi   = ilayer->matrix.width + ilayer->matrix.pad * 2;
    int    mwo   = olayer->matrix.width + olayer->matrix.pad * 2;
    float *datai = ilayer->matrix.data + ilayer->matrix.pad * mwi + ilayer->matrix.pad;
    float *datao = olayer->matrix.data + olayer->matrix.pad * mwo + olayer->matrix.pad;
    int    stride= ilayer->stride, i, x, y;
    for (i=0; i<ilayer->matrix.channels; i++) {
        for (y=0; y<olayer->matrix.height; y++) {
            for (x=0; x<olayer->matrix.width; x++) {
                datao[y * mwo + x] = datai[(y / stride) * mwi + (x / stride)];
            }
        }
        datai += (ilayer->matrix.height + ilayer->matrix.pad * 2) * mwi;
        datao += (olayer->matrix.height + olayer->matrix.pad * 2) * mwo;
    }
}

static void layer_dropout_forward(LAYER *ilayer, LAYER *olayer)
{
    olayer->matrix.data = ilayer->matrix.data;
    ilayer->matrix.data = NULL;
}

static void layer_shortcut_forward(NET *net, LAYER *ilayer, LAYER *olayer)
{
    LAYER  *slayer = net->layer_list + ilayer->depend_list[0] + 1;
    MATRIX *mr = &olayer->matrix, *m1 = &ilayer->matrix, *m2 = &slayer->matrix;
    int     mwr= mr->width + mr->pad * 2;
    int     mw1= m1->width + m1->pad * 2;
    int     mw2= m2->width + m2->pad * 2;
    float  *datar = mr->data + mr->pad * mwr + mr->pad;
    float  *data1 = m1->data + m1->pad * mw1 + m1->pad;
    float  *data2 = m2->data + m2->pad * mw2 + m2->pad;
    int     i, x, y;
    for (i=0; i<mr->channels; i++) {
        for (y=0; y<mr->height; y++) {
            for (x=0; x<mr->width; x++) {
                datar[y * mwr + x] = activate(data1[y * mw1 + x] + data2[y * mw2 + x], ilayer->activate);
            }
        }
        datar += (mr->height + mr->pad * 2) * mwr;
        data1 += (m1->height + m1->pad * 2) * mw1;
        data2 += (m2->height + m2->pad * 2) * mw2;
    }
}

static void layer_route_forward(NET *net, LAYER *ilayer, LAYER *olayer)
{
    int    mwo   = olayer->matrix.width + olayer->matrix.pad * 2;
    float *datao = olayer->matrix.data + olayer->matrix.pad * mwo + olayer->matrix.pad;
    int    i, j, k;
    for (i=0; i<ilayer->depend_num; i++) {
        LAYER *rlayer = net->layer_list + ilayer->depend_list[i] + 1;
        int    mwr    = rlayer->matrix.width + rlayer->matrix.pad * 2;
        float *datar  = rlayer->matrix.data + rlayer->matrix.pad * mwr + rlayer->matrix.pad;
        for (j=0; j<rlayer->matrix.channels; j++) {
            for (k=0; k<rlayer->matrix.height; k++) memcpy(datao + k * mwo, datar + k * mwr, olayer->matrix.width * sizeof(float));
            datao += mwo * (olayer->matrix.height + olayer->matrix.pad * 2);
            datar += mwr * (rlayer->matrix.height + rlayer->matrix.pad * 2);
        }
    }
}

static float get_matrix_data(MATRIX *mat, int w, int h, int c)
{
    int mw = mat->width  + mat->pad * 2;
    int mh = mat->height + mat->pad * 2;
    return mat->data[c * mw * mh + h * mw + w];
}

static void layer_yolo_forward(NET *net, LAYER *ilayer, LAYER *olayer)
{
    int i, j, k, l; float confidence;
    for (i=0; i<ilayer->matrix.height; i++) {
        for (j=0; j<ilayer->matrix.width; j++) {
            for (k=0; k<3; k++) {
                int dstart = k * (4 + 1 + ilayer->class_num), cindex = 0;
                float tx = get_matrix_data(&ilayer->matrix, j, i, dstart + 0);
                float ty = get_matrix_data(&ilayer->matrix, j, i, dstart + 1);
                float tw = get_matrix_data(&ilayer->matrix, j, i, dstart + 2);
                float th = get_matrix_data(&ilayer->matrix, j, i, dstart + 3);
                float bs = get_matrix_data(&ilayer->matrix, j, i, dstart + 4);
                float cs = get_matrix_data(&ilayer->matrix, j, i, dstart + 5);
                for (l=1; l<ilayer->class_num; l++) {
                    float val = get_matrix_data(&ilayer->matrix, j, i, dstart + 5 + l);
                    if (cs < val) { cs = val; cindex = l; }
                }
                confidence = 1.0f / ((1.0f + (float)exp(-bs) * (1.0f + (float)exp(-cs))));
                if (confidence >= ilayer->ignore_thres) {
                    float bbox_cx   = (j + activate(tx, ACTIVATE_TYPE_SIGMOID)) * net->layer_list[0].matrix.width / ilayer->matrix.width;
                    float bbox_cy   = (i + activate(ty, ACTIVATE_TYPE_SIGMOID)) * net->layer_list[0].matrix.width / ilayer->matrix.height;
                    float bbox_w    = (float)exp(tw) * ilayer->anchor_list[k][0] * ilayer->scale_x_y;
                    float bbox_h    = (float)exp(th) * ilayer->anchor_list[k][1] * ilayer->scale_x_y;
                    if (net->bbox_num < net->bbox_max && net->bbox_list) {
                        net->bbox_list[net->bbox_num].type  = cindex;
                        net->bbox_list[net->bbox_num].score = confidence;
                        net->bbox_list[net->bbox_num].x1    = bbox_cx - bbox_w * 0.5f;
                        net->bbox_list[net->bbox_num].y1    = bbox_cy - bbox_h * 0.5f;
                        net->bbox_list[net->bbox_num].x2    = bbox_cx + bbox_w * 0.5f;
                        net->bbox_list[net->bbox_num].y2    = bbox_cy + bbox_h * 0.5f;
                        net->bbox_num++;
                    }
                }
            }
        }
    }
}

static void layer_forward(NET *net, LAYER *ilayer, LAYER *olayer)
{
    switch (ilayer->type) {
    case LAYER_TYPE_CONV    : layer_groupconv_forward (ilayer, olayer);      break;
    case LAYER_TYPE_AVGPOOL : layer_avgmaxpool_forward(ilayer, olayer, 0);   break;
    case LAYER_TYPE_MAXPOOL : layer_avgmaxpool_forward(ilayer, olayer, 1);   break;
    case LAYER_TYPE_UPSAMPLE: layer_upsample_forward  (ilayer, olayer);      break;
    case LAYER_TYPE_DROPOUT : layer_dropout_forward   (ilayer, olayer);      break;
    case LAYER_TYPE_SHORTCUT: layer_shortcut_forward  (net, ilayer, olayer); break;
    case LAYER_TYPE_ROUTE   : layer_route_forward     (net, ilayer, olayer); break;
    case LAYER_TYPE_YOLO    : layer_yolo_forward      (net, ilayer, olayer); break;
    }
}

static char* load_file_to_buffer(char *file)
{
    FILE *fp = fopen(file, "rb");
    char *buf= NULL;
    int   size;
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    buf  = malloc(size + 1);
    if (buf) { fread(buf, 1, size, fp); buf[size] = '\0'; }
    fclose(fp);
    return buf;
}

static int get_total_layers(char *str)
{
    static const char *LAYER_TYPE[] = { "[conv]", "[convolutional]", "[avg]", "[avgpool]", "[max]", "[maxpool]", "[upsample]", "[dropout]", "[shortcut]", "[route]", "[yolo]", NULL };
    int n = 0, i;
    while (str && (str = strstr(str, "["))) {
        for (i=0; LAYER_TYPE[i]; i++) {
            if (strstr(str, LAYER_TYPE[i]) == str) break;
        }
        if (LAYER_TYPE[i]) n++;
        str = strstr(str, "]");
    }
    return n;
}

static char* parse_params(const char *str, const char *end, const char *key, char *val, int len)
{
    char *p = (char*)strstr(str, key); int i;

    *val = '\0';
    if (!p || (end && p >= end)) return NULL;
    p += strlen(key);
    if (*p == '\0') return NULL;

    while (*p) {
        if (*p != '=' && *p != ' ') break;
        else p++;
    }

    for (i=0; i<len; i++) {
        if (*p == '\n' || *p == '\0') break;
        val[i] = *p++;
    }
    val[i < len ? i : len - 1] = '\0';
    return val;
}

static int get_activation_type_int(char *str)
{
    static const char *STR_TAB[] = { "linear", "relu", "leaky", NULL };
    int  i;
    for (i=0; STR_TAB[i]; i++) {
        if (strstr(str, STR_TAB[i]) == str) return i;
    }
    return -1;
}

static char* get_activation_type_string(int type)
{
    static const char *STR_TAB[] = { "linear", "relu", "leaky", NULL };
    return (type >= 0 && type <= 2) ? (char*)STR_TAB[type] : "unknown";
}

static char* get_layer_type_string(int type)
{
    static const char *STR_TAB[] = { "conv", "avgpool", "maxpool", "upsample", "dropout", "shortcut", "route", "yolo" };
    return (type >= 0 && type <= 7) ? (char*)STR_TAB[type] : "unknown";
}

static void calculate_output_whc(LAYER *in, LAYER *out)
{
    in ->matrix.pad      = in ->matrix.pad ? in->filter.width / 2 : 0;
    in ->matrix.pad      = in ->matrix.pad ? in->filter.height/ 2 : 0;
    out->matrix.channels = in->filter.n;
    out->matrix.width    =(in->matrix.width - in->filter.width + in->matrix.pad * 2) / in->stride + 1;
    out->matrix.height   =(in->matrix.height- in->filter.height+ in->matrix.pad * 2) / in->stride + 1;
}

#pragma pack(1)
typedef struct {
    int32_t  ver_major, ver_minor, ver_revision;
    uint64_t net_seen;
} WEIGHTS_FILE_HEADER;
#pragma pack()

NET* net_load(char *fcfg, char *fweights)
{
    char *cfgstr = load_file_to_buffer(fcfg), *pstart, *pend, strval[256];
    NET  *net    = NULL;
    int   layers, layercur = 0, i;
    if (!cfgstr) return NULL;

    layers = get_total_layers(cfgstr);
    net    = calloc(1, sizeof(NET) + (layers + 1) * sizeof(LAYER));
    pstart = cfgstr;
    net->layer_list = (LAYER*)((char*)net + sizeof(NET));
    net->layer_num  = layers;

    while (pstart && (pstart = strstr(pstart, "["))) {
        pend = strstr(pstart + 1, "[");
        if (pend) pend = pend - 1;

        if (strstr(pstart, "[net]") == pstart) {
            parse_params(pstart, pend, "width"   , strval, sizeof(strval)); net->layer_list[0].matrix.width   = atoi(strval);
            parse_params(pstart, pend, "height"  , strval, sizeof(strval)); net->layer_list[0].matrix.height  = atoi(strval);
            parse_params(pstart, pend, "channels", strval, sizeof(strval)); net->layer_list[0].matrix.channels= atoi(strval);
        } else if (strstr(pstart, "[conv]") == pstart || strstr(pstart, "[convolutional]") == pstart) {
            parse_params(pstart, pend, "filters" , strval, sizeof(strval)); net->layer_list[layercur].filter.n     = atoi(strval);
            parse_params(pstart, pend, "size"    , strval, sizeof(strval)); net->layer_list[layercur].filter.width = net->layer_list[layercur].filter.height = atoi(strval);
            parse_params(pstart, pend, "stride"  , strval, sizeof(strval)); net->layer_list[layercur].stride       = atoi(strval);
            parse_params(pstart, pend, "pad"     , strval, sizeof(strval)); net->layer_list[layercur].matrix.pad   = atoi(strval);
            parse_params(pstart, pend, "groups"  , strval, sizeof(strval)); net->layer_list[layercur].groups       = atoi(strval);
            parse_params(pstart, pend, "batch_normalize", strval, sizeof(strval)); net->layer_list[layercur].batchnorm = atoi(strval);
            parse_params(pstart, pend, "activation"     , strval, sizeof(strval)); net->layer_list[layercur].activate  = get_activation_type_int(strval);
            if (net->layer_list[layercur].stride== 0) net->layer_list[layercur].stride= 1;
            if (net->layer_list[layercur].groups== 0) net->layer_list[layercur].groups= 1;
            net->layer_list[layercur].filter.channels = net->layer_list[layercur].matrix.channels / net->layer_list[layercur].groups;
            net->weight_size += net->layer_list[layercur].filter.width * net->layer_list[layercur].filter.height * net->layer_list[layercur].filter.channels * net->layer_list[layercur].filter.n;
            net->weight_size += net->layer_list[layercur].filter.n * (1 + !!net->layer_list[layercur].batchnorm * 3);
            net->layer_list[layercur++].type = LAYER_TYPE_CONV;
            calculate_output_whc(net->layer_list + layercur - 1, net->layer_list + layercur);
        } else if (strstr(pstart, "[avg]") == pstart || strstr(pstart, "[avgpool]") == pstart || strstr(pstart, "[max]") == pstart || strstr(pstart, "[maxpool]") == pstart) {
            parse_params(pstart, pend, "size"  , strval, sizeof(strval)); net->layer_list[layercur].filter.width = net->layer_list[layercur].filter.height = atoi(strval);
            parse_params(pstart, pend, "stride", strval, sizeof(strval)); net->layer_list[layercur].stride       = atoi(strval);
            parse_params(pstart, pend, "pad"   , strval, sizeof(strval)); net->layer_list[layercur].matrix.pad   = (strcmp(strval, "") == 0) ? 1 : atoi(strval);
            net->layer_list[layercur  ].filter.n = net->layer_list[layercur].matrix.channels;
            net->layer_list[layercur++].type = (strstr(pstart, "[avg") == pstart) ? LAYER_TYPE_AVGPOOL : LAYER_TYPE_MAXPOOL;
            calculate_output_whc(net->layer_list + layercur - 1, net->layer_list + layercur);
        } else if (strstr(pstart, "[upsample]") == pstart) {
            parse_params(pstart, pend, "stride" , strval, sizeof(strval)); net->layer_list[layercur].stride = atoi(strval);
            net->layer_list[layercur+1].matrix.channels = net->layer_list[layercur].matrix.channels;
            net->layer_list[layercur+1].matrix.width    = net->layer_list[layercur].matrix.width  * net->layer_list[layercur].stride;
            net->layer_list[layercur+1].matrix.height   = net->layer_list[layercur].matrix.height * net->layer_list[layercur].stride;
            net->layer_list[layercur++].type = LAYER_TYPE_UPSAMPLE;
        } else if (strstr(pstart, "[dropout]") == pstart || strstr(pstart, "[shortcut]") == pstart) {
            if (strstr(pstart, "[dropout]") == pstart) {
                net->layer_list[layercur++].type = LAYER_TYPE_DROPOUT;
            } else {
                parse_params(pstart, pend, "from"      , strval, sizeof(strval)); net->layer_list[layercur].depend_list[0] = atoi(strval) + layercur;
                parse_params(pstart, pend, "activation", strval, sizeof(strval)); net->layer_list[layercur].activate = get_activation_type_int(strval);
                net->layer_list[layercur  ].depend_num = 1;
                net->layer_list[layercur++].type = LAYER_TYPE_SHORTCUT;
            }
            net->layer_list[layercur].matrix.channels = net->layer_list[layercur - 1].matrix.channels;
            net->layer_list[layercur].matrix.width    = net->layer_list[layercur - 1].matrix.width;
            net->layer_list[layercur].matrix.height   = net->layer_list[layercur - 1].matrix.height;
        } else if (strstr(pstart, "[route]") == pstart) {
            char *str; int dep = 0;
            parse_params(pstart, pend, "layers", strval, sizeof(strval));
            for (i=0; i<4 && (str = strtok(i ? NULL : strval, ",")); i++) {
                dep = atoi(str);
                dep = dep > 0 ? dep : layercur + dep;
                net->layer_list[layercur + 0].depend_list[i]   = dep;
                net->layer_list[layercur + 1].matrix.channels += net->layer_list[dep + 1].matrix.channels;
                net->layer_list[layercur + 1].matrix.width     = net->layer_list[dep + 1].matrix.width;
                net->layer_list[layercur + 1].matrix.height    = net->layer_list[dep + 1].matrix.height;
            }
            net->layer_list[layercur].depend_num = i;
            net->layer_list[layercur++].type = LAYER_TYPE_ROUTE;
        } else if (strstr(pstart, "[yolo]") == pstart) {
            char *str; int masks[9]; int anchors[9][2];
            parse_params(pstart, pend, "classes"      , strval, sizeof(strval)); net->layer_list[layercur].class_num   = (int  )atoi(strval);
            parse_params(pstart, pend, "scale_x_y"    , strval, sizeof(strval)); net->layer_list[layercur].scale_x_y   = (float)atof(strval);
            parse_params(pstart, pend, "ignore_thresh", strval, sizeof(strval)); net->layer_list[layercur].ignore_thres= (float)atof(strval);
            parse_params(pstart, pend, "mask", strval, sizeof(strval));
            for (i=0; i<9 && (str = strtok(i ? NULL : strval, ",")); i++) masks[i] = atoi(str);
            net->layer_list[layercur].anchor_num = i;
            parse_params(pstart, pend, "anchors", strval, sizeof(strval));
            for (i=0; i<9 && (str = strtok(i ? NULL : strval, ",")); i++) {
                anchors[i][0] = atoi(str);
                str = strtok(NULL, ",");
                anchors[i][1] = atoi(str);
            }
            for (i=0; i<net->layer_list[layercur].anchor_num; i++) {
                net->layer_list[layercur].anchor_list[i][0] = anchors[masks[i]][0];
                net->layer_list[layercur].anchor_list[i][1] = anchors[masks[i]][1];
            }
            net->layer_list[layercur++].type = LAYER_TYPE_YOLO;
        }
        pstart = pend;
    }
    free(cfgstr);

    net->weight_buf = malloc(net->weight_size * sizeof(float));
    if (net->weight_buf) {
        float *pfloat = net->weight_buf; FILE *fp;
        for (i=0; i<layers; i++) {
            if (net->layer_list[i].type == LAYER_TYPE_CONV) {
                FILTER *filter = &net->layer_list[i].filter;
                filter->bias = pfloat; pfloat += filter->n;
                if (net->layer_list[i].batchnorm) {
                    filter->scale            = pfloat; pfloat += filter->n;
                    filter->rolling_mean     = pfloat; pfloat += filter->n;
                    filter->rolling_variance = pfloat; pfloat += filter->n;
                }
                filter->data = pfloat; pfloat += filter->width * filter->height * filter->channels * filter->n;
            }
        }
        if ((fp = fopen(fweights, "rb"))) { fseek (fp, sizeof(WEIGHTS_FILE_HEADER), SEEK_SET); fread (net->weight_buf, 1, net->weight_size * sizeof(float), fp); fclose(fp); }
    }
    net->bbox_max = (net->layer_list[0].matrix.width / 32) * (net->layer_list[0].matrix.height / 32) * 3;
    net->bbox_list= malloc(net->bbox_max * sizeof(BBOX));
    return net;
}

void net_free(NET *net)
{
    int  i;
    if (!net) return;
    for (i=0; i<net->layer_num+1; i++) {
        if (net->layer_list[i].matrix.data) {
            printf("net_free, free matrix memory for layer %d\n", i);
            free(net->layer_list[i].matrix.data);
        }
    }
    free(net->weight_buf);
    free(net->bbox_list );
    free(net);
}

void net_input(NET *net, unsigned char *bgr, int w, int h, float *mean, float *norm)
{
    MATRIX *mat = NULL;
    int    sw, sh, i, j;
    float  *p1, *p2, *p3;
    if (!net) return;

    mat = &(net->layer_list[0].matrix);
    if (mat->channels != 3) { printf("invalid input matrix channels: %d !\n", mat->channels); return; }
    if (mat->data == NULL) {
        mat->data = calloc(1, (mat->width + mat->pad * 2) * (mat->height + mat->pad * 2) * mat->channels * sizeof(float));
        if (!mat->data) { printf("failed to allocate memory for net input !\n"); return; }
    }

    if (w * mat->height > h * mat->width) {
        sw = mat->width ; sh = mat->width * h / w;
        net->s1 = w; net->s2 = sw;
    } else {
        sh = mat->height; sw = mat->height* w / h;
        net->s1 = h; net->s2 = sh;
    }
    p1 = mat->data + (mat->width + mat->pad * 2) * mat->pad + mat->pad;
    p2 = p1 + (mat->width + mat->pad * 2) * (mat->height + mat->pad * 2);
    p3 = p2 + (mat->width + mat->pad * 2) * (mat->height + mat->pad * 2);
    for (i=0; i<sh; i++) {
        for (j=0; j<sw; j++) {
            int x = j * net->s1 / net->s2;
            int y = i * net->s1 / net->s2;
            int b = bgr[y * w * 3 + x * 3 + 0];
            int g = bgr[y * w * 3 + x * 3 + 1];
            int r = bgr[y * w * 3 + x * 3 + 2];
            p1[i * (mat->width + mat->pad * 2) + j] = (r - mean[0]) * norm[0];
            p2[i * (mat->width + mat->pad * 2) + j] = (g - mean[1]) * norm[1];
            p3[i * (mat->width + mat->pad * 2) + j] = (b - mean[2]) * norm[2];
        }
    }
}

static int bbox_cmp(const void *p1, const void *p2)
{
    if      (((BBOX*)p1)->score < ((BBOX*)p2)->score) return  1;
    else if (((BBOX*)p1)->score > ((BBOX*)p2)->score) return -1;
    else return 0;
}

static int nms(BBOX *bboxlist, int n, float threshold, int min, int s1, int s2)
{
    int i, j, c;
    if (!bboxlist || !n) return 0;
    qsort(bboxlist, n, sizeof(BBOX), bbox_cmp);
    for (i=0; i<n && i!=-1; ) {
        for (c=i,j=i+1,i=-1; j<n; j++) {
            if (bboxlist[j].score == 0) continue;
            if (bboxlist[c].type == bboxlist[j].type) {
                float xc1, yc1, xc2, yc2, sc, s1, s2, ss, iou;
                xc1 = bboxlist[c].x1 > bboxlist[j].x1 ? bboxlist[c].x1 : bboxlist[j].x1;
                yc1 = bboxlist[c].y1 > bboxlist[j].y1 ? bboxlist[c].y1 : bboxlist[j].y1;
                xc2 = bboxlist[c].x2 < bboxlist[j].x2 ? bboxlist[c].x2 : bboxlist[j].x2;
                yc2 = bboxlist[c].y2 < bboxlist[j].y2 ? bboxlist[c].y2 : bboxlist[j].y2;
                sc  = (xc1 < xc2 && yc1 < yc2) ? (xc2 - xc1) * (yc2 - yc1) : 0;
                s1  = (bboxlist[c].x2 - bboxlist[c].x1) * (bboxlist[c].y2 - bboxlist[c].y1);
                s2  = (bboxlist[j].x2 - bboxlist[j].x1) * (bboxlist[j].y2 - bboxlist[j].y1);
                ss  = s1 + s2 - sc;
                if (min) iou = sc / (s1 < s2 ? s1 : s2);
                else     iou = sc / ss;
                if (iou > threshold) bboxlist[j].score = 0;
                else if (i == -1) i = j;
            } else if (i == -1) i = j;
        }
    }
    for (i=0,j=0; i<n; i++) {
        if (bboxlist[i].score) {
            bboxlist[j  ].x1 = bboxlist[i].x1 * s1 / s2;
            bboxlist[j  ].y1 = bboxlist[i].y1 * s1 / s2;
            bboxlist[j  ].x2 = bboxlist[i].x2 * s1 / s2;
            bboxlist[j++].y2 = bboxlist[i].y2 * s1 / s2;
        }
    }
    return j;
}

void net_forward(NET *net)
{
    LAYER *ilayer, *olayer; int i, j;
    if (!net) return;
    net->bbox_num = 0;
    for (i=0; i<net->layer_num; i++) {
        if (net->layer_list[i].depend_num > 0) {
            for (j=0; j<net->layer_list[i].depend_num; j++) {
                net->layer_list[net->layer_list[i].depend_list[j] + 1].refcnt++;
            }
        }
    }
    for (i=0; i<net->layer_num; i++) {
        ilayer = net->layer_list + i + 0;
        olayer = net->layer_list + i + 1;
        if (!olayer->matrix.data && ilayer->type != LAYER_TYPE_DROPOUT && ilayer->type != LAYER_TYPE_YOLO) {
            olayer->matrix.data = malloc((olayer->matrix.width + olayer->matrix.pad * 2) * (olayer->matrix.height + olayer->matrix.pad * 2) * olayer->matrix.channels * sizeof(float));
            if (!olayer->matrix.data) { printf("failed to allocate memory for output layer !\n"); return; }
            else matrix_fill_pad(&olayer->matrix, olayer->type == LAYER_TYPE_MAXPOOL ? -FLT_MAX : 0);
        }

        layer_forward(net, ilayer, olayer);

        if (i > 0 && ilayer->refcnt == 0) { free(ilayer->matrix.data); ilayer->matrix.data = NULL; }
        for (j=0; j<ilayer->depend_num; j++) {
            if (--net->layer_list[ilayer->depend_list[j] + 1].refcnt == 0) {
                free(net->layer_list[ilayer->depend_list[j] + 1].matrix.data);
                net->layer_list[ilayer->depend_list[j] + 1].matrix.data = NULL;
            }
        }
    }
    net->bbox_num = nms(net->bbox_list, net->bbox_num, 0.5f, 1, net->s1, net->s2);
}

void net_dump(NET *net)
{
    int i, j;
    if (!net) return;
    printf("layer   type  filters fltsize  pad/strd input          output       bn/act  ref\n");
    for (i=0; i<net->layer_num; i++) {
        if (net->layer_list[i].type == LAYER_TYPE_YOLO) {
            printf("%3d %8s\n", i, get_layer_type_string(net->layer_list[i].type));
        } else if (net->layer_list[i].type == LAYER_TYPE_DROPOUT) {
            printf("%3d %8s %-38s -> %3dx%3dx%3d\n", i, get_layer_type_string(net->layer_list[i].type), "",
                net->layer_list[i+1].matrix.width, net->layer_list[i+1].matrix.height, net->layer_list[i+1].matrix.channels);
        } else if (net->layer_list[i].type == LAYER_TYPE_SHORTCUT || net->layer_list[i].type == LAYER_TYPE_ROUTE) {
            char strdeps[256] = "layers:", strnum[16];
            for (j=0; j<net->layer_list[i].depend_num; j++) {
                snprintf(strnum, sizeof(strnum), " %d", net->layer_list[i].depend_list[j]);
                strncat(strdeps, strnum, sizeof(strdeps) - 1);
            }
            printf("%3d %8s %-38s -> %3dx%3dx%3d           %d\n", i, get_layer_type_string(net->layer_list[i].type), strdeps,
                net->layer_list[i+1].matrix.width, net->layer_list[i+1].matrix.height, net->layer_list[i+1].matrix.channels, net->layer_list[i].refcnt);
        } else {
            printf("%3d %8s %3d/%3d %2dx%2dx%3d   %d/%2d   %3dx%3dx%3d -> %3dx%3dx%3d  %d/%-6s %d\n", i,
                get_layer_type_string(net->layer_list[i].type), net->layer_list[i].filter.n, net->layer_list[i].groups,
                net->layer_list[i].filter.width, net->layer_list[i].filter.height, net->layer_list[i].filter.channels,
                net->layer_list[i].matrix.pad, net->layer_list[i].stride,
                net->layer_list[i+0].matrix.width, net->layer_list[i+0].matrix.height, net->layer_list[i+0].matrix.channels,
                net->layer_list[i+1].matrix.width, net->layer_list[i+1].matrix.height, net->layer_list[i+1].matrix.channels,
                net->layer_list[i].batchnorm, get_activation_type_string(net->layer_list[i].activate), net->layer_list[i].refcnt);
        }
    }
    printf("total weights: %d, total bytes: %d\n", net->weight_size, sizeof(WEIGHTS_FILE_HEADER) + net->weight_size * sizeof(float));
}

#if 1
#include "bmpfile.h"

#ifdef WIN32
#include <windows.h>
#define get_tick_count GetTickCount
#else
#include <time.h>
static uint32_t get_tick_count()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}
#endif

int main(int argc, char *argv[])
{
    static const float MEAN[3] = { 0.0f, 0.0f, 0.0f };
    static const float NORM[3] = { 1/255.f, 1/255.f, 1/255.f };
    char *file_bmp    = "test.bmp";
    char *file_cfg    = "yolo-fastest-1.1.cfg";
    char *file_weights= "yolo-fastest-1.1.weights";
    NET  *mynet       = NULL;
    BMP   mybmp       = {0};
    int   tick, i;

    if (argc > 1) file_bmp    = argv[1];
    if (argc > 2) file_cfg    = argv[2];
    if (argc > 3) file_weights= argv[3];
    printf("file_bmp    : %s\n", file_bmp    );
    printf("file_cfg    : %s\n", file_cfg    );
    printf("file_weights: %s\n", file_weights);

    if (0 != bmp_load(&mybmp, file_bmp)) { printf("failed to load bmp file: %s !\n", file_bmp); return -1; }
    mynet = net_load(file_cfg, file_weights);
    net_dump(mynet);
    tick = (int)get_tick_count();
    for (i=0; i<100; i++) {
        net_input  (mynet, mybmp.pdata, mybmp.width, mybmp.height, (float*)MEAN, (float*)NORM);
        net_forward(mynet);
    }
    printf("%dms\n", (int)get_tick_count() - (int)tick);
    for (i=0; i<mynet->bbox_num; i++) bmp_rectangle(&mybmp, (int)mynet->bbox_list[i].x1, (int)mynet->bbox_list[i].y1, (int)mynet->bbox_list[i].x2, (int)mynet->bbox_list[i].y2, 0, 255, 0);
    net_free(mynet);
    bmp_save(&mybmp, "out.bmp");
    bmp_free(&mybmp);
    return 0;
}
#endif
