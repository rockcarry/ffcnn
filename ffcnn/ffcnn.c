#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include "ffcnn.h"

#ifdef _MSC_VER
#pragma warning(disable:4996)
#define snprintf _snprintf
#endif
typedef int                int32_t;
typedef unsigned           uint32_t;
typedef unsigned long long uint64_t;

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

#define ENABLE_NET_PROFILE  0

enum {
    ACTIVATE_TYPE_LINEAR ,
    ACTIVATE_TYPE_RELU   ,
    ACTIVATE_TYPE_LEAKY  ,
    ACTIVATE_TYPE_SIGMOID,
};
#define ALIGN(x, n) (((x) + ((n) - 1)) & ~((n) - 1))

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
    if (buf) { size = (int)fread(buf, 1, size, fp); buf[size] = '\0'; }
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
        p++;
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
    static const char *STR_TAB[] = { "linear", "relu", "leaky", NULL }; int  i;
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

#pragma pack(1)
typedef struct {
    int32_t  ver_major, ver_minor, ver_revision;
    uint64_t net_seen;
} WEIGHTS_FILE_HEADER;
#pragma pack()

NET* net_load(char *fcfg, char *fweights)
{
    char *cfgstr = load_file_to_buffer(fcfg), *pstart, *pend, strval[256];
    NET  *net = NULL; LAYER *ilayer = NULL, *olayer = NULL;
    int   layers = 0, layercur = 0, got_layer, ftsize, ret, i, j;
    if (!cfgstr) return NULL;

    (void)ret;
    layers = get_total_layers(cfgstr);
    net    = calloc(1, sizeof(NET) + (layers + 1) * sizeof(LAYER));
    pstart = cfgstr;
    net->layer_list = (LAYER*)((char*)net + sizeof(NET));
    net->layer_num  = layers;

    while (pstart && (pstart = strstr(pstart, "["))) {
        if ((pend = strstr(pstart + 1, "["))) pend = pend - 1;
        ilayer = net->layer_list + layercur; olayer = net->layer_list + layercur + 1;
        ilayer->stride = ilayer->groups = got_layer = 1;
        if (strstr(pstart, "[net]") == pstart) {
            parse_params(pstart, pend, "width"   , strval, sizeof(strval)); net->layer_list[0].w = atoi(strval);
            parse_params(pstart, pend, "height"  , strval, sizeof(strval)); net->layer_list[0].h = atoi(strval);
            parse_params(pstart, pend, "channels", strval, sizeof(strval)); net->layer_list[0].c = atoi(strval);
            got_layer = 0;
        } else if (strstr(pstart, "[conv]") == pstart || strstr(pstart, "[convolutional]") == pstart) {
            parse_params(pstart, pend, "filters" , strval, sizeof(strval)); ilayer->fn    = atoi(strval);
            parse_params(pstart, pend, "size"    , strval, sizeof(strval)); ilayer->fs    = atoi(strval);
            parse_params(pstart, pend, "stride"  , strval, sizeof(strval)); ilayer->stride= atoi(strval) ? atoi(strval) : 1;
            parse_params(pstart, pend, "groups"  , strval, sizeof(strval)); ilayer->groups= atoi(strval) ? atoi(strval) : 1;
            parse_params(pstart, pend, "pad"     , strval, sizeof(strval)); ilayer->pad   = atoi(strval);
            parse_params(pstart, pend, "batch_normalize", strval, sizeof(strval)); ilayer->batchnorm = !!atoi(strval);
            parse_params(pstart, pend, "activation"     , strval, sizeof(strval)); ilayer->activation= get_activation_type_int(strval);
            ilayer->pad  = ilayer->pad ? ilayer->fs / 2 : 0;
            ilayer->type = LAYER_TYPE_CONV;
            olayer->c    =  ilayer->fn;
            olayer->w    = (ilayer->w - ilayer->fs + ilayer->pad * 2) / ilayer->stride + 1;
            olayer->h    = (ilayer->h - ilayer->fs + ilayer->pad * 2) / ilayer->stride + 1;
            net->weight_size += ilayer->fn * (ALIGN(ilayer->fs * ilayer->fs * ilayer->c, 4) + 4);
        } else if (strstr(pstart, "[avg]") == pstart || strstr(pstart, "[avgpool]") == pstart || strstr(pstart, "[max]") == pstart || strstr(pstart, "[maxpool]") == pstart) {
            parse_params(pstart, pend, "size"  , strval, sizeof(strval)); ilayer->fs    = atoi(strval);
            parse_params(pstart, pend, "stride", strval, sizeof(strval)); ilayer->stride= atoi(strval) ? atoi(strval) : 1;
            ilayer->type = (strstr(pstart, "[avg") == pstart) ? LAYER_TYPE_AVGPOOL : LAYER_TYPE_MAXPOOL;
            olayer->c    = ilayer->c;
            olayer->w    = ilayer->w / ilayer->stride;
            olayer->h    = ilayer->h / ilayer->stride;
        } else if (strstr(pstart, "[upsample]") == pstart) {
            parse_params(pstart, pend, "stride", strval, sizeof(strval)); ilayer->stride= atoi(strval) ? atoi(strval) : 1;
            ilayer->type = LAYER_TYPE_UPSAMPLE;
            olayer->c    = ilayer->c;
            olayer->w    = ilayer->w * ilayer->stride;
            olayer->h    = ilayer->h * ilayer->stride;
        } else if (strstr(pstart, "[dropout]") == pstart || strstr(pstart, "[shortcut]") == pstart) {
            if (strstr(pstart, "[dropout]") == pstart) {
                ilayer->type = LAYER_TYPE_DROPOUT;
            } else {
                parse_params(pstart, pend, "from"      , strval, sizeof(strval)); ilayer->depend_list[0] = atoi(strval) + layercur;
                parse_params(pstart, pend, "activation", strval, sizeof(strval)); ilayer->activation = get_activation_type_int(strval);
                ilayer->depend_num = 1;
                ilayer->type = LAYER_TYPE_SHORTCUT;
            }
            olayer->c = ilayer->c; olayer->w = ilayer->w; olayer->h = ilayer->h;
        } else if (strstr(pstart, "[route]") == pstart) {
            char *str; int dep = 0;
            parse_params(pstart, pend, "layers", strval, sizeof(strval));
            for (i=0; i<4 && (str = strtok(i ? NULL : strval, ",")); i++) {
                dep = atoi(str);
                dep = dep > 0 ? dep : layercur + dep;
                ilayer->depend_list[i] = dep;
                olayer->c += net->layer_list[dep + 1].c;
                olayer->w  = net->layer_list[dep + 1].w;
                olayer->h  = net->layer_list[dep + 1].h;
            }
            ilayer->depend_num = i;
            ilayer->type = LAYER_TYPE_ROUTE;
        } else if (strstr(pstart, "[yolo]") == pstart) {
            char *str; int masks[9]; int anchors[9][2];
            parse_params(pstart, pend, "classes"      , strval, sizeof(strval)); ilayer->class_num   = atoi(strval);
            parse_params(pstart, pend, "scale_x_y"    , strval, sizeof(strval)); ilayer->scale_x_y   = (strcmp(strval, "") == 0) ? 1.0f : (float)atof(strval);
            parse_params(pstart, pend, "ignore_thresh", strval, sizeof(strval)); ilayer->ignore_thres= (float)atof(strval);
            parse_params(pstart, pend, "mask", strval, sizeof(strval));
            for (i=0; i<9 && (str = strtok(i ? NULL : strval, ",")); i++) masks[i] = atoi(str);
            parse_params(pstart, pend, "anchors", strval, sizeof(strval));
            for (i=0; i<9 && (str = strtok(i ? NULL : strval, ",")); i++) {
                anchors[i][0] = atoi(str);
                str = strtok(NULL, ",");
                anchors[i][1] = atoi(str);
            }
            for (i=0; i<3; i++) {
                ilayer->anchor_list[i][0] = anchors[masks[i]][0];
                ilayer->anchor_list[i][1] = anchors[masks[i]][1];
            }
            ilayer->type = LAYER_TYPE_YOLO;
        } else got_layer = 0;
		if (got_layer) layercur++;
        pstart = pend;
    }
    free(cfgstr);

    net->weight_buf = calloc(1, net->weight_size * sizeof(float));
    if (net->weight_buf) {
        float *pfloat = net->weight_buf; FILE *fp = fopen(fweights, "rb");
        if (fp) fseek(fp, sizeof(WEIGHTS_FILE_HEADER), SEEK_SET);
        for (i=0; i<layers; i++) {
            if (net->layer_list[i].type == LAYER_TYPE_CONV) {
                ilayer = net->layer_list + i;
                ftsize = ALIGN(ilayer->fs * ilayer->fs * (ilayer->c / ilayer->groups), 4) + 4;
                ilayer->filter = pfloat; pfloat += ilayer->fn * ftsize;
                if (fp) {
                    for (j=0; j<ilayer->fn; j++) ret = (int)fread(ilayer->filter + ftsize * j + ftsize - 4, 1, sizeof(float), fp); // bias
                    if (ilayer->batchnorm) {
                        for (j=0; j<ilayer->fn; j++) ret = (int)fread(ilayer->filter + ftsize * j + ftsize - 2, 1, sizeof(float), fp); // scale/norm
                        for (j=0; j<ilayer->fn; j++) ret = (int)fread(ilayer->filter + ftsize * j + ftsize - 3, 1, sizeof(float), fp); // rolling_mean
                        for (j=0; j<ilayer->fn; j++) ret = (int)fread(ilayer->filter + ftsize * j + ftsize - 1, 1, sizeof(float), fp); // rolling_variance
                        for (j=0; j<ilayer->fn; j++) ilayer->filter[ftsize * j + ftsize - 2] = (float)(ilayer->filter[ftsize * j + ftsize - 2] / sqrt(ilayer->filter[ftsize * j + ftsize - 1] + 0.00001f));
                    }
                    for (j=0; j<ilayer->fn; j++) ret = (int)fread(ilayer->filter + ftsize * j, 1, ilayer->fs * ilayer->fs * (ilayer->c / ilayer->groups) * sizeof(float), fp);
                }
            }
        }
        if (fp) fclose(fp);
    }

    ilayer        = net->layer_list;
    ilayer->data  = calloc(1, ilayer->w * ilayer->h * ilayer->c * sizeof(float));
    net->bbox_max = ilayer->w * ilayer->h * ilayer->c * sizeof(float) / sizeof(BBOX);
    net->bbox_list= (BBOX*)ilayer->data;
    if (!net->weight_buf || !ilayer->data) { printf("failed to allocate buffers for net_load !\n"); net_free(net); net = NULL; }
    return net;
}

void net_free(NET *net)
{
    int  i;
    if (!net) return;
    for (i=0; i<net->layer_num+1; i++) free(net->layer_list[i].data);
    free(net->cnntempbuf);
    free(net->weight_buf);
    free(net);
}

void net_input(NET *net, unsigned char *bgr, int w, int h, float *mean, float *norm)
{
    LAYER *ilayer = net->layer_list;
    int    linebytes, sw, sh, i, j;
    float  *p1, *p2, *p3;
    if (!net) return;

    memset(net->bbox_list, 0, sizeof(BBOX) * net->bbox_num); net->bbox_num = 0;
    if (w * ilayer->h > h * ilayer->w) {
        sw = ilayer->w; sh = sw * h / w;
        net->s1 = w; net->s2 = sw;
    } else {
        sh = ilayer->h; sw = sh * w / h;
        net->s1 = h; net->s2 = sh;
    }
    linebytes = ALIGN(w * 3, 4); // align to 4 bytes
    p1 = ilayer->data + ilayer->w * ilayer->h * 0;
    p2 = ilayer->data + ilayer->w * ilayer->h * 1;
    p3 = ilayer->data + ilayer->w * ilayer->h * 2;
    for (i=0; i<sh; i++) {
        for (j=0; j<sw; j++) {
            int x = j * net->s1 / net->s2, y = i * net->s1 / net->s2, k = y * linebytes + x * 3;
            *p1++ = (bgr[k + 2] - mean[0]) * norm[0]; // r
            *p2++ = (bgr[k + 1] - mean[1]) * norm[1]; // g
            *p3++ = (bgr[k + 0] - mean[2]) * norm[2]; // b
        }
        p1 += ilayer->w - sw;
        p2 += ilayer->w - sw;
        p3 += ilayer->w - sw;
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
            bboxlist[j  ].score= bboxlist[i].score;
            bboxlist[j  ].type = bboxlist[i].type;
            bboxlist[j  ].x1   = bboxlist[i].x1 * s1 / s2;
            bboxlist[j  ].y1   = bboxlist[i].y1 * s1 / s2;
            bboxlist[j  ].x2   = bboxlist[i].x2 * s1 / s2;
            bboxlist[j++].y2   = bboxlist[i].y2 * s1 / s2;
        }
    }
    memset(bboxlist + j, 0, sizeof(BBOX) * (n - j));
    return j;
}

static float activate(float x, int type)
{
    switch (type) {
    case ACTIVATE_TYPE_RELU   : return x > 0 ? x : 0;
    case ACTIVATE_TYPE_LEAKY  : return x > 0 ? x : 0.1f * x;
    case ACTIVATE_TYPE_SIGMOID: return 1.0f / (1.0f + (float)exp(-x));
    default: return x;
    }
}

static void im2row(float *data, int w, int h, int c, int pad, int fs, int x, int y, float *buf)
{
    float *src = data + (y - pad) * w + (x - pad);
    int    i, j, k;
    if (pad == 0 || (x - pad >= 0 && y - pad >= 0 && x + pad < w && y + pad < h)) {
        i = fs * fs * c;
        x = y = 0;
        do {
            *buf++ = *src++;
            if (++x == fs) {
                if (1)         { x = 0; src += w - fs; }
                if (++y == fs) { y = 0; src += w * (h - fs); }
            }
        } while (--i);
    } else {
        for (i=0; i<c; i++) {
            for (j=0; j<fs; j++) {
                for (k=0; k<fs; k++,src++) {
                    // x - pad + k >= 0 && x - pad + k < w, base on fast range check: a >= 0 && a < b <==> (unsigned)a < (unsigned)b
                    *buf++ = ((unsigned)(j - pad + y) < (unsigned)h && (unsigned)(k - pad + x) < (unsigned)w) ? *src : 0;
                }
                src += w - fs;
            }
            src += w * (h - fs);
        }
    }
}

static void layer_groupconv_forward(NET *net, LAYER *ilayer, LAYER *olayer)
{
    int    walign, ftsize, x, y, g, c, ic, oc, fn, i;
    float *datai, *datao, *dataf, sum;

    datai  = ilayer->data;
    datao  = olayer->data;
    dataf  = ilayer->filter;
    ic     = ilayer->c / ilayer->groups;
    oc     = olayer->c / ilayer->groups;
    fn     = ilayer->fn/ ilayer->groups;
    walign = ALIGN(ilayer->fs * ilayer->fs * ic, 4);
    ftsize = walign + 4;
    if (net->cnnbufsize < walign) {
        net->cnnbufsize = walign;
        free(net->cnntempbuf); net->cnntempbuf = malloc(net->cnnbufsize * sizeof(float));
        if (net->cnntempbuf == NULL) { printf("failed to allocate memory for cnntempbuf !"); return; }
    }

    for (g=0; g<ilayer->groups; g++) {
        for (y=0; y<olayer->h; y++) {
            for (x=0; x<olayer->w; x++) {
                im2row(datai, ilayer->w, ilayer->h, ic, ilayer->pad, ilayer->fs, x * ilayer->stride, y * ilayer->stride, net->cnntempbuf);
                for (c=0; c<oc; c++) {
                    for (sum=0,i=0; i<walign; i++) sum += dataf[c * ftsize + i] * net->cnntempbuf[i];
                    if (ilayer->batchnorm) sum = (sum - dataf[c * ftsize + walign + 1]) * dataf[c * ftsize + walign + 2];
                    datao[c * olayer->w * olayer->h + y * olayer->w + x] = activate(sum + dataf[c * ftsize + walign + 0], ilayer->activation);
                }
            }
        }
        datai += ilayer->w * ilayer->h * ic;
        datao += olayer->w * olayer->h * oc;
        dataf += ftsize * fn;
    }
}

static float filter_avgpool(float *mat, int mw, int mh, int x, int y, int fsize)
{
    int xmin, ymin, xmax, ymax; float val = 0;
    xmin = x - (fsize - 1) / 2;
    ymin = y - (fsize - 1) / 2;
    xmax = xmin + fsize;
    ymax = ymin + fsize;
    if (xmin < 0 ) xmin = 0;
    if (ymin < 0 ) ymin = 0;
    if (xmax > mw) xmax = mw;
    if (ymax > mh) ymax = mh;
    for (y=ymin; y<ymax; y++) {
        for (x=xmin; x<xmax; x++) val += mat[y * mw + x];
    }
    return val / (fsize * fsize);
}

static float filter_maxpool(float *mat, int mw, int mh, int x, int y, int fsize)
{
    int xmin, ymin, xmax, ymax; float val;
    xmin = x - (fsize - 1) / 2;
    ymin = y - (fsize - 1) / 2;
    xmax = xmin + fsize;
    ymax = ymin + fsize;
    if (xmin < 0 ) xmin = 0;
    if (ymin < 0 ) ymin = 0;
    if (xmax > mw) xmax = mw;
    if (ymax > mh) ymax = mh;
    val = mat[ymin * mw + xmin];
    for (y=ymin; y<ymax; y++) {
        for (x=xmin; x<xmax; x++) {
            if (val < mat[y * mw + x]) val = mat[y * mw + x];
        }
    }
    return val;
}

static void layer_avgmaxpool_forward(LAYER *ilayer, LAYER *olayer, int flag)
{
    int ix, iy, ox, oy, i;
    float *datai = ilayer->data;
    float *datao = olayer->data;
    for (i=0; i<ilayer->c; i++) {
        for (iy=0,oy=0; iy<ilayer->h; iy+=ilayer->stride,oy++) {
            for (ix=0,ox=0; ix<ilayer->w; ix+=ilayer->stride,ox++) {
                *datao++ = (flag ? filter_maxpool : filter_avgpool)(datai, ilayer->w, ilayer->h, ix, iy, ilayer->fs);
            }
        }
        datai += ilayer->w * ilayer->h;
    }
}

static void layer_upsample_forward(LAYER *ilayer, LAYER *olayer)
{
    float *datai = ilayer->data;
    float *datao = olayer->data;
    int    i, x, y;
    for (i=0; i<ilayer->c; i++) {
        for (y=0; y<olayer->h;) {
            for (x=0; x<olayer->w;) {
                *datao++ = *datai;
                if (++x % ilayer->stride == 0) datai++;
            }
            datai -= olayer->w / ilayer->stride;
            if (++y % ilayer->stride == 0) datai += ilayer->w;
        }
    }
}

static void layer_dropout_forward(LAYER *ilayer, LAYER *olayer)
{
    olayer->data = ilayer->data;
    ilayer->data = NULL;
}

static void layer_shortcut_forward(NET *net, LAYER *ilayer, LAYER *olayer)
{
    LAYER *slayer = net->layer_list + ilayer->depend_list[0] + 1;
    int    n = ALIGN(olayer->w * olayer->h * olayer->c, 4), i;
    for (i=0; i<n; i++) olayer->data[i] = activate(ilayer->data[i] + slayer->data[i], ilayer->activation);
}

static void layer_route_forward(NET *net, LAYER *ilayer, LAYER *olayer)
{
    float *datao = olayer->data; int i;
    for (i=0; i<ilayer->depend_num; i++) {
        LAYER *rlayer = net->layer_list + ilayer->depend_list[i] + 1;
        memcpy(datao, rlayer->data, rlayer->w * rlayer->h * rlayer->c * sizeof(float));
        datao += rlayer->w * rlayer->h * rlayer->c;
    }
}

static float get_layer_data(LAYER *layer, int w, int h, int c) { return layer->data[c * layer->w * layer->h + h * layer->w + w]; }

static void layer_yolo_forward(NET *net, LAYER *ilayer)
{
    int i, j, k, l; float confidence;
    for (i=0; i<ilayer->h; i++) {
        for (j=0; j<ilayer->w; j++) {
            for (k=0; k<3; k++) {
                int dstart = k * (4 + 1 + ilayer->class_num), cindex = 0;
                float bs = get_layer_data(ilayer, j, i, dstart + 4);
                float cs = get_layer_data(ilayer, j, i, dstart + 5);
                for (l=1; l<ilayer->class_num; l++) {
                    float val = get_layer_data(ilayer, j, i, dstart + 5 + l);
                    if (cs < val) { cs = val; cindex = l; }
                }
                confidence = 1.0f / ((1.0f + (float)exp(-bs) * (1.0f + (float)exp(-cs))));
                if (confidence >= ilayer->ignore_thres) {
                    float tx = get_layer_data(ilayer, j, i, dstart + 0);
                    float ty = get_layer_data(ilayer, j, i, dstart + 1);
                    float tw = get_layer_data(ilayer, j, i, dstart + 2);
                    float th = get_layer_data(ilayer, j, i, dstart + 3);
                    float bbox_cx = (j + activate(tx, ACTIVATE_TYPE_SIGMOID)) * net->layer_list[0].w / ilayer->w ;
                    float bbox_cy = (i + activate(ty, ACTIVATE_TYPE_SIGMOID)) * net->layer_list[0].h / ilayer->h;
                    float bbox_w  = (float)exp(tw) * ilayer->anchor_list[k][0] * ilayer->scale_x_y;
                    float bbox_h  = (float)exp(th) * ilayer->anchor_list[k][1] * ilayer->scale_x_y;
                    if (net->bbox_num < net->bbox_max) {
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

void net_forward(NET *net)
{
    LAYER *ilayer = net->layer_list, *olayer = net->layer_list + 1; uint32_t tick; int i, j;
    if (!net) return;
    (void)tick;
    for (i=0; i<net->layer_num; i++) {
        if (net->layer_list[i].depend_num > 0) {
            for (j=0; j<net->layer_list[i].depend_num; j++) {
                net->layer_list[net->layer_list[i].depend_list[j] + 1].refcnt++;
            }
        }
    }
    for (i=0; i<net->layer_num; i++,ilayer++,olayer++) {
        if (!olayer->data && ilayer->type != LAYER_TYPE_DROPOUT && ilayer->type != LAYER_TYPE_YOLO) {
            olayer->data = malloc(ALIGN(olayer->w * olayer->h * olayer->c, 4) * sizeof(float));
            if (!olayer->data) { printf("failed to allocate memory for output layer !\n"); return; }
        }

#if ENABLE_NET_PROFILE
        tick = get_tick_count();
#endif
        switch (ilayer->type) {
        case LAYER_TYPE_CONV    : layer_groupconv_forward (net, ilayer, olayer); break;
        case LAYER_TYPE_AVGPOOL : layer_avgmaxpool_forward(ilayer, olayer, 0);   break;
        case LAYER_TYPE_MAXPOOL : layer_avgmaxpool_forward(ilayer, olayer, 1);   break;
        case LAYER_TYPE_UPSAMPLE: layer_upsample_forward  (ilayer, olayer);      break;
        case LAYER_TYPE_DROPOUT : layer_dropout_forward   (ilayer, olayer);      break;
        case LAYER_TYPE_SHORTCUT: layer_shortcut_forward  (net, ilayer, olayer); break;
        case LAYER_TYPE_ROUTE   : layer_route_forward     (net, ilayer, olayer); break;
        case LAYER_TYPE_YOLO    : layer_yolo_forward      (net, ilayer);         break;
        }
#if ENABLE_NET_PROFILE
        tick = (int32_t)get_tick_count() - (int32_t)tick;
        net->timeused[ilayer->type] += tick;
#endif
        if (i > 0 && ilayer->refcnt == 0) { free(ilayer->data); ilayer->data = NULL; }
        for (j=0; j<ilayer->depend_num; j++) {
            if (--net->layer_list[ilayer->depend_list[j] + 1].refcnt == 0) {
                free(net->layer_list[ilayer->depend_list[j] + 1].data);
                net->layer_list[ilayer->depend_list[j] + 1].data = NULL;
            }
        }
    }
    net->bbox_num = nms(net->bbox_list, net->bbox_num, 0.5f, 1, net->s1, net->s2);
}

void net_dump(NET *net)
{
    LAYER *ilayer, *olayer; int i, j;
    if (!net) return;
    printf("layer   type  filters fltsize  pad/strd input          output       bn/act\n");
    for (i=0; i<net->layer_num; i++) {
        ilayer = net->layer_list + i + 0;
        olayer = net->layer_list + i + 1;
        if (ilayer->type == LAYER_TYPE_YOLO) {
            printf("%3d %8s class_num: %d ignore_thres: %3.2f [%d, %d] [%d, %d] [%d, %d]\n", i, get_layer_type_string(ilayer->type), ilayer->class_num, ilayer->ignore_thres,
                ilayer->anchor_list[0][0], ilayer->anchor_list[0][1], ilayer->anchor_list[1][0], ilayer->anchor_list[1][1], ilayer->anchor_list[2][0], ilayer->anchor_list[2][1]);
        } else if (ilayer->type == LAYER_TYPE_DROPOUT) {
            printf("%3d %8s %-38s -> %3dx%3dx%3d\n", i, get_layer_type_string(ilayer->type), "", olayer->w, olayer->h, olayer->c);
        } else if (ilayer->type == LAYER_TYPE_SHORTCUT || ilayer->type == LAYER_TYPE_ROUTE) {
            char strdeps[256] = "layers:", strnum[16];
            for (j=0; j<ilayer->depend_num; j++) {
                snprintf(strnum, sizeof(strnum), " %d", ilayer->depend_list[j]);
                strncat(strdeps, strnum, sizeof(strdeps) - 1);
            }
            printf("%3d %8s %-38s -> %3dx%3dx%3d\n", i, get_layer_type_string(ilayer->type), strdeps, olayer->w, olayer->h, olayer->c);
        } else {
            printf("%3d %8s %3d/%3d %2dx%2dx%3d   %d/%2d   %3dx%3dx%3d -> %3dx%3dx%3d  %d/%-6s\n", i, get_layer_type_string(ilayer->type),
                ilayer->fn, ilayer->groups, ilayer->fs, ilayer->fs, ilayer->c / ilayer->groups, ilayer->pad, ilayer->stride, ilayer->w, ilayer->h, ilayer->c,
                olayer->w, olayer->h, olayer->c, ilayer->batchnorm, get_activation_type_string(ilayer->activation));
        }
    }
}

void net_profile(NET *net) { int i; for (i=0; i<LAYER_TYPE_TOTOAL; i++) printf("%8s: %5d ms\n", get_layer_type_string(i), net->timeused[i]); }

#if 1
#include "bmpfile.h"
int main(int argc, char *argv[])
{
    static const float MEAN[3] = { 0.0f, 0.0f, 0.0f };
    static const float NORM[3] = { 1/255.f, 1/255.f, 1/255.f };
    char *file_bmp    = "test.bmp";
    char *file_cfg    = "yolo-fastest-1.1.cfg";
    char *file_weights= "yolo-fastest-1.1.weights";
    NET  *mynet       = NULL;
    BMP   mybmp       = {0};
    int   tick, n = 10, i;

    if (argc > 1) n           = atoi(argv[1]);
    if (argc > 2) file_bmp    = argv[2];
    if (argc > 3) file_cfg    = argv[3];
    if (argc > 4) file_weights= argv[4];
    printf("file_bmp    : %s\n", file_bmp    );
    printf("file_cfg    : %s\n", file_cfg    );
    printf("file_weights: %s\n", file_weights);

    if (0 != bmp_load(&mybmp, file_bmp)) { printf("failed to load bmp file: %s !\n", file_bmp); return -1; }
    mynet = net_load(file_cfg, file_weights);
    net_dump(mynet);
    tick = (int)get_tick_count();
    for (i=0; i<n; i++) {
        net_input  (mynet, mybmp.pdata, mybmp.width, mybmp.height, (float*)MEAN, (float*)NORM);
        net_forward(mynet);
    }
    printf("%d times inference: %d ms\n", n, (int)get_tick_count() - (int)tick);
    net_profile(mynet);
    for (i=0; i<mynet->bbox_num; i++) {
        printf("score: %.2f, category: %2d, rect: (%3d %3d %3d %3d)\n", mynet->bbox_list[i].score, mynet->bbox_list[i].type,
            (int)mynet->bbox_list[i].x1, (int)mynet->bbox_list[i].y1, (int)mynet->bbox_list[i].x2, (int)mynet->bbox_list[i].y2);
        bmp_rectangle(&mybmp, (int)mynet->bbox_list[i].x1, (int)mynet->bbox_list[i].y1, (int)mynet->bbox_list[i].x2, (int)mynet->bbox_list[i].y2, 0, 255, 0);
    }
    net_free(mynet);
    bmp_save(&mybmp, "out.bmp");
    bmp_free(&mybmp);
    return 0;
}
#endif
