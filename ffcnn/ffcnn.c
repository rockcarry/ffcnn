#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <conio.h>
#include "ffcnn.h"

#pragma warning(disable:4996)

float activate(float x, int type)
{
    switch (type) {
    case ACTIVATE_TYPE_RELU  : return x > 0 ? x : 0;
    case ACTIVATE_TYPE_LEAKY : return x > 0 ? x : 0.1f * x;
    default: return x;
    }
}

void layer_forward(LAYER *ilayer, LAYER *olayer)
{
}

static char* load_file_to_string(char *file)
{
    FILE *fp = fopen(file, "rb");
    char *buf= NULL;
    int   size;
    if (fp) {
        fseek(fp, 0, SEEK_END);
        size = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        buf  = malloc(size + 1);
        if (buf) {
            fread(buf, 1, size, fp);
            buf[size] = '\0';
        }
        fclose(fp);
    }
    return buf;
}

static int get_total_layers(char *str)
{
    static const char *STRTAB_LAYER_TYPE[] = {
        "[conv]", "[convolutional]", "[avg]", "[avgpool]", "[max]", "[maxpool]", "[upsample]", "[dropout]", "[shortcut]", "[route]", "[yolo]", NULL,
    };
    int n = 0, i;
    while (str && (str = strstr(str, "["))) {
        for (i=0; STRTAB_LAYER_TYPE[i]; i++) {
            if (strstr(str, STRTAB_LAYER_TYPE[i]) == str) break;
        }
        if (STRTAB_LAYER_TYPE[i]) n++;
        str = strstr(str, "]");
    }
    return n;
}

static char* parse_params(const char *str, const char *end, const char *key, char *val, int len)
{
    char *p = (char*)strstr(str, key);
    int   i;

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
    int padw, padh;
    padw = in->pad ? in->filter.width / 2 : 0;
    padh = in->pad ? in->filter.height/ 2 : 0;
    out->matrix.channels =  in->filter.n;
    out->matrix.width    = (in->matrix.width - in->filter.width + padw * 2) / in->stride + 1;
    out->matrix.height   = (in->matrix.height- in->filter.height+ padh * 2) / in->stride + 1;
}

NET* net_load(char *file1, char *file2)
{
    char *cfgstr = load_file_to_string(file1), *pstart, *pend, strval[256];
    NET  *net    = NULL;
    int   layers, layercur = 0;
    if (cfgstr) {
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
                parse_params(pstart, pend, "filters", strval, sizeof(strval)); net->layer_list[layercur].filter.n = atoi(strval);
                parse_params(pstart, pend, "size"   , strval, sizeof(strval)); net->layer_list[layercur].filter.width = net->layer_list[layercur].filter.height = atoi(strval);
                parse_params(pstart, pend, "stride" , strval, sizeof(strval)); net->layer_list[layercur].stride = atoi(strval);
                parse_params(pstart, pend, "pad"    , strval, sizeof(strval)); net->layer_list[layercur].pad    = atoi(strval);
                parse_params(pstart, pend, "groups" , strval, sizeof(strval)); net->layer_list[layercur].groups = atoi(strval);
                parse_params(pstart, pend, "batch_normalize", strval, sizeof(strval)); net->layer_list[layercur].batchnorm = atoi(strval);
                parse_params(pstart, pend, "activation"     , strval, sizeof(strval)); net->layer_list[layercur].activate  = get_activation_type_int(strval);
                if (net->layer_list[layercur].stride== 0) net->layer_list[layercur].stride= 1;
                if (net->layer_list[layercur].groups== 0) net->layer_list[layercur].groups= 1;
                net->layer_list[layercur  ].filter.channels = net->layer_list[layercur].matrix.channels;
                net->layer_list[layercur++].type = LAYER_TYPE_CONV;
                calculate_output_whc(net->layer_list + layercur - 1, net->layer_list + layercur);
            } else if (strstr(pstart, "[avg]") == pstart || strstr(pstart, "[avgpool]") == pstart || strstr(pstart, "[max]") == pstart || strstr(pstart, "[maxpool]") == pstart) {
                parse_params(pstart, pend, "size"  , strval, sizeof(strval)); net->layer_list[layercur].filter.width = net->layer_list[layercur].filter.height = atoi(strval);
                parse_params(pstart, pend, "stride", strval, sizeof(strval)); net->layer_list[layercur].stride = atoi(strval);
                parse_params(pstart, pend, "pad"   , strval, sizeof(strval)); net->layer_list[layercur].pad = (strcmp(strval, "") == 0) ? 1 : atoi(strval);
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
                char *str; int n = 0, dep = 0;
                parse_params(pstart, pend, "layers", strval, sizeof(strval));
                while (n < 4 && (str = strtok(n ? NULL : strval, ","))) {
                    dep = atoi(str);
                    dep = dep > 0 ? dep : layercur + dep;
                    net->layer_list[layercur + 0].depend_list[n++] = dep;
                    net->layer_list[layercur + 1].matrix.channels += net->layer_list[dep + 1].matrix.channels;
                    net->layer_list[layercur + 1].matrix.width     = net->layer_list[dep + 1].matrix.width;
                    net->layer_list[layercur + 1].matrix.height    = net->layer_list[dep + 1].matrix.height;
                }
                net->layer_list[layercur].depend_num = n;
                net->layer_list[layercur++].type = LAYER_TYPE_ROUTE;
            } else if (strstr(pstart, "[yolo]") == pstart) {
                net->layer_list[layercur++].type = LAYER_TYPE_YOLO;
            }
            pstart = pend;
        }
        free(cfgstr);
    }
    return net;
}

void net_dump(NET *net)
{
    int i, j;
    printf("layer   type  filters fltsize  pad/strd input          output       bn/act\n");
    for (i=0; i<net->layer_num; i++) {
        if (net->layer_list[i].type == LAYER_TYPE_YOLO) {
            printf("%3d %8s\n", i, get_layer_type_string(net->layer_list[i].type));
        } else if (net->layer_list[i].type == LAYER_TYPE_DROPOUT) {
            printf("%3d %8s %-38s -> %3dx%3dx%3d\n", i, get_layer_type_string(net->layer_list[i].type), "",
                net->layer_list[i+1].matrix.width, net->layer_list[i+1].matrix.height, net->layer_list[i+1].matrix.channels);
        } else if (net->layer_list[i].type == LAYER_TYPE_SHORTCUT || net->layer_list[i].type == LAYER_TYPE_ROUTE) {
            char strdeps[256] = "layers:", strnum[16];
            for (j=0; j<net->layer_list[i].depend_num; j++) {
                _snprintf(strnum, sizeof(strnum), " %d", net->layer_list[i].depend_list[j]);
                strncat(strdeps, strnum, sizeof(strdeps) - 1);
            }
            printf("%3d %8s %-38s -> %3dx%3dx%3d\n", i, get_layer_type_string(net->layer_list[i].type), strdeps,
                net->layer_list[i+1].matrix.width, net->layer_list[i+1].matrix.height, net->layer_list[i+1].matrix.channels);
        } else {
            printf("%3d %8s %3d/%3d %2dx%2dx%3d   %d/%2d   %3dx%3dx%3d -> %3dx%3dx%3d  %d/%s\n", i, get_layer_type_string(net->layer_list[i].type), net->layer_list[i].filter.n, net->layer_list[i].groups,
                net->layer_list[i].filter.width, net->layer_list[i].filter.height, net->layer_list[i].filter.channels,
                net->layer_list[i].pad, net->layer_list[i].stride,
                net->layer_list[i+0].matrix.width, net->layer_list[i+0].matrix.height, net->layer_list[i+0].matrix.channels,
                net->layer_list[i+1].matrix.width, net->layer_list[i+1].matrix.height, net->layer_list[i+1].matrix.channels,
                net->layer_list[i].batchnorm, get_activation_type_string(net->layer_list[i].activate));
        }
    }
}

int main(int argc, char *argv[])
{
    char *file_cfg    = "yolo-fastest-1.1.cfg";
    char *file_weight = "yolo-fastest-1.1.weight";
    NET  *net  = NULL;

    if (argc > 1) file_cfg    = argv[1];
    if (argc > 2) file_weight = argv[2];

    net = net_load(file_cfg, file_weight);
    net_dump(net);
    getch();
    return 0;
}
