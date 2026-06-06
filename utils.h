#ifndef __UTILS_H__
#define __UTILS_H__

#define MAX(a, b)   ((a) > (b) ? (a) : (b))
#define MIN(a, b)   ((a) < (b) ? (a) : (b))
#define ALIGN(x, n) (((x) + ((n) - 1)) & ~((n) - 1))

enum {
    ACTIVATE_TYPE_LINEAR ,
    ACTIVATE_TYPE_RELU   ,
    ACTIVATE_TYPE_LEAKY  ,
    ACTIVATE_TYPE_SIGMOID,
};

static inline float activate(float x, int type)
{
    switch (type) {
    case ACTIVATE_TYPE_RELU   : return x > 0 ? x : 0;
    case ACTIVATE_TYPE_LEAKY  : return x > 0 ? x : 0.1f * x;
    case ACTIVATE_TYPE_SIGMOID: return 1.0f / (1.0f + (float)exp(-x));
    default: return x;
    }
}

#endif
