#ifndef __BMPFILE_H__
#define __BMPFILE_H__

#ifdef __cplusplus
extern "C" {
#endif

/* BMP 对象的类型定义 */
typedef struct {
    int   width;   /* 宽度 */
    int   height;  /* 高度 */
    int   stride;  /* 行字节数 */
    int   cdepth;  /* 像素位数 */
    void *pdata;   /* 指向数据 */
} BMP;

int  bmp_load(BMP *pb, char *file);
int  bmp_save(BMP *pb, char *file);
void bmp_free(BMP *pb);
void bmp_setpixel (BMP *pb, int x, int y, int  r, int  g, int  b);
void bmp_getpixel (BMP *pb, int x, int y, int *r, int *g, int *b);
void bmp_rectangle(BMP *pb, int x1, int y1, int x2, int y2, int r, int g, int b);

#ifdef __cplusplus
}
#endif

#endif
