+----------------------------+
 ffcnn 卷积神经网络前向推理库
+----------------------------+

ffcnn 是一个 c 语言编写的卷积神经网络前向推理库
只用了 600 多行代码就实现了完整的 yolov3、yolo-fastest 网络的前向推理
不依赖于任何第三方库，在标准 c 环境下就可以编译通过，在 VC、msys2+gcc、ubuntu+gcc
等多个平台上都可以正确的编译运行

这个代码相对于 darknet、ncnn 来说，性能还没做任何优化，但代码更加简洁易懂，可以作为
大家学习卷积神经网络的一个参考


darknet 与 yolov3 的一些总结
----------------------------
yolov3 的网络结构里面，只有卷积层、dropout 层、shortcut 层、route 层、maxpool 层、
upsample 层和 yolo 层这几种类型。因此要实现起来还是比较容易的

卷积层：
1. 要搞明白卷积的含义和计算方法
2. 卷积运算的 pad、stride 的含义
3. 每个卷积核还有一个 bias 参数，计算完每个点后需要加上这个 bias
   没有归一化的情况（batch_normalize），其计算方法：
   x += bias;
   x  = activate(x, type);
4. 要搞明白什么是分组卷积
5. 卷积运算每个输出的点，都要经过激活函数
6. 卷积层如果有归一化操作（batch_normalize），其计算方法：
   x  = (x + rolling_mean) / sqrt(rolling_variance + 0.00001f)
   x *= scale;
   x += bias;
   x  = activate(x, type);
   其中 rolling_mean、rolling_variance、scale、bias 在 darknet 的 weights 文件中可以读取到

dropout 层：
前向推理时，这一层可以当做不存在，输入数据不做任何处理，直接传给下一层即可

shortcut 层：
把指定层的数据和当前层的数据相加，然后结果输出到下一层

route 层：
把指定的层（最多可以有 4 个）做拼接，宽高不变，channel 个数增加，然后结果输出到下一层

maxpool 层：
max 池化层，将 filter 覆盖的数据取最大值作为结果

upsample 层：
上采样层，可以理解为把图像放大，stride 指定了放大倍数，一般用最近邻法就可以了

yolo 层：
这一层主要是根据输入的 feature map 计算出 bbox
以 yolo-fastest 为例，总共有两个 yolo 层，其输入分别是 10x10x255 和 20x20x255
其中 255 表示有 255 个通道，其每个数据的含义如下：
255 = 3 * (4 + 1 + 80)
3 表示这个 grid 里面有 3 个 bbox 结果数据
每个 bbox 结果数据里面，4 个 x, y, w, h 坐标数据，1 个 object score 评分，然后是 80 个分类的评分
每个 bbox 里面在 80 个分类中找出评分最高的，作为这个 bbox 的分类，评分如果小于阈值（ignore_thresh）则丢弃
将符合要求的全部 bbox 放入一个列表保存，然后再做一个 nms 操作，就得到最终结果了

每个 bbox 的评分和 (x, y, w, h) 计算方法：
设 tx, ty, tw, th, bs 分别对应channel 0, 1, 2, 3, 4 的值（后面还有 80 个分类的评分）

评分的计算方法：score = sigmoid(bs); （80 个分类评分计算方法是一样的）
坐标计算方法：
float bbox_cx = (j + sigmod(tx)) * grid_width;  （grid_width 就是网络输入层即 0 层的宽度除以格子数目，即每个格子的像素宽度）
float bbox_cy = (i + sigmod(ty)) * grid_height; （方法与 bbox_cx 一致）
float bbox_w  = (float)exp(tw) * anchor_box_w;  （如果有缩放系数还要乘以这个系数）
float bbox_h  = (float)exp(th) * anchor_box_h;  （方法与 bbox_w  一致）

bbox_cx、bbox_cy 是中心点坐标，bbox_w、bbox_h 是宽高，转换一下得到：
x1 = bbox_cx - bbox_w * 0.5f;
y1 = bbox_cy - bbox_h * 0.5f;
x2 = bbox_cx + bbox_w * 0.5f;
y2 = bbox_cy + bbox_h * 0.5f;


darknet 的 weights 文件
-----------------------

文件最前面有一个文件头：
#pragma pack(1)
typedef struct {
    int32_t  ver_major, ver_minor, ver_revision;
    uint64_t net_seen;
} WEIGHTS_FILE_HEADER;
#pragma pack()

然后就是全部的权重数据，yolov3、yolo-fastest 的模块基本上就只有卷积层的权重，其它层是没有权重数据的。
图像和卷积核（filter）的数据都是 NCHW 格式，filter 的数据存放顺序为：

n 个 bias
if (batchnorm) {
    n 个 scale
    n 个 rolling_mean
    n 个 rolling_variance
}
n * c * h * w 个权重数据


ffcnn 的特点
------------
1. 极为简洁易懂的 c 语言代码实现
2. 核心算法仅仅 600 行
3. 不依赖于任何第三方库
4. 可以很方便的移植到各种平台
5. 推理时会自动释放不需要的 layer 减小内存占用
6. 现阶段是 make it work first 后面有时间再优化性能
7. 直接使用 darknet 的 .cfg 和 .weights 文件（不需要再转换）


ffcnn vs ncnn 性能评测
----------------------
ffcnn 代码：https://github.com/rockcarry/ffcnn
ncnn + yolo-fastest 代码：https://github.com/rockcarry/ffyolodet

两个代码都是使用的 yolo-fastest-1.1 模型，测试图片都是 test.bmp
在我自己的 win7 x64 PC + msys2 gcc -O3 测试结果：
ffcnn: 100 次计算耗时：26224 ms  每帧 262ms  内存占用：4.5MB 左右
ncnn : 100 次计算耗时： 8424 ms  每帧 84ms   内存占用：45MB  左右

ncnn 还是比 ffcnn 快很多，大概是 3.1 倍。但是内存占用 ffcnn 少了很多


rockcarry@163.com
20:22 2021/8/7









