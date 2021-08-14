#!/bin/sh

gcc -Wall -Ofast -msse3 bmpfile.c ffcnn.c -o ffcnn-clan -lm
gcc -Wall -Ofast -msse3 -DFFCNN_SSE3 bmpfile.c ffcnn.c sse3.c -o ffcnn-sse3 -lm
