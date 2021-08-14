#!/bin/sh

gcc -Wall -Ofast -msse3 bmpfile.c ffcnn.c -o ffcnn-clan -lm

gcc -Wall -Ofast -msse3 -c sse3.c
gcc -Wall -Ofast -DFFCNN_SSE3 bmpfile.c ffcnn.c sse3.o -o ffcnn-sse3 -lm
