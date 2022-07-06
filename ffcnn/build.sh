#!/bin/sh

CONV_TYPE=v1

case "$1" in
v0|v1)
    CONV_TYPE=$1
    ;;
esac

echo "build with conv type: $CONV_TYPE"

gcc -Wall -flto -ffunction-sections -fdata-sections -Ofast -msse2 -D_TEST_ bmpfile.c ffcnn.c conv-$CONV_TYPE.c -lm -o ffcnn

