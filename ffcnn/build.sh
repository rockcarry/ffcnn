#!/bin/sh

gcc -Wall -Ofast -msse2 bmpfile.c ffcnn.c -o ffcnn -lm

