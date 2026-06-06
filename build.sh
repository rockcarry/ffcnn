#!/bin/sh

# Parse version argument, default to v6
CONV_TYPE="${1:-v6}"

# Check for PGO mode
PGO_MODE=""
case "$CONV_TYPE" in
    *-pgo-gen)
        PGO_MODE="gen"
        CONV_TYPE="${CONV_TYPE%-pgo-gen}"
        ;;
    *-pgo-use)
        PGO_MODE="use"
        CONV_TYPE="${CONV_TYPE%-pgo-use}"
        ;;
esac

echo "build with conv type: $CONV_TYPE (PGO: ${PGO_MODE:-none})"

# Base optimization flags
CFLAGS="-Wall -Ofast -funroll-loops -fomit-frame-pointer -msse2 -march=native"

# Link-time optimization (disable with PGO gen to avoid conflicts)
if [ "$PGO_MODE" != "gen" ]; then
    CFLAGS="$CFLAGS -flto"
fi

# OpenMP for v4
if [ "$CONV_TYPE" = "v4" ]; then
    CFLAGS="$CFLAGS -fopenmp"
fi

# PGO flags
if [ "$PGO_MODE" = "gen" ]; then
    echo "=== PGO: generating profile data ==="
    CFLAGS="$CFLAGS -fprofile-generate"
elif [ "$PGO_MODE" = "use" ]; then
    echo "=== PGO: using profile data ==="
    CFLAGS="$CFLAGS -fprofile-use -fprofile-correction"
fi

# Clean old gcda if generating new profile
if [ "$PGO_MODE" = "gen" ]; then
    rm -f *.gcda
fi

gcc $CFLAGS -D_TEST_ bmpfile.c ffcnn.c conv-$CONV_TYPE.c -lm -o ffcnn

if [ $? -eq 0 ]; then
    echo "=== build success: ffcnn ==="
else
    echo "=== build failed ==="
    exit 1
fi
