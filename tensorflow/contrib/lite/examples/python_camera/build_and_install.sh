#!/bin/bash

# Download dependencies and build
sh tensorflow/contrib/lite/download_dependencies.sh
make -f tensorflow/contrib/lite/Makefile -j 32 clean
make -f tensorflow/contrib/lite/Makefile -j 32 CROSS=$1

# Make a install directory
mkdir /tmp/inst
cp tensorflow/contrib/lite/gen/bin/tflite_c.so /tmp/inst
cp tensorflow/contrib/lite/examples/python_camera/*.py  /tmp/inst
cp tensorflow/contrib/lite/examples/python_camera/*.h  /tmp/inst

