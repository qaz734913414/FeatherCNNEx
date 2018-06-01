#!/bin/bash
flatc -c flatbuffer_protocols/feather_simple.fbs && mv feather_simple_generated.h ../src/
protoc --cpp_out=. ./caffe.proto
g++ -g feather_convert_caffe.cc caffe.pb.cc -I/usr/include `pkg-config --cflags --libs protobuf` -o feather_convert_caffe -std=c++11 -I../src
./feather_convert_caffe mbface.prototxt mbface.caffemodel mbface 13 0.01
./feather_convert_caffe mbface.prototxt mbface.caffemodel mbface 14 0.01
./feather_convert_caffe mbface.prototxt mbface.caffemodel mbface_float
 cp mbface*.feathermodel /media/psf/Home/nfs/
echo "build ok"
