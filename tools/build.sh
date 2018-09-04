#!/bin/bash
flatc -c flatbuffer_protocols/feather_simple.fbs && mv feather_simple_generated.h ../src/
protoc --cpp_out=. ./caffe.proto
g++ -DX86_PC -g feather_convert_caffe.cc caffe.pb.cc ../src/common.cpp ../src/crypto/aes.cpp -I/usr/include `pkg-config --cflags --libs protobuf` -o feather_convert_caffe -std=c++11 -I../src -I../src/crypto
#./feather_convert_caffe 12net.prototxt det1.caffemodel.convert C-PNet 8 1.0 1 feather.key det1.int8scaletable
