#!/bin/bash
flatc -c flatbuffer_protocols/feather_simple.fbs && mv feather_simple_generated.h ../src/
protoc --cpp_out=. ./caffe.proto
g++ -DX86_PC -g feather_convert_caffe.cc caffe.pb.cc ../src/common.cpp ../src/crypto/aes.cpp -I/usr/include `pkg-config --cflags --libs protobuf` -o feather_convert_caffe -std=c++11 -I../src -I../src/crypto
#./feather_convert_caffe 12net.prototxt 12net.caffemodel 12net
#./feather_convert_caffe insano_256_deploy.proto insano_9856_256.model insano 8 0.01
#./feather_convert_caffe insano_256_deploy.proto insano_9856_256.model insano 14 0.01
#./feather_convert_caffe insano_256_deploy.proto insano_9856_256.model insano
#cp *.feathermodel /media/psf/Home/nfs/
#echo "build ok"
