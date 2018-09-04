#!/bin/bash

mkdir -p build-android
pushd build-android
mkdir -p arm64-v8a-gcc
pushd arm64-v8a-gcc
cmake -DCMAKE_TOOLCHAIN_FILE=$NDK_ROOT/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-22 -DANDROID_ARM_NEON=ON -DANDROID_TOOLCHAIN=gcc ../..
make -j4
make install
popd
popd
