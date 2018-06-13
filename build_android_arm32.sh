#!/bin/bash

mkdir -p build-android
pushd build-android
mkdir -p armeabi-v7a
pushd armeabi-v7a
cmake -DCMAKE_TOOLCHAIN_FILE=$NDK_ROOT/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a" -DANDROID_ARM_NEON=ON -DANDROID_PLATFORM=android-22 ../..
make -j4
make install
popd
popd
