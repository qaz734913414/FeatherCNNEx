#!/bin/bash
mkdir -p build-android
pushd build-android
mkdir -p arm64-v8a
pushd arm64-v8a
cmake -DCMAKE_TOOLCHAIN_FILE=$NDK_ROOT/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DARCH_ARM64=1 -DANDROID_PLATFORM=android-22 ../..
make clean && make && cp feather_test64 /media/psf/Home/nfs
popd
popd
