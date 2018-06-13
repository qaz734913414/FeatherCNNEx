LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := feather
LOCAL_SRC_FILES := ../../build-android/$(TARGET_ARCH_ABI)/install/feather/lib/libfeather.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)

include /home/leejohnnie/tools/OpenCV-android-sdk/sdk/native/jni/OpenCV.mk

LOCAL_MODULE := test
LOCAL_SRC_FILES := ../test.cpp

LOCAL_C_INCLUDES :=../build-android/feather/include/feather ../src /home/leejohnnie/tools/OpenCV-android-sdk/sdk/native/jni/include
LOCAL_STATIC_LIBRARIES := feather

LOCAL_CFLAGS := -O2 -fvisibility=hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math
LOCAL_CPPFLAGS := -O2 -fvisibility=hidden -fvisibility-inlines-hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math
LOCAL_CPPFLAGS += -Wall -frtti

ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
LOCAL_ARM_NEON = true
endif

LOCAL_CFLAGS += -fopenmp
LOCAL_CPPFLAGS += -fopenmp -fexceptions
LOCAL_LDFLAGS += -fopenmp

LOCAL_LDLIBS += -Wl,--gc-sections -L$(SYSROOT)/usr/lib -pthread

include $(BUILD_EXECUTABLE)
