# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)
set( OpenCV_DIR /home/lee/software/OpenCV-android-sdk/sdk/native/jni )

find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
if(NOT OpenCV_FOUND)
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
endif()

include(FindOpenMP)
if(OPENMP_FOUND)
    message(STATUS ${OpenMP_C_FLAGS})
    message(STATUS ${OpenMP_CXX_FLAGS})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
message(STATUS "OpenMP flags ${CMAKE_CXX_FLAGS}")
# ###################### 项目原生模块 ###########################

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    message(STATUS "compile use clang")
    if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
        include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/feather64/include)
        link_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/feather64/lib)
    else("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a")
        include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/feather32/include)
        link_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/feather32/lib)
    endif ()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    message(STATUS "compile use gcc")
    if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
        include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/feather64gcc/include)
        link_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/feather64gcc/lib)
    else("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a")
        include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/feather32gcc/include)
        link_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/feather32gcc/lib)
    endif ()
endif ()

if( CMAKE_SYSTEM_PROCESSOR MATCHES armv7-a )
    add_definitions(-mfpu=neon-fp16)
endif()

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
        native-lib

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        src/main/cpp/native-lib.cpp
        src/main/cpp/featherCNNEx.cpp)

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( native-lib
        ${OpenCV_LIBS}
        feather
        log
        jnigraphics)