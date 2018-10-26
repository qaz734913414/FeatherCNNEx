cmake_minimum_required(VERSION 3.11)

set(OpenCV_DIR "/home/lee/software/OpenCV-android-sdk-3.4.3/sdk/native/jni")

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

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
message(STATUS "compile name ${CMAKE_CXX_COMPILER_ID}")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
message(STATUS "compile use clang")
if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../build-android/arm64-v8a/install/feather/lib)
else("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a")
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../build-android/armeabi-v7a/install/feather/lib)
endif ()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
message(STATUS "compile use gcc")
if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../build-android/arm64-v8a-gcc/install/feather/lib)
else("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a")
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../build-android/armeabi-v7a-gcc/install/feather/lib)
endif ()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
# using Intel C++
message(STATUS "compile use intel")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
# using Visual Studio C++
message(STATUS "compile use msvc")
endif()

if( CMAKE_SYSTEM_PROCESSOR MATCHES armv7-a )
add_definitions(-mfpu=neon-fp16)
endif()

add_definitions(-std=c++11 -DINFERENCE_FEATHERCNNEX)

if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")

add_executable(feather_test64 test.cpp)
target_link_libraries(feather_test64 feather ${OpenCV_LIBS})

#add_executable(mtcnn-test-64 mtcnn/inference.cpp  mtcnn/mtcnndetect.cpp)
#target_link_libraries(mtcnn-test-64 feather ${OpenCV_LIBS})

else("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a")

add_executable(feather_test32 test.cpp)
target_link_libraries(feather_test32 feather ${OpenCV_LIBS})

#add_executable(mtcnn-test mtcnn/inference.cpp  mtcnn/mtcnndetect.cpp)
#target_link_libraries(mtcnn-test feather ${OpenCV_LIBS})

endif ()
