ENABLE_LANGUAGE(ASM)

file(GLOB ARM_SRC ./*.cpp)
file(GLOB ARM_HEADERS ./*.h)
file(GLOB ARM_ASM_SRC ./*.S)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -O3 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -Wno-format -Wno-unused-parameter")

add_library(arm_backend_obj OBJECT ${ARM_SRC} ${ARM_ASM_SRC} ${ARM_HEADERS})

set(ARM_INSTALL_DIR "${PROJECT_BINARY_DIR}/install/feather_backend_arm/")
