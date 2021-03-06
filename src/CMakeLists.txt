file(GLOB LIB_HEADERS *.h)
file(GLOB LIB_SRC *.cpp)
file(GLOB LAYER_HEADERS layers/*.h)
file(GLOB LAYER_SRC layers/*.cpp)
file(GLOB FLATBUFFERS_HEADERS flatbuffers/*.h)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -Wno-format")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -std=c++11 -Wall")

include_directories("${PROJECT_SOURCE_DIR}/src")

add_subdirectory(./arm)

add_library(feather STATIC ${LIB_SRC} ${LIB_HEADERS} ${LAYER_SRC} ${LAYER_HEADERS} $<TARGET_OBJECTS:arm_backend_obj> $<TARGET_OBJECTS:arm_backend_sgemm_obj>)

set(FEATHER_INSTALL_DIR "${PROJECT_BINARY_DIR}/install/feather")

message(Library headers: ${LIB_HEADERS})
list(REMOVE_ITEM LIB_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/feather_simple_generated.h)
message(Library headers: ${LIB_HEADERS})
install(TARGETS feather DESTINATION "${FEATHER_INSTALL_DIR}/lib")
install(FILES ${LIB_HEADERS} DESTINATION "${FEATHER_INSTALL_DIR}/include")
