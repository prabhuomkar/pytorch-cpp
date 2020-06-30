cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(image-io VERSION 1.0.0 LANGUAGES CXX)

if(NOT Torch_FOUND)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")
    find_package(Torch REQUIRED PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../libtorch")
endif()

add_library(image-io SHARED)
target_sources(image-io PUBLIC src/image_io.cpp
                               include/image_io.h
)
target_include_directories(image-io PUBLIC include)

target_link_libraries(image-io ${TORCH_LIBRARIES} stb-image stb-image-write stb-image-resize)
                          
set_target_properties(image-io PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)