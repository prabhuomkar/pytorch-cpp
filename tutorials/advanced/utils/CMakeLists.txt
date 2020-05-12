cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(image-utils VERSION 1.0.0 LANGUAGES CXX)

if(NOT Torch_FOUND)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake")
    find_package(Torch REQUIRED PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../../libtorch")
endif()

add_library(load-image SHARED)
target_sources(load-image PRIVATE include/external/stb_image.h
                                  include/external/stb_image_resize.h
                          PUBLIC src/load_image.cpp
                                 include/load_image.h)
                          
target_include_directories(load-image PUBLIC include 
                                             include/external)

target_link_libraries(load-image ${TORCH_LIBRARIES})

set_target_properties(load-image PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)
                                
add_library(save-image SHARED)
target_sources(save-image PRIVATE include/external/stb_image_write.h
                          PUBLIC src/save_image.cpp
                                 include/save_image.h)
                          
target_include_directories(save-image PUBLIC include 
                                             include/external)

target_link_libraries(save-image ${TORCH_LIBRARIES})

set_target_properties(save-image PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)