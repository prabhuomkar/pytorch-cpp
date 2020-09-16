cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(logistic-regression VERSION 1.0.0 LANGUAGES CXX)

set(EXECUTABLE_NAME logistic-regression)

add_executable(${EXECUTABLE_NAME})
target_sources(${EXECUTABLE_NAME} PRIVATE main.cpp)

target_link_libraries(${EXECUTABLE_NAME} ${TORCH_LIBRARIES})

set_target_properties(${EXECUTABLE_NAME} PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

if(DOWNLOAD_DATASETS)
    add_dependencies(${EXECUTABLE_NAME} mnist)
endif()

if(MSVC)
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR})
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR})
    include(copy_torch_dlls)
    copy_torch_dlls(${EXECUTABLE_NAME})
endif(MSVC)
