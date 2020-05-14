cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(neural-style-transfer VERSION 1.0.0 LANGUAGES CXX)

if(NOT Torch_FOUND)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake")
    find_package(Torch REQUIRED PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../../libtorch")
endif()

set(EXECUTABLE_NAME neural-style-transfer)

add_executable(${EXECUTABLE_NAME})
target_sources(${EXECUTABLE_NAME} PRIVATE src/main.cpp
                                          src/vggnet.cpp
                                          include/vggnet.h
)
target_include_directories(${EXECUTABLE_NAME} PRIVATE include)

target_link_libraries(${EXECUTABLE_NAME} "${TORCH_LIBRARIES}" save-image load-image)

set_target_properties(${EXECUTABLE_NAME} PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

add_custom_command(TARGET ${EXECUTABLE_NAME}
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E make_directory output)

if(MSVC)
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR})
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR})
    include(copy_torch_dlls)
    copy_torch_dlls(${EXECUTABLE_NAME})
endif(MSVC)