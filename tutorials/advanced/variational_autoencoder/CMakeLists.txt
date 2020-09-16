cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(variational-autoencoder VERSION 1.0.0 LANGUAGES CXX)

set(EXECUTABLE_NAME variational-autoencoder)

add_executable(${EXECUTABLE_NAME})
target_sources(${EXECUTABLE_NAME} PRIVATE src/main.cpp
                                          src/variational_autoencoder.cpp
                                          include/variational_autoencoder.h
)
target_include_directories(${EXECUTABLE_NAME} PRIVATE include)

target_link_libraries(${EXECUTABLE_NAME} ${TORCH_LIBRARIES} image-io)

set_target_properties(${EXECUTABLE_NAME} PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

if(DOWNLOAD_DATASETS)
    add_dependencies(${EXECUTABLE_NAME} mnist)
endif()

add_custom_command(TARGET ${EXECUTABLE_NAME}
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E make_directory output)

if(MSVC)
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR})
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR})
    include(copy_torch_dlls)
    copy_torch_dlls(${EXECUTABLE_NAME})
endif(MSVC)
