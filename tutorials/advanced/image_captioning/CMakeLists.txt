cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(image-captioning VERSION 1.0.0 LANGUAGES CXX)

if(NOT Torch_FOUND)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake")
    find_package(Torch REQUIRED PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../../libtorch")
endif()

set(EXECUTABLE_NAME image-captioning)

add_executable(${EXECUTABLE_NAME})
target_sources(${EXECUTABLE_NAME} PRIVATE src/main.cpp
                                          include/vocabulary.h
                                          src/vocabulary.cpp
                                          include/caption_dataset.h
                                          src/caption_dataset.cpp
                                          include/transform.h
                                          src/transform.cpp
                                          include/encoder_cnn.h
                                          src/encoder_cnn.cpp
                                          include/decoder_rnn.h
                                          src/decoder_rnn.cpp
                                          include/score.h
                                          src/score.cpp
                                          include/scheduler.h
                                          src/data_utils.cpp
                                          include/data_utils.h
                                          include/validate.h
)

target_include_directories(${EXECUTABLE_NAME} PRIVATE include)

target_link_libraries(${EXECUTABLE_NAME} ${TORCH_LIBRARIES} image-io cxxopts)

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