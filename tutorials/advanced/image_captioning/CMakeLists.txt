cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(image-captioning VERSION 1.0.0 LANGUAGES CXX)

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

if(DOWNLOAD_DATASETS)
    add_dependencies(${EXECUTABLE_NAME} flickr8k)
endif()

add_custom_command(TARGET ${EXECUTABLE_NAME}
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E make_directory output)
               
if(CREATE_SCRIPTMODULES AND NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/model/encoder_cnn_backbone.pt")
    execute_process(COMMAND ${Python3_EXECUTABLE} "create_encoder_cnn_backbone_scriptmodule.py" 
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/model")
endif()

if(MSVC)
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR})
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR})
    include(copy_torch_dlls)
    copy_torch_dlls(${EXECUTABLE_NAME})
endif(MSVC)
