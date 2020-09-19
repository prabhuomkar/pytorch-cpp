cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(pytorch-basics VERSION 1.0.0 LANGUAGES CXX)

set(EXECUTABLE_NAME pytorch-basics)

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

add_custom_command(TARGET ${EXECUTABLE_NAME}
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E make_directory output)

if(CREATE_SCRIPTMODULES AND NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/model/resnet18_scriptmodule.pt")
    execute_process(COMMAND ${Python3_EXECUTABLE} "create_resnet18_scriptmodule.py" 
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/model")
endif()

if(MSVC)
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR})
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR})
    include(copy_torch_dlls)
    copy_torch_dlls(${EXECUTABLE_NAME})
endif(MSVC)
