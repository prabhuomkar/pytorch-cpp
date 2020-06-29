cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(pytorch-cpp VERSION 1.0.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

option(DOWNLOAD_DATASETS "Download datasets used in the tutorials." ON)

set(PYTORCH_VERSION "1.5.1")

find_package(Torch ${PYTORCH_VERSION} EXACT QUIET PATHS "${CMAKE_SOURCE_DIR}/libtorch")

if(NOT Torch_FOUND)
    unset(Torch_FOUND)
    include(download_libtorch)
endif()

if(DOWNLOAD_DATASETS)
    include(download_datasets)
endif()

set(EXECUTABLE_NAME pytorch-cpp)

add_executable(${EXECUTABLE_NAME})

target_sources(${EXECUTABLE_NAME} PRIVATE main.cpp)

set_target_properties(${EXECUTABLE_NAME} PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

target_link_libraries(${EXECUTABLE_NAME} ${TORCH_LIBRARIES})

# External dependencies
add_subdirectory("extern")

# Utils
add_subdirectory("utils/image_io")

# Add tutorial sub-projects:

# Basic
add_subdirectory("tutorials/basics/feedforward_neural_network")
add_subdirectory("tutorials/basics/linear_regression")
add_subdirectory("tutorials/basics/logistic_regression")
add_subdirectory("tutorials/basics/pytorch_basics")

# Intermediate
add_subdirectory("tutorials/intermediate/convolutional_neural_network")
add_subdirectory("tutorials/intermediate/deep_residual_network")
add_subdirectory("tutorials/intermediate/recurrent_neural_network")
add_subdirectory("tutorials/intermediate/bidirectional_recurrent_neural_network")
add_subdirectory("tutorials/intermediate/language_model")

# Advanced
add_subdirectory("tutorials/advanced/generative_adversarial_network")
add_subdirectory("tutorials/advanced/variational_autoencoder")
add_subdirectory("tutorials/advanced/neural_style_transfer")

if(MSVC)
    include(copy_torch_dlls)
    copy_torch_dlls(${EXECUTABLE_NAME})
endif()
