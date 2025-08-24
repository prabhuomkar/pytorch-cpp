cmake_minimum_required(VERSION 3.28.6 FATAL_ERROR)

project(pytorch-cpp VERSION 1.0.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(DOWNLOAD_DATASETS "Automatically download required datasets at build-time." ON)
option(CREATE_SCRIPTMODULES "Automatically create all required scriptmodule files at build-time (requires python3)." OFF)

set(PYTORCH_VERSION "2.8.0")
set(PYTORCH_MIN_VERSION "1.12.0")

find_package(Torch QUIET PATHS "${CMAKE_SOURCE_DIR}/libtorch")

if((NOT Torch_FOUND) OR (("${Torch_VERSION}" VERSION_LESS "${PYTORCH_MIN_VERSION}") OR
                         ("${Torch_VERSION}" VERSION_GREATER "${PYTORCH_VERSION}")))
    unset(Torch_FOUND)
    message(STATUS "Could not find compatible Torch version (>= ${PYTORCH_MIN_VERSION}, <= ${PYTORCH_VERSION})")
    include(fetch_libtorch)
endif()

message(STATUS "Torch version ${Torch_VERSION}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

if(CREATE_SCRIPTMODULES)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

set(EXECUTABLE_NAME pytorch-cpp)

add_executable(${EXECUTABLE_NAME})

target_sources(${EXECUTABLE_NAME} PRIVATE main.cpp)

set_target_properties(${EXECUTABLE_NAME} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)

target_link_libraries(${EXECUTABLE_NAME} ${TORCH_LIBRARIES})

# External dependencies
add_subdirectory("extern")

# Utils
add_subdirectory("utils/image_io")

# Dataset fetching
if(DOWNLOAD_DATASETS)
    set(DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data CACHE PATH "Dataset download directory")
    file(MAKE_DIRECTORY ${DATA_DIR})

    add_custom_target(mnist COMMAND ${CMAKE_COMMAND}
        -D DATA_DIR=${DATA_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/fetch_mnist.cmake)
    add_custom_target(cifar10 COMMAND ${CMAKE_COMMAND}
        -D DATA_DIR=${DATA_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/fetch_cifar10.cmake)
    add_custom_target(penntreebank COMMAND ${CMAKE_COMMAND}
        -D DATA_DIR=${DATA_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/fetch_penntreebank.cmake)
    add_custom_target(neural_style_transfer_images COMMAND ${CMAKE_COMMAND}
        -D DATA_DIR=${DATA_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/fetch_neural_style_transfer_images.cmake)
    add_custom_target(flickr8k COMMAND ${CMAKE_COMMAND}
        -D DATA_DIR=${DATA_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/fetch_flickr8k.cmake)
    add_custom_target(imagenette COMMAND ${CMAKE_COMMAND}
        -D DATA_DIR=${DATA_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/fetch_imagenette.cmake)
endif()

# Add tutorial sub-projects:

# Basics
add_subdirectory("tutorials/basics/feedforward_neural_network")
add_subdirectory("tutorials/basics/linear_regression")
add_subdirectory("tutorials/basics/logistic_regression")
add_subdirectory("tutorials/basics/pytorch_basics")

add_custom_target(basics)
add_dependencies(basics
    feedforward-neural-network
    linear-regression
    logistic-regression
    pytorch-basics)

# Intermediate
add_subdirectory("tutorials/intermediate/convolutional_neural_network")
add_subdirectory("tutorials/intermediate/deep_residual_network")
add_subdirectory("tutorials/intermediate/recurrent_neural_network")
add_subdirectory("tutorials/intermediate/bidirectional_recurrent_neural_network")
add_subdirectory("tutorials/intermediate/language_model")

add_custom_target(intermediate)
add_dependencies(intermediate
    convolutional-neural-network
    deep-residual-network
    recurrent-neural-network
    bidirectional-recurrent-neural-network
    language-model)

# Advanced
add_subdirectory("tutorials/advanced/generative_adversarial_network")
add_subdirectory("tutorials/advanced/variational_autoencoder")
add_subdirectory("tutorials/advanced/neural_style_transfer")
add_subdirectory("tutorials/advanced/image_captioning")

add_custom_target(advanced)
add_dependencies(advanced
    generative-adversarial-network
    variational-autoencoder
    neural-style-transfer
    image-captioning)

# Popular
add_subdirectory("tutorials/popular/blitz/tensors")
add_subdirectory("tutorials/popular/blitz/autograd")
add_subdirectory("tutorials/popular/blitz/neural_networks")
add_subdirectory("tutorials/popular/blitz/training_a_classifier")

add_custom_target(popular)
add_dependencies(popular
    tensors
    autograd
    neural-networks
    training-a-classifier)

if(MSVC)
    include(copy_torch_dlls)
    copy_torch_dlls(${EXECUTABLE_NAME})
endif()
