cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(pytorch-cpp VERSION 1.0.0 LANGUAGES CXX)

find_package(Torch REQUIRED)

set(EXECUTABLE_NAME pytorch-cpp)
add_executable(${EXECUTABLE_NAME} main.cpp)
set_target_properties(${EXECUTABLE_NAME} PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
)

target_link_libraries(${EXECUTABLE_NAME} "${TORCH_LIBRARIES}")

# Add tutorial projects:

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

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
# See https://pytorch.org/cppdocs/installing.html.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${EXECUTABLE_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${EXECUTABLE_NAME}>)
endif (MSVC)
