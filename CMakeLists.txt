cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pytorch-cpp)


find_package(Torch REQUIRED)

add_executable(pytorch-cpp main.cpp)
target_link_libraries(pytorch-cpp "${TORCH_LIBRARIES}")

add_executable(pytorch_basics tutorials/basics/pytorch_basics/main.cpp)
target_link_libraries(pytorch_basics "${TORCH_LIBRARIES}")
# add_executable(linear_regression tutorials/basics/linear_regression/main.cpp)
# target_link_libraries(linear_regression "${TORCH_LIBRARIES}")
# add_executable(logistic_regression tutorials/basics/logistic_regression/main.cpp)
# target_link_libraries(logistic_regression "${TORCH_LIBRARIES}")
# add_executable(feedforward_neural_network tutorials/basics/feedforward_neural_network/main.cpp)
# target_link_libraries(feedforward_neural_network "${TORCH_LIBRARIES}")

set_property(TARGET pytorch-cpp PROPERTY CXX_STANDARD 11)