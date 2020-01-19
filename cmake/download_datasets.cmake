cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

message(STATUS "Fetching datasets")
include(download_mnist)
include(download_cifar10)
include(download_penntreebank)
include(download_neural_style_transfer_images)
message(STATUS "Fetching datasets - done")