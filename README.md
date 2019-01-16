<p align="center"><img width="50%" src="images/pytorch_logo.svg" /></p>

--------------------------------------------------------------------------------

This repository provides tutorial code in C++ for deep learning researchers to learn PyTorch.

## How to Build
- Install
```
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
```
- Build
```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
make
```

## Table of Contents

#### 1. Basics
* [PyTorch Basics](https://github.com/prabhuomkar/pytorch-cpp/tree/master/tutorials/basics/pytorch_basics.cpp)
* [Linear Regression](https://github.com/prabhuomkar/pytorch-cpp/tree/master/tutorials/basics/linear_regression.cpp)
* [Logistic Regression](https://github.com/prabhuomkar/pytorch-cpp/tree/master/tutorials/basics/logistic_regression.cpp)
* [Feedforward Neural Network](https://github.com/prabhuomkar/pytorch-cpp/tree/master/tutorials/basics/feedforward_neural_network.cpp)