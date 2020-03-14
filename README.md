<h1 align="center">
    <img src="images/pytorch_cpp.png" width="50%">
</h1>
<p align="center">
    C++ Implementation of PyTorch Tutorials for Deep Learning Researchers
    <br />
<img src="https://img.shields.io/travis/prabhuomkar/pytorch-cpp">
<img src="https://img.shields.io/github/license/prabhuomkar/pytorch-cpp">
<img src="https://img.shields.io/badge/libtorch-1.4-ee4c2c">
<img src="https://img.shields.io/badge/cmake-3.14-064f8d">
</p>

This repository provides tutorial code in C++ for deep learning researchers to learn PyTorch.  
**Python Tutorial**: [https://github.com/yunjey/pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial)

## Getting Started

### Requirements

1. [C++](http://www.cplusplus.com/doc/tutorial/introduction/)
2. [CMake](https://cmake.org/download/)
3. [LibTorch v1.4.0](https://pytorch.org/cppdocs/installing.html)

### Fork/Clone and Build

```bash
git clone https://github.com/prabhuomkar/pytorch-cpp.git
cd pytorch-cpp
```

#### Generate build system

```cmake
cmake -B build #<options>
```
> **_Note for Windows users:_**<br> 
> Libtorch only supports 64bit Windows and an x64 generator needs to be specified. For Visual Studio this can be done by appending `-A x64` to the above command.

Some useful options:

| Option       | Default           | Description  |
| :------------- |:------------|-----:|
| `-D CUDA_V=(9.2\|10.1\|none)`     | `none` | Download libtorch for a CUDA version (`none` = download CPU version). |
| `-D DOWNLOAD_DATASETS=(OFF\|ON)`     | `ON`      |   Download all datasets used in the tutorials. |
| `-D CMAKE_PREFIX_PATH=path/to/libtorch/share/cmake/Torch` |       |    Skip the downloading of libtorch and use your own local version instead. |

#### Build

```cmake
cmake --build build
```

>**_Note for Windows users:_** <br>
>The CMake script downloads the *Release* version of libtorch, so `--config Release` has to be appended to the build command.
>
>**_General Note:_** <br>
>By default all tutorials will be built. If you only want to build  one specific tutorial, specify the `target` parameter for the build command. For example to only build the language model tutorial, append `--target language-model` (target name = tutorial foldername with all underscores replaced with hyphens).

## Table of Contents

### 1. Basics
* [PyTorch Basics](tutorials/basics/pytorch_basics/main.cpp)
* [Linear Regression](tutorials/basics/linear_regression/main.cpp)
* [Logistic Regression](tutorials/basics/logistic_regression/main.cpp)
* [Feedforward Neural Network](tutorials/basics/feedforward_neural_network/src/main.cpp)

### 2. Intermediate
* [Convolutional Neural Network](tutorials/intermediate/convolutional_neural_network/src/main.cpp)
* [Deep Residual Network](tutorials/intermediate/deep_residual_network/src/main.cpp)
* [Recurrent Neural Network](tutorials/intermediate/recurrent_neural_network/src/main.cpp)
* [Bidirectional Recurrent Neural Network](tutorials/intermediate/bidirectional_recurrent_neural_network/src/main.cpp)
* [Language Model (RNN-LM)](tutorials/intermediate/language_model/src/main.cpp)

### 3. Advanced
* [Generative Adversarial Networks](tutorials/advanced/generative_adversarial_network/main.cpp)
* [Variational Auto-Encoder](tutorials/advanced/variational_autoencoder/src/main.cpp)
* [Neural Style Transfer](tutorials/advanced/neural_style_transfer/src/main.cpp)
* [Image Captioning (CNN-RNN)]()