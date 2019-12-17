<p align="center"><img width="50%" src="images/pytorch_logo.svg" /></p>

--------------------------------------------------------------------------------
[![Build Status](https://travis-ci.org/prabhuomkar/pytorch-cpp.svg?branch=master)](https://travis-ci.org/prabhuomkar/pytorch-cpp)
![MIT License](https://img.shields.io/github/license/prabhuomkar/pytorch-cpp)
![C++ PyTorch](https://img.shields.io/badge/c%2B%2B-pytorch-orange) 

This repository provides tutorial code in C++ for deep learning researchers to learn PyTorch.  
**Python Tutorial**: [https://github.com/yunjey/pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial)

## Getting Started
- Fork/Clone and Install
```bash
$ git clone https://github.com/prabhuomkar/pytorch-cpp.git
$ chmod +x scripts.sh
$ ./scripts.sh install #optional: --cuda=(9.2 or 10.1) to install libtorch cuda versions (by default cpu version is installed) 
```
- Download all datasets used in the tutorials
```bash
$ ./scripts.sh download_datasets
```
- Build
```bash
$ ./scripts.sh build
```

## Table of Contents

#### 1. Basics
* [PyTorch Basics](tutorials/basics/pytorch_basics/main.cpp)
* [Linear Regression](tutorials/basics/linear_regression/main.cpp)
* [Logistic Regression](tutorials/basics/logistic_regression/main.cpp)
* [Feedforward Neural Network](tutorials/basics/feedforward_neural_network/src/main.cpp)

#### 2. Intermediate
* [Convolutional Neural Network](tutorials/intermediate/convolutional_neural_network/src/main.cpp)
* [Deep Residual Network](tutorials/intermediate/deep_residual_network/src/main.cpp)
* [Recurrent Neural Network](tutorials/intermediate/recurrent_neural_network/src/main.cpp)
* [Bidirectional Recurrent Neural Network](tutorials/intermediate/bidirectional_recurrent_neural_network/src/main.cpp)
* [Language Model (RNN-LM)](tutorials/intermediate/language_model/src/main.cpp)

#### 3. Advanced
* [Generative Adversarial Networks](tutorials/advanced/generative_adversarial_network/main.cpp)
* [Variational Auto-Encoder](tutorials/advanced/variational_autoencoder/src/main.cpp)
* [Neural Style Transfer]()
* [Image Captioning (CNN-RNN)]()

## Dependencies
- C++
- PyTorch C++ API

## Authors
- Omkar Prabhu - [prabhuomkar](https://github.com/prabhuomkar)
- Markus Fleischhacker - [mfl28](https://github.com/mfl28)
 
