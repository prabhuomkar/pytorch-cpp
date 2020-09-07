// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>

class NetImpl : public torch::nn::Module {
 public:
    NetImpl();
    torch::Tensor forward(torch::Tensor x);

 private:
    torch::nn::Conv2d conv1;
    torch::nn::MaxPool2d pool;
    torch::nn::Conv2d conv2;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
    torch::nn::Linear fc3;
};

TORCH_MODULE(Net);
