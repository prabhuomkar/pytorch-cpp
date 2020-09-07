// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>

class NetImpl : public torch::nn::Module {
 public:
    NetImpl();
    torch::Tensor forward(torch::Tensor x);

    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
    torch::nn::Linear fc3;

 private:
    int num_flat_features(torch::Tensor x);
};

TORCH_MODULE(Net);
