// Copyright 2019 Markus Fleischhacker
#pragma once

#include <torch/torch.h>

class ConvNetImpl : public torch::nn::Module {
 public:
    explicit ConvNetImpl(int64_t num_classes = 10);
    torch::Tensor forward(torch::Tensor x);

 private:
    torch::nn::Sequential layer1{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 5).stride(1).padding(2)),
        torch::nn::BatchNorm(16),
        torch::nn::Functional(torch::relu),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
    };

    torch::nn::Sequential layer2{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 5).stride(1).padding(2)),
        torch::nn::BatchNorm(32),
        torch::nn::Functional(torch::relu),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
    };

    torch::nn::Linear fc;
};

TORCH_MODULE(ConvNet);


