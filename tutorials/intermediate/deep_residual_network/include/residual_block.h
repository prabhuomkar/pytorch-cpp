// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>

namespace resnet {
class ResidualBlockImpl : public torch::nn::Module {
 public:
    ResidualBlockImpl(int64_t in_channels, int64_t out_channels, int64_t stride = 1,
        torch::nn::Sequential downsample = nullptr);
    torch::Tensor forward(torch::Tensor x);

 private:
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm2d bn1;
    torch::nn::ReLU relu;
    torch::nn::Conv2d conv2;
    torch::nn::BatchNorm2d bn2;
    torch::nn::Sequential downsampler;
};

torch::nn::Conv2d conv3x3(int64_t in_channels, int64_t out_channels, int64_t stride = 1);

TORCH_MODULE(ResidualBlock);
}  // namespace resnet
