// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>
#include <utility>

class EncoderCNNImpl : public torch::nn::Module {
 public:
    EncoderCNNImpl(const std::string &backbone_scriptmodule_file_path, int64_t out_wh, int64_t out_size);

    torch::Tensor forward(torch::Tensor x);

    void to(torch::Device device, bool non_blocking = false) override;

    void to(torch::ScalarType dtype, bool non_blocking = false) override;

    void to(torch::Device device, torch::ScalarType dtype, bool non_blocking = false) override;

 private:
    torch::jit::script::Module backbone;
    torch::nn::AdaptiveAvgPool2d pool;
    torch::nn::BatchNorm2d bn;
    torch::nn::Conv2d conv{nullptr};
};

TORCH_MODULE(EncoderCNN);
