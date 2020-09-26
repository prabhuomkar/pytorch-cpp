// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>

class PositionalEncodingImpl : public torch::nn::Module {
 public:
    PositionalEncodingImpl(int64_t d_model,
                           double dropout = 0.1,
                           int64_t max_len = 5000);
    torch::Tensor forward(torch::Tensor x);

    torch::nn::Dropout drpout;
    torch::Tensor pe;
};

TORCH_MODULE(PositionalEncoding);
