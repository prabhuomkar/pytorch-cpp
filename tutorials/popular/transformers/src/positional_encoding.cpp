// Copyright 2020-present pytorch-cpp Authors
#include "positional_encoding.h"
#include <torch/torch.h>
#include <cmath>

using torch::indexing::Slice;
using torch::indexing::None;

PositionalEncodingImpl::PositionalEncodingImpl(int64_t d_model,
                                               double dropout,
                                               int64_t max_len)
        : drpout(dropout) {
    pe = torch::zeros({max_len, d_model});
    auto position = torch::arange(0, max_len,
        torch::TensorOptions(torch::kFloat32)).unsqueeze_(1);
    auto div_term = torch::exp(torch::arange(0, d_model, 2,
        torch::TensorOptions(torch::kFloat32)) * (-std::log(10000.0) / d_model));
    pe.index({Slice(), Slice({0, None, 2})}) = torch::sin(position * div_term);
    pe.index({Slice(), Slice({1, None, 2})}) = torch::cos(position * div_term);
    pe = pe.unsqueeze_(0).transpose_(0, 1);

    register_buffer("pe", pe);
    register_module("drpout", drpout);
}

torch::Tensor PositionalEncodingImpl::forward(torch::Tensor x) {
    auto out = x + pe.index({Slice(None, x.size(0)), Slice()});
    return drpout->forward(out);
}
