// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>
#include "positional_encoding.h"

class TransformerImpl : public torch::nn::Module {
 public:
    TransformerImpl(int64_t ntoken,
                    int64_t ninp,
                    int64_t nhead,
                    int64_t nhid,
                    int64_t nlayers,
                    double dropout = 0.5);
    torch::Tensor forward(torch::Tensor src);

    torch::Tensor src_mask;
    PositionalEncoding pos_encoder;
    torch::nn::TransformerEncoderLayer encoder_layers;
    torch::nn::TransformerEncoder transformer_encoder;
    torch::nn::Embedding encoder;
    int64_t _ninp;
    torch::nn::Linear decoder;

 private:
    torch::Tensor _generate_square_subsequent_mask(int sz);
    void init_weights();
};

TORCH_MODULE(Transformer);
