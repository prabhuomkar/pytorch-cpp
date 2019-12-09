// Copyright 2019 Markus Fleischhacker
#pragma once

#include <torch/torch.h>

class RNNLMImpl : public torch::nn::Module {
 public:
    RNNLMImpl(int64_t vocab_size, int64_t embed_size, int64_t hidden_size, int64_t num_layers);
    torch::nn::RNNOutput forward(torch::Tensor x, torch::Tensor h);

 private:
    torch::nn::Embedding embed;
    torch::nn::LSTM lstm;
    torch::nn::Linear linear;
};

TORCH_MODULE(RNNLM);
