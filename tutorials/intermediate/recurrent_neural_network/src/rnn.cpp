// Copyright 2019 Markus Fleischhacker
#include "rnn.h"
#include <torch/torch.h>

RNNImpl::RNNImpl(int64_t input_size, int64_t hidden_size, int64_t num_layers, int64_t num_classes)
    : lstm(torch::nn::LSTMOptions(input_size, hidden_size).layers(num_layers).batch_first(true)),
      fc(hidden_size, num_classes) {
    register_module("lstm", lstm);
    register_module("fc", fc);
}

torch::Tensor RNNImpl::forward(torch::Tensor x) {
    auto out = lstm->forward(x)
        .output
        .slice(1, -1)
        .squeeze(1);
    return fc->forward(out);
}
