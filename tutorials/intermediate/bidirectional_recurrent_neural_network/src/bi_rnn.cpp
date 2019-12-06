// Copyright 2019 Markus Fleischhacker
#include "bi_rnn.h"
#include <torch/torch.h>

BiRNNImpl::BiRNNImpl(int64_t input_size, int64_t hidden_size, int64_t num_layers, int64_t num_classes)
    : lstm(torch::nn::LSTMOptions(input_size, hidden_size).layers(num_layers).batch_first(true).bidirectional(true)),
      fc(hidden_size * 2, num_classes) {
    register_module("lstm", lstm);
    register_module("fc", fc);
}

torch::Tensor BiRNNImpl::forward(torch::Tensor x) {
    auto out = lstm->forward(x)
        .output
        .slice(1, -1)
        .squeeze(1);
    out = fc->forward(out);
    return torch::log_softmax(out, 1);
}
