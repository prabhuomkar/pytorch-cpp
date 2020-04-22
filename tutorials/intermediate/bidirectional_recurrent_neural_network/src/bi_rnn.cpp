// Copyright 2020-present pytorch-cpp Authors
#include "bi_rnn.h"
#include <torch/torch.h>

using torch::indexing::Slice;

BiRNNImpl::BiRNNImpl(int64_t input_size, int64_t hidden_size, int64_t num_layers, int64_t num_classes)
    : lstm(torch::nn::LSTMOptions(input_size, hidden_size)
        .num_layers(num_layers)
        .batch_first(true)
        .bidirectional(true)),
      fc(hidden_size * 2, num_classes) {
    register_module("lstm", lstm);
    register_module("fc", fc);
}

torch::Tensor BiRNNImpl::forward(torch::Tensor x) {
    auto out = std::get<0>(lstm->forward(x));  // out: tensor of shape (batch_size, sequence_length, 2 * hidden_size)

    // Concatenate the last hidden state of forward LSTM and first hidden state of backward LSTM
    //
    // Source: Translated from python code at
    // https://github.com/yunjey/pytorch-tutorial/pull/174/commits/8c0897ee93fed8d9b352d33a60c1f931c9be5351
    auto out_directions = out.chunk(2, 2);
    // Last hidden state of forward direction output
    auto out_1 = out_directions[0].index({Slice(), -1});  // out_1: tensor of shape (batch_size, hidden_size)
    // First hidden state of backward direction output
    auto out_2 = out_directions[1].index({Slice(), 0});  // out_2: tensor of shape (batch_size, hidden_size)
    auto out_cat = torch::cat({out_1, out_2}, 1);  // out_cat: tensor of shape (batch_size, 2 * hidden_size)

    return fc->forward(out_cat);
}

