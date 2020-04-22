// Copyright 2020-present pytorch-cpp Authors
#include "rnn_lm.h"
#include <torch/torch.h>
#include <tuple>

RNNLMImpl::RNNLMImpl(int64_t vocab_size, int64_t embed_size, int64_t hidden_size, int64_t num_layers)
    : embed(vocab_size, embed_size),
    lstm(torch::nn::LSTMOptions(embed_size, hidden_size).num_layers(num_layers).batch_first(true)),
    linear(hidden_size, vocab_size) {
    register_module("embed", embed);
    register_module("lstm", lstm);
    register_module("linear", linear);
}

std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> RNNLMImpl::forward(torch::Tensor x,
    std::tuple<torch::Tensor, torch::Tensor> hx) {
    torch::Tensor output;
    std::tuple<torch::Tensor, torch::Tensor> state;

    std::tie(output, state) = lstm->forward(embed->forward(x), hx);

    output = output.reshape({-1, output.size(2)});
    output = linear->forward(output);
    output = torch::nn::functional::log_softmax(output, 1);
    return std::make_tuple(output, state);
}
