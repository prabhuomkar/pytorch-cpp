// Copyright 2019 Markus Fleischhacker
#include "rnn_lm.h"
#include <torch/torch.h>

RNNLMImpl::RNNLMImpl(int64_t vocab_size, int64_t embed_size, int64_t hidden_size, int64_t num_layers)
    : embed(vocab_size, embed_size),
    lstm(torch::nn::LSTMOptions(embed_size, hidden_size).layers(num_layers).batch_first(true)),
    linear(hidden_size, vocab_size) {
    register_module("embed", embed);
    register_module("lstm", lstm);
    register_module("linear", linear);
}

torch::nn::RNNOutput RNNLMImpl::forward(torch::Tensor x, torch::Tensor h) {
    auto lstm_out = lstm->forward(embed->forward(x), h);
    auto out = lstm_out.output;
    out = out.reshape({-1, out.size(2)});
    out = linear->forward(out);
    out = torch::log_softmax(out, 1);
    return {out, lstm_out.state};
}
