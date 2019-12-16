// Copyright 2019 Markus Fleischhacker
#include "neural_net.h"
#include <torch/torch.h>

NeuralNetImpl::NeuralNetImpl(int64_t input_size, int64_t hidden_size, int64_t num_classes)
    : fc1(input_size, hidden_size), fc2(hidden_size, num_classes) {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
}

torch::Tensor NeuralNetImpl::forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    x = fc2->forward(x);
    return torch::log_softmax(x, 1);
}
