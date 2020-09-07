// Copyright 2020-present pytorch-cpp Authors
#include "nnet.h"
#include <torch/torch.h>

NetImpl::NetImpl() :
    conv1(torch::nn::Conv2dOptions(1, 6, 3)),
    conv2(torch::nn::Conv2dOptions(6, 16, 3)),
    fc1(torch::nn::LinearOptions(16 * 6 * 6, 120)),
    fc2(torch::nn::LinearOptions(120, 84)),
    fc3(torch::nn::LinearOptions(84, 10)) {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
}

int NetImpl::num_flat_features(torch::Tensor x) {
    auto sz = x.sizes().slice(1);  // all dimensions except the batch dimension
    int num_features = 1;
    for (auto s : sz) {
        num_features *= s;
    }
    return num_features;
}

torch::Tensor NetImpl::forward(torch::Tensor x) {
    // Max pooling over a (2, 2) window
    auto out = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2, 2}))->forward(torch::relu(conv1->forward(x)));
    // If the size is a square you can only specify a single number
    out = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2))->forward(torch::relu(conv2->forward(out)));
    out = out.view({-1, num_flat_features(out)});
    out = torch::relu(fc1->forward(out));
    out = torch::relu(fc2->forward(out));
    return fc3->forward(out);
}
