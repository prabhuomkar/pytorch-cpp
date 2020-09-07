// Copyright 2020-present pytorch-cpp Authors
#include "nnet.h"
#include <torch/torch.h>

NetImpl::NetImpl() :
    conv1(torch::nn::Conv2dOptions(3, 6, 5)),
    pool(torch::nn::MaxPool2dOptions({2, 2})),
    conv2(torch::nn::Conv2dOptions(6, 16, 5)),
    fc1(torch::nn::LinearOptions(16 * 5 * 5, 120)),
    fc2(torch::nn::LinearOptions(120, 84)),
    fc3(torch::nn::LinearOptions(84, 10)) {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
}

torch::Tensor NetImpl::forward(torch::Tensor x) {
    auto out = pool->forward(torch::relu(conv1->forward(x)));
    out = pool->forward(torch::relu(conv2->forward(out)));
    out = out.view({-1, 16 * 5 * 5});
    out = torch::relu(fc1->forward(out));
    out = torch::relu(fc2->forward(out));
    return fc3->forward(out);
}
