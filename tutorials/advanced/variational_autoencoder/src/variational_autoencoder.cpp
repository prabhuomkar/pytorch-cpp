// Copyright 2020-present pytorch-cpp Authors
#include "variational_autoencoder.h"
#include <utility>

VAEImpl::VAEImpl(int64_t image_size, int64_t h_dim, int64_t z_dim)
    : fc1(image_size, h_dim),
      fc2(h_dim, z_dim),
      fc3(h_dim, z_dim),
      fc4(z_dim, h_dim),
      fc5(h_dim, image_size) {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
    register_module("fc4", fc4);
    register_module("fc5", fc5);
}

std::pair<torch::Tensor, torch::Tensor> VAEImpl::encode(torch::Tensor x) {
    auto h = torch::nn::functional::relu(fc1->forward(x));
    return {fc2->forward(h), fc3->forward(h)};
}

torch::Tensor VAEImpl::reparameterize(torch::Tensor mu, torch::Tensor log_var) {
    if (is_training()) {
        auto std = log_var.div(2).exp_();
        auto eps = torch::randn_like(std);
        return eps.mul(std).add_(mu);
    } else {
        // During inference, return mean of the learned distribution
        // See https://vxlabs.com/2017/12/08/variational-autoencoder-in-pytorch-commented-and-annotated/
        return mu;
    }
}

torch::Tensor VAEImpl::decode(torch::Tensor z) {
    auto h = torch::nn::functional::relu(fc4->forward(z));
    return torch::sigmoid(fc5->forward(h));
}

VAEOutput VAEImpl::forward(torch::Tensor x) {
    auto encode_output = encode(x);
    auto mu = encode_output.first;
    auto log_var = encode_output.second;
    auto z = reparameterize(mu, log_var);
    auto x_reconstructed = decode(z);
    return {x_reconstructed, mu, log_var};
}
