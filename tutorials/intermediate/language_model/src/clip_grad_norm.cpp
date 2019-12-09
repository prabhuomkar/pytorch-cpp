// Copyright 2019 Markus Fleischhacker
#include "clip_grad_norm.h"
#include <torch/torch.h>
#include <vector>
#include <algorithm>

namespace nn_utils {
    // Clips gradient norm of a vector of tensors
    //
    // Source (slightly modified):
    // https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/nn/utils/clip_grad.h
    void clip_grad_l2_norm(std::vector<torch::Tensor> parameters, double max_norm) {
        std::vector<torch::Tensor> params_with_grad;

        for (const auto& param : parameters) {
            auto& grad = param.grad();
            if (grad.defined()) {
                params_with_grad.push_back(param);
            }
        }

        double total_norm = 0.0;

        for (const auto& param : params_with_grad) {
            auto param_norm = param.grad().data().norm(2.0);
            total_norm += std::pow(param_norm.item().toDouble(), 2.0);
        }
        total_norm = std::pow(total_norm, 1.0 / 2.0);

        auto clip_coef = max_norm / (total_norm + 1e-6);
        if (clip_coef < 1) {
            for (auto& param : params_with_grad) {
                param.grad().data().mul_(clip_coef);
            }
        }
    }
}  // namespace nn_utils
