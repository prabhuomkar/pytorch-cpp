// Copyright 2019 Markus Fleischhacker
#pragma once

#include <torch/torch.h>
#include <vector>

namespace nn_utils {
    void clip_grad_l2_norm(std::vector<torch::Tensor> parameters, double max_norm);
}  // namespace nn_utils
