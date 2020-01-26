// Copyright 2020 Markus Fleischhacker
#pragma once

#include <torch/torch.h>
#include <string>

namespace image_utils {
    torch::Tensor load_image(const std::string& file_path,
        torch::IntArrayRef shape = {},
        std::function<torch::Tensor(torch::Tensor)> transform = [] (torch::Tensor x) { return x; });
}  // namespace image_utils
