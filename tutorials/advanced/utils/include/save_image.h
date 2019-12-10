// Copyright 2019 Markus Fleischhacker
#pragma once

#include <torch/torch.h>
#include <string>

namespace image_utils {
    void save_image(torch::Tensor tensor, const std::string& file_path, int64_t nrow = 10, int64_t padding = 2);
}  // namespace image_utils
