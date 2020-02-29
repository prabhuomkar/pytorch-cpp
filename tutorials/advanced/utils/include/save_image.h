// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>
#include <string>

namespace image_utils {
    enum class ImageFormat { PNG, JPG, BMP };

    void save_image(torch::Tensor tensor,
        const std::string& file_path,
        int64_t nrow = 10, int64_t padding = 2,
        bool normalize = false,
        const std::vector<double>& range = {},
        bool scale_each = false,
        torch::Scalar pad_value = 0,
        ImageFormat format = ImageFormat::PNG);
}  // namespace image_utils
