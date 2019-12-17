// Copyright 2019 Markus Fleischhacker
#include "save_image.h"
#include <torch/torch.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace image_utils {
namespace {
    // Makes a grid of images
    //
    // Translated and slightly modified from source python code at
    // https://pytorch.org/docs/stable/_modules/torchvision/utils.html#make_grid
    torch::Tensor make_grid(torch::Tensor tensor, int64_t nrow, int64_t padding) {
        if (tensor.size(0) == 1) {
            return tensor.squeeze(0);
        }

        auto nmaps = tensor.size(0);
        auto xmaps = std::min(nrow, nmaps);
        auto ymaps = static_cast<int64_t>(std::ceil(static_cast<double>(nmaps) / xmaps));
        auto height = tensor.size(2) + padding;
        auto width = tensor.size(3) + padding;
        auto num_channels = tensor.size(1);

        auto grid = torch::full({num_channels, height * ymaps + padding, width * xmaps + padding}, 0);

        int64_t k = 0;

        for (int64_t y = 0; y != ymaps; ++y) {
            for (int64_t x = 0; x != xmaps; ++x) {
                if (k >= nmaps) {
                    break;
                }

                grid.narrow(1, y * height + padding, height - padding)
                    .narrow(2, x * width + padding, width - padding)
                    .copy_(tensor[k]);
                ++k;
            }
        }
        return grid;
    }
}  // namespace

void save_image(torch::Tensor tensor, const std::string& file_path, int64_t nrow, int64_t padding) {
    auto grid = make_grid(tensor, nrow, padding)
        .mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute({1, 2, 0})
        .to(torch::kCPU, torch::kUInt8);

    stbi_write_png(file_path.c_str(), grid.size(1), grid.size(0), grid.size(2), grid.data_ptr(), grid.stride(0));
}
}  // namespace image_utils
