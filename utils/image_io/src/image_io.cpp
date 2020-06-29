// Copyright 2020-present pytorch-cpp Authors
#include "image_io.h"
#include <torch/torch.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

namespace image_io {
namespace {
    void norm_ip(torch::Tensor img, double min, double max) {
        img.clamp_(min, max);
        img.sub_(min).div_(max - min + 1e-5);
    }

    void norm_range(torch::Tensor t, const std::vector<double>& range) {
        if (range.empty()) {
            norm_ip(t, t.min().item<double>(), t.max().item<double>());
        } else if (range.size() == 2) {
            norm_ip(t, range[0], range[1]);
        } else {
            throw std::invalid_argument("Range must either be empty or contain exactly 2 elements.");
        }
    }

    // Makes a grid of images
    //
    // Translated from source python code at
    // https://pytorch.org/docs/stable/_modules/torchvision/utils.html#make_grid
    torch::Tensor make_grid(torch::Tensor tensor, int64_t nrow, int64_t padding,
        bool normalize, const std::vector<double>& range, bool scale_each, torch::Scalar pad_value) {
        if (tensor.dim() == 2) {
            tensor = tensor.unsqueeze(0);
        }

        if (tensor.dim() == 3) {
            if (tensor.size(0) == 1) {
                tensor = torch::cat({tensor, tensor, tensor}, 0);
            }
            tensor = tensor.unsqueeze(0);
        }

        if (tensor.dim() == 4 && tensor.size(1) == 1) {
            tensor = torch::cat({tensor, tensor, tensor}, 1);
        }

        tensor = tensor.to(torch::kFloat32);

        if (normalize) {
            if (scale_each) {
                auto mini_batches = tensor.chunk(tensor.size(1), 1);
                for_each(mini_batches.begin(), mini_batches.end(), [&range] (auto& t) { norm_range(t, range); });
            } else {
                norm_range(tensor, range);
            }
        }

        if (tensor.size(0) == 1) {
            return tensor.squeeze(0);
        }

        auto nmaps = tensor.size(0);
        auto xmaps = std::min(nrow, nmaps);
        auto ymaps = static_cast<int64_t>(std::ceil(static_cast<double>(nmaps) / xmaps));
        auto height = tensor.size(2) + padding;
        auto width = tensor.size(3) + padding;
        auto num_channels = tensor.size(1);

        auto grid = torch::full({num_channels, height * ymaps + padding, width * xmaps + padding},
            pad_value, tensor.dtype());

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

// Saves a tensor to an image file.
//
// Translated (and slightly modified) from source python code at
// https://pytorch.org/docs/stable/_modules/torchvision/utils.html#save_image
void save_image(torch::Tensor tensor, const std::string& file_path, int64_t nrow, int64_t padding,
    bool normalize, const std::vector<double>& range,
    bool scale_each, torch::Scalar pad_value, ImageFormat format) {
    auto grid = make_grid(tensor, nrow, padding, normalize, range, scale_each, pad_value)
        .mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute({1, 2, 0})
        .to(torch::kCPU, torch::kUInt8, false, false, torch::MemoryFormat::Contiguous);

    switch (format) {
        case ImageFormat::PNG:
            stbi_write_png(file_path.c_str(), grid.size(1), grid.size(0), grid.size(2),
                grid.data_ptr(), grid.stride(0));
            break;
        case ImageFormat::BMP:
            stbi_write_bmp(file_path.c_str(), grid.size(1), grid.size(0), grid.size(2), grid.data_ptr());
            break;
        case ImageFormat::JPG:
            stbi_write_jpg(file_path.c_str(), grid.size(1), grid.size(0), grid.size(2), grid.data_ptr(), 100);
            break;
        default:
            throw std::runtime_error("Unknown file format.");
    }
}

// Loads a tensor from an image file
torch::Tensor load_image(const std::string& file_path,
    torch::IntArrayRef shape, std::function<torch::Tensor(torch::Tensor)> transform) {
    if (!shape.empty() && shape.size() != 1 && shape.size() != 2) {
        throw std::invalid_argument("Shape must be empty or contain exactly one or two elements.");
    }

    int width = 0;
    int height = 0;
    int depth = 0;

    std::unique_ptr<unsigned char, decltype(&stbi_image_free)> image_raw(stbi_load(file_path.c_str(),
        &width, &height, &depth, 0), &stbi_image_free);

    if (!image_raw) {
        throw std::runtime_error("Unable to load image file " + file_path + ".");
    }

    if (shape.empty()) {
        return transform(torch::from_blob(image_raw.get(),
            {height, width, depth}, torch::kUInt8).clone().to(torch::kFloat32).permute({2, 0, 1}).div_(255));
    }

    int new_width = 0;
    int new_height = 0;

    if (shape.size() == 1) {
        double scale = static_cast<double>(shape[0]) / std::max(width, height);
        new_width = width * scale;
        new_height = height * scale;
    } else {
        new_width = shape[1];
        new_height = shape[0];
    }

    if (new_width < 0 || new_height < 0) {
        throw std::invalid_argument("Invalid shape.");
    }

    size_t buffer_size = new_width * new_height * depth;

    std::vector<unsigned char> image_resized_buffer(buffer_size);

    stbir_resize_uint8(image_raw.get(), width, height, 0,
        image_resized_buffer.data(), new_width, new_height, 0, depth);

    return transform(torch::from_blob(image_resized_buffer.data(),
        {new_height, new_width, depth}, torch::kUInt8).clone().to(torch::kFloat32).permute({2, 0, 1}).div_(255));
}
}  // namespace image_io
