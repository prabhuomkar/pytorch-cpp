// Copyright 2020-present pytorch-cpp Authors
#include "load_image.h"
#include <torch/torch.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

// Loads a tensor from an image file
torch::Tensor image_utils::load_image(const std::string& file_path,
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
