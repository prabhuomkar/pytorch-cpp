// Copyright 2020-present pytorch-cpp Authors
#include "transform.h"
#include <cmath>
#include <vector>

using torch::indexing::Slice;
using torch::indexing::Ellipsis;
using torch::indexing::None;

namespace transform {
double rand_double() {
    return torch::rand(1)[0].item<double>();
}

int64_t rand_int(int64_t max) {
    return torch::randint(max, 1)[0].item<int64_t>();
}

ImageCaptionExample ImageCaptionCollate::apply_batch(std::vector<dataset::ImageCaptionSample> data) {
    std::sort(data.begin(), data.end(), [](const auto &left, const auto &right) {
        return left.target.caption().size(0) > right.target.caption().size(0);
    });

    std::vector<int64_t> lengths;
    lengths.reserve(data.size());

    for (const auto &entry : data) {
        lengths.push_back(entry.target.caption().size(0));
    }

    auto max_caption_length = lengths.front();
    std::vector<torch::data::Example<>> image_caption_examples;
    image_caption_examples.reserve(data.size());

    std::vector<std::vector<torch::Tensor>> all_captions;
    all_captions.reserve(data.size());

    for (auto &entry : data) {
        image_caption_examples.emplace_back(entry.data, torch::nn::functional::pad(entry.target.caption(),
                torch::nn::functional::PadFuncOptions(
                        {0, max_caption_length - entry.target.caption().size(0)}).value(0)));

        all_captions.push_back(entry.target.reference_captions());
    }

    auto stacked_data = torch::data::transforms::Stack<>().apply_batch(image_caption_examples);

    auto caption_lengths = torch::from_blob(lengths.data(), lengths.size(), torch::kLong).clone();

    return {stacked_data.data, {stacked_data.target, caption_lengths, all_captions}};
}

GaussBlur2d::GaussBlur2d(int64_t kernel_size, double sigma) {
    auto sample_points = torch::arange(-(kernel_size - 1) / 2, (kernel_size - 1) / 2 + 1,
                                       torch::TensorOptions(torch::kFloat32).requires_grad(false))
            .unsqueeze_(1)
            .expand({-1, kernel_size});

    kernel = torch::exp(-(sample_points.square() + sample_points.transpose(0, 1).square())
                        / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);

    kernel /= kernel.sum();

    kernel.unsqueeze_(0).unsqueeze_(0);
}

torch::Tensor GaussBlur2d::operator()(torch::Tensor input) {
    if (input.ndimension() == 3) {
        input = input.unsqueeze(0);
    }

    const auto padding_left_right = (kernel.size(-1) - 1) / 2;
    const auto padding_top_bottom = (kernel.size(-2) - 1) / 2;

    auto padded_input = torch::reflection_pad2d(input,
                                                {padding_left_right, padding_left_right,
                                                 padding_top_bottom, padding_top_bottom});

    return torch::nn::functional::conv2d(padded_input,
                                         kernel.expand({input.size(-3), -1, -1, -1})
                                                 .to(input.device()),
                                         torch::nn::functional::Conv2dFuncOptions()
                                                 .groups(input.size(-3)));
}
}  // namespace transform
