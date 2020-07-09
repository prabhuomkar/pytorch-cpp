// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>
#include <random>
#include <vector>
#include "caption_dataset.h"

namespace transform {

double rand_double();

int64_t rand_int(int64_t max);

struct CaptionBatchTarget {
    torch::Tensor captions;
    torch::Tensor caption_lengths;
    std::vector<std::vector<torch::Tensor>> reference_captions;
};

using ImageCaptionExample = torch::data::Example<torch::Tensor, CaptionBatchTarget>;

class ImageCaptionCollate : public torch::data::transforms::Collation<ImageCaptionExample,
        std::vector<dataset::ImageCaptionSample>> {
 public:
    ImageCaptionExample apply_batch(std::vector<dataset::ImageCaptionSample> data) override;
};

template<typename Target>
class RandomHorizontalFlip : public torch::data::transforms::TensorTransform<Target> {
 public:
    // Creates a transformation that randomly horizontally flips a tensor.
    //
    // The parameter `p` determines the probability that a tensor is flipped (default = 0.5).
    explicit RandomHorizontalFlip(double p = 0.5) : p_(p) {}

    torch::Tensor operator()(torch::Tensor input) override {
        if (rand_double() < p_) {
            return input.flip(-1);
        }

        return input;
    }

 private:
    double p_;
};

template<typename Target>
class RandomCrop : public torch::data::transforms::TensorTransform<Target> {
 public:
    // Creates a transformation that randomly crops a tensor.
    //
    // The parameter `size` is expected to be a vector of size 2
    // and determines the output size {height, width}.
    explicit RandomCrop(const std::vector<int64_t> &size) : size_(size) {}

    torch::Tensor operator()(torch::Tensor input) override {
        auto height_offset_length = input.size(-2) - size_[0];
        auto width_offset_length = input.size(-1) - size_[1];

        auto height_offset = rand_int(height_offset_length);
        auto width_offset = rand_int(width_offset_length);

        return input.index({torch::indexing::Ellipsis,
                            torch::indexing::Slice(height_offset, height_offset + size_[0]),
                            torch::indexing::Slice(width_offset, width_offset + size_[1])});
    }

 private:
    std::vector<int64_t> size_;
};

class GaussBlur2d : public torch::data::transforms::TensorTransform<torch::Tensor> {
 public:
    explicit GaussBlur2d(int64_t kernel_size, double sigma = 1.0);

    torch::Tensor operator()(torch::Tensor input) override;

 private:
    torch::Tensor kernel;
};
}  // namespace transform
