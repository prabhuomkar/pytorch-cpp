// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <string>
#include <vector>
#include <unordered_map>

namespace dataset {
/**
 * Dataset class that provides image-label samples.
 */
class ImageFolderDataset : public torch::data::datasets::Dataset<ImageFolderDataset> {
 public:
    enum class Mode {
       TRAIN,
       VAL
    };

    explicit ImageFolderDataset(const std::string &root, Mode mode = Mode::TRAIN,
                                torch::IntArrayRef image_load_size = {});

    torch::data::Example<> get(size_t index) override;

    torch::optional<size_t> size() const override;

 private:
    Mode mode_;
    std::vector<int64_t> image_load_size_;
    std::string mode_dir_;
    std::vector<std::string> classes_;
    std::unordered_map<std::string, int> class_to_index_;
    std::vector<std::pair<std::string, int>> samples_;
};
}  // namespace dataset
