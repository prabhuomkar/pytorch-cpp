// Copyright 2020-present pytorch-cpp Authors
#include <imagefolder_dataset.h>
#include <torch/torch.h>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <unordered_map>
#include "image_io.h"

namespace fs = std::filesystem;

using image_io::load_image;

namespace dataset {
namespace {
std::vector<std::string> parse_classes(const std::string &directory) {
    std::vector<std::string> classes;

    for (auto &p : fs::directory_iterator(directory)) {
        if (p.is_directory()) {
            classes.push_back(p.path().filename().string());
        }
    }

    std::sort(classes.begin(), classes.end());

    return classes;
}

std::unordered_map<std::string, int> create_class_to_index_map(const std::vector<std::string> &classes) {
    std::unordered_map<std::string, int> class_to_index;

    int index = 0;

    for (const auto &class_name : classes) {
        class_to_index[class_name] = index++;
    }

    return class_to_index;
}

std::vector<std::pair<std::string, int>> create_samples(
    const std::string &directory,
    const std::unordered_map<std::string, int> &class_to_index) {
    std::vector<std::pair<std::string, int>> samples;

    for (const auto &[class_name, class_index] : class_to_index) {
        for (const auto &p : fs::directory_iterator(directory + "/" + class_name)) {
            if (p.is_regular_file()) {
                samples.emplace_back(p.path().string(), class_index);
            }
        }
    }

    return samples;
}
}  // namespace

ImageFolderDataset::ImageFolderDataset(const std::string &root, Mode mode, torch::IntArrayRef image_load_size)
    : mode_(mode),
      image_load_size_(image_load_size.begin(), image_load_size.end()),
      mode_dir_(root + "/" + (mode == Mode::TRAIN ? "train" : "val")),
      classes_(parse_classes(mode_dir_)),
      class_to_index_(create_class_to_index_map(classes_)),
      samples_(create_samples(mode_dir_, class_to_index_)) {}

torch::optional<size_t> ImageFolderDataset::size() const {
    return samples_.size();
}

torch::data::Example<> ImageFolderDataset::get(size_t index) {
    const auto &[image_path, class_index] = samples_[index];

    return {load_image(image_path, image_load_size_), torch::tensor(class_index)};
}
}  // namespace dataset
