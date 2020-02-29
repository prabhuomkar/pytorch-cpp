// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>
#include <string>
#include "dictionary.h"

namespace data_utils {
class Corpus {
 public:
    explicit Corpus(const std::string& path) : path_(path) {}
    torch::Tensor get_data(int64_t batch_size);
    const Dictionary& get_dictionary() const { return dictionary_; }
 private:
    std::string path_;
    Dictionary dictionary_;
};
}  // namespace data_utils

