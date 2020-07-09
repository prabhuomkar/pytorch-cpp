// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace data_utils {
class Vocabulary {
 public:
    Vocabulary();

    int64_t add_word(const std::string &word);

    std::string index_to_word(int64_t index) const { return idx2word_[index]; }

    int64_t word_to_index(const std::string &word) const;

    size_t size() const { return word2idx_.size(); }

 private:
    std::unordered_map<std::string, size_t> word2idx_;
    std::vector<std::string> idx2word_;
};
}  // namespace data_utils
