// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace data_utils {
class Dictionary {
 public:
    int64_t add_word(const std::string& word);
    std::string word_at_index(int64_t index) const { return idx2word_[index]; }
    size_t size() const { return word2idx_.size(); }
 private:
    std::unordered_map<std::string, size_t> word2idx_;
    std::vector<std::string> idx2word_;
};
}  // namespace data_utils
