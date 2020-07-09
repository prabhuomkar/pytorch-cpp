// Copyright 2020-present pytorch-cpp Authors
#include "vocabulary.h"
#include <algorithm>
#include <iostream>

namespace data_utils {
Vocabulary::Vocabulary() {
    add_word("<pad>");
    add_word("<start>");
    add_word("<end>");
    add_word("<unk>");
}

int64_t Vocabulary::add_word(const std::string &word) {
    auto it = word2idx_.find(word);

    if (it == word2idx_.end()) {
        idx2word_.push_back(word);

        auto new_index = idx2word_.size() - 1;
        word2idx_[word] = new_index;
        return new_index;
    }

    return it->second;
}

int64_t Vocabulary::word_to_index(const std::string &word) const {
    auto it = word2idx_.find(word);

    if (it == word2idx_.end()) {
        return word2idx_.at("<unk>");
    }

    return it->second;
}
}  // namespace data_utils
