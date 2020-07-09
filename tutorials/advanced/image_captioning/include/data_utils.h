// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>
#include <vector>
#include "vocabulary.h"

namespace data_utils {
struct CaptionData {
    Vocabulary vocabulary;
    std::unordered_map<std::string, std::vector<std::string>> captions;
    size_t max_length = 0;
    size_t num_samples = 0;
};

/**
 * Builds caption data from a file containing the captions.
 * @param filePath The path to the file containing the captions.
 * @param word_minimum_frequency The minimum number of times a word must occur in the text for it to be
 *                               placed into the vocabulary.
 * @return The loaded caption data.
 */
CaptionData load_caption_data(const std::string &filePath, size_t word_minimum_frequency);

std::string translate_index_tensor_to_string(const Vocabulary &vocabulary, torch::Tensor indices);

torch::Tensor translate_string_to_index_tensor(const Vocabulary &vocabulary, const std::string &sentence);

std::vector<torch::Tensor> unbind_caption_batch(const torch::Tensor &input, int64_t eos_value);

std::vector<torch::Tensor> unbind_variable_lengths(const torch::Tensor &input, const torch::Tensor &lengths);
}  // namespace data_utils
