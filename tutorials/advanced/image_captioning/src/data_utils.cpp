// Copyright 2020-present pytorch-cpp Authors
#include "data_utils.h"
#include <string>
#include <fstream>
#include <vector>

using torch::indexing::Slice;
using torch::indexing::None;

namespace data_utils {
namespace {
std::vector<std::string> tokenize_line(std::istringstream &line_stream) {
    std::vector<std::string> result{"<start>"};

    std::string word;
    while (line_stream >> word) {
        std::transform(word.begin(), word.end(), word.begin(), tolower);
        word.erase(std::remove_if(word.begin(), word.end(), ispunct), word.end());

        if (word.size() > 1 && !std::any_of(word.begin(), word.end(), isdigit)) {
            result.push_back(word);
        }
    }

    result.emplace_back("<end>");

    return result;
}

torch::Tensor calculate_caption_lengths(const torch::Tensor &captions, int64_t eos_value) {
    auto max_size = captions.size(1);
    auto max_size_tensor = torch::from_blob(&max_size, 1, captions.options());

    return captions.eq(eos_value).cumsum(1).eq(0).sum(1).add(1).min(max_size_tensor);
}
}  // namespace

CaptionData load_caption_data(const std::string &file_path, size_t word_minimum_frequency) {
    if (std::ifstream input{file_path}) {
        CaptionData caption_data;

        std::string line;

        std::vector<std::string> words;
        std::unordered_map<std::string, size_t> word_to_count;

        while (std::getline(input, line)) {
            if (line.empty()) {
                continue;
            }

            std::istringstream line_stream(line);

            std::string image_file_name;

            if (line_stream >> image_file_name) {
                image_file_name.erase(image_file_name.size() - 2, std::string::npos);
            } else {
                continue;
            }

            std::vector<std::string> caption_tokens = tokenize_line(line_stream);

            std::string caption;

            for (decltype(caption_tokens.size()) i = 0; i != caption_tokens.size(); ++i) {
                words.push_back(caption_tokens[i]);
                ++word_to_count[caption_tokens[i]];
                caption.append(caption_tokens[i] + ((i != caption_tokens.size() - 1) ? " " : ""));
            }

            if (caption_tokens.size() > 2) {
                caption_data.captions[image_file_name].push_back(caption);
                caption_data.max_length = std::max(caption_data.max_length, caption_tokens.size());
                caption_data.num_samples += 1;
            }
        }

        for (const auto &word : words) {
            if (word_to_count[word] >= word_minimum_frequency) {
                caption_data.vocabulary.add_word(word);
            }
        }

        return caption_data;
    } else {
        throw std::runtime_error("Could not read input file " + file_path);
    }
}

std::string translate_index_tensor_to_string(const Vocabulary &vocabulary, torch::Tensor indices) {
    std::ostringstream output_stream;

    for (decltype(indices.size(0)) i = 0; i != indices.size(0); ++i) {
        const auto word = vocabulary.index_to_word(indices[i].item<int64_t>());

        if (i != 0) {
            output_stream << " ";
        }

        output_stream << word;

        if (word == "<end>") {
            break;
        }
    }

    return output_stream.str();
}

torch::Tensor translate_string_to_index_tensor(const Vocabulary &vocabulary, const std::string &sentence) {
    std::istringstream line_stream(sentence);
    std::istream_iterator<std::string> line_it(line_stream), eos;

    std::vector<int64_t> ids;
    std::transform(line_it, eos, std::back_inserter(ids),
                   [&vocabulary](const std::string &word) { return vocabulary.word_to_index(word); });

    return torch::from_blob(ids.data(), ids.size(), torch::kLong).clone();
}

std::vector<torch::Tensor> unbind_variable_lengths(const torch::Tensor &input, const torch::Tensor &lengths) {
    std::vector<torch::Tensor> output;
    output.reserve(input.size(0));

    for (decltype(lengths.size(0)) i = 0; i != lengths.size(0); ++i) {
        output.push_back(input.index({i, Slice(None, lengths[i].item<int64_t>())}));
    }

    return output;
}

std::vector<torch::Tensor> unbind_caption_batch(const torch::Tensor &input, int64_t eos_value) {
    return unbind_variable_lengths(input, calculate_caption_lengths(input, eos_value));
}
}  // namespace data_utils
