// Copyright 2020-present pytorch-cpp Authors
#include "caption_dataset.h"
#include <torch/torch.h>
#include <vector>
#include <algorithm>
#include <random>
#include "vocabulary.h"
#include "image_io.h"

using torch::indexing::Slice;
using torch::indexing::None;
using torch::indexing::Ellipsis;

namespace dataset {
CaptionTarget::CaptionTarget(size_t caption_index, const std::vector<torch::Tensor> &captions)
        : caption_index_(caption_index), captions_(captions) {}

torch::Tensor CaptionTarget::caption() const {
    return captions_[caption_index_];
}

std::vector<torch::Tensor> CaptionTarget::reference_captions() const {
    std::vector<torch::Tensor> result;
    result.reserve(captions_.size());

    std::transform(captions_.cbegin(), captions_.cend(), std::back_inserter(result),
                   [](const auto &reference) { return reference.index({Slice(1, None), Ellipsis}); });

    return result;
}


ImageCaptionDataset::ImageCaptionDataset(const std::string &dataset_file, const std::string &image_directory,
                                         const data_utils::CaptionData &caption_data,
                                         torch::IntArrayRef image_load_size)
        : image_directory_(image_directory), image_load_size_(image_load_size.begin(), image_load_size.end()) {
    if (std::ifstream file{dataset_file}) {
        std::istream_iterator<std::string> file_iterator(file), eos;
        std::vector<std::string> image_filenames(file_iterator, eos);

        create_samples(caption_data, image_filenames);
    } else {
        throw std::runtime_error("Could not read file " + dataset_file);
    }
}

ImageCaptionSample ImageCaptionDataset::get(size_t index) {
    const auto sample_entry = samples_[index];

    CaptionTarget caption_target(sample_entry.caption_index, captions_[sample_entry.image_filename]);

    auto image = image_io::load_image(image_directory_ + "/" +
                                         sample_entry.image_filename, image_load_size_);

    return {image, caption_target};
}

torch::optional<size_t> ImageCaptionDataset::size() const {
    return samples_.size();
}

void ImageCaptionDataset::create_samples(const data_utils::CaptionData &caption_data,
                                         const std::vector<std::string> &image_filenames) {
    for (const auto &file_name : image_filenames) {
        const auto it = caption_data.captions.find(file_name);

        if (it != caption_data.captions.cend()) {
            const auto &caption_entries = it->second;

            std::vector<torch::Tensor> caption_code_sequences;
            caption_code_sequences.reserve(caption_entries.size());

            for (decltype(caption_entries.size()) i = 0; i != caption_entries.size(); ++i) {
                caption_code_sequences.push_back(
                        data_utils::translate_string_to_index_tensor(caption_data.vocabulary, caption_entries[i]));
                samples_.emplace_back(file_name, i);
            }

            captions_.insert({file_name, caption_code_sequences});
        } else {
            throw std::runtime_error("Filename " + file_name + " does not exist in caption data.");
        }
    }
}
}  // namespace dataset
