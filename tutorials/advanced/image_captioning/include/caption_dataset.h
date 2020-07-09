// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <cstddef>
#include <fstream>
#include <string>
#include <vector>
#include "vocabulary.h"
#include "data_utils.h"

namespace dataset {
/**
 * Target data for an image-caption sample. Holds reference captions and index of
 * the caption that this caption target represents.
 */
class CaptionTarget {
 public:
    CaptionTarget(size_t caption_index, const std::vector<torch::Tensor> &captions);

    /**
     * Returns the caption that this object represents.
     * @return A tensor containing the encoded caption (including start and end tokens)
     */
    torch::Tensor caption() const;

    /**
     * Returns the reference captions.
     * @return A vector of encoded caption tensors (including the end token, but excluding the start token)
     */
    std::vector<torch::Tensor> reference_captions() const;

 private:
    size_t caption_index_;
    std::vector<torch::Tensor> captions_;
};

using ImageCaptionSample = torch::data::Example<torch::Tensor, CaptionTarget>;

/**
 * Dataset class that provides image-caption samples.
 */
class ImageCaptionDataset : public torch::data::datasets::Dataset<ImageCaptionDataset, ImageCaptionSample> {
 public:
    /**
     * Constructs new dataset that provides image-caption samples.
     * @param dataset_file File containing the names of images belonging to this dataset.
     * @param image_directory The directory of the images.
     * @param caption_data CaptionData object holding the vocabulary and parsed captions.
     * @param image_load_size The size to use when loading the image files from disk.
     */
    ImageCaptionDataset(const std::string &dataset_file, const std::string &image_directory,
                        const data_utils::CaptionData &caption_data, torch::IntArrayRef image_load_size = {});

    ImageCaptionSample get(size_t index) override;

    torch::optional<size_t> size() const override;

 private:
    struct Sample {
        Sample(const std::string &filename, size_t index)
                : image_filename(filename), caption_index(index) {}

        std::string image_filename;
        size_t caption_index;
    };

    std::vector<Sample> samples_;
    std::unordered_map<std::string, std::vector<torch::Tensor>> captions_;
    std::string image_directory_;
    std::vector<int64_t> image_load_size_;

    void create_samples(const data_utils::CaptionData &caption_data, const std::vector<std::string> &file_names);
};
}  // namespace dataset
