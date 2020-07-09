// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>
#include <vector>
#include <iostream>
#include "vocabulary.h"
#include "image_io.h"

template<typename TDataset>
using ValidationLoader =
std::unique_ptr<torch::data::StatelessDataLoader<TDataset, torch::data::samplers::SequentialSampler>>;

using LossFunction = std::function<torch::Tensor(const torch::Tensor &, const torch::Tensor &, const torch::Tensor &)>;

template<typename TEncoder, typename TDecoder, typename TDataset>
void validate(torch::nn::ModuleHolder<TEncoder> encoder, torch::nn::ModuleHolder<TDecoder> decoder,
              const ValidationLoader<TDataset> &data_loader, const LossFunction &criterion,
              const torch::Device &device) {
    const auto encoder_is_training = encoder->is_training();
    const auto decoder_is_training = decoder->is_training();

    std::cout << "Validating...\n";

    encoder->eval();
    decoder->eval();

    torch::NoGradGuard no_grad;

    double running_loss = 0;
    size_t num_correct = 0;
    size_t output_packed_length_sum = 0;

    const auto start_time = std::chrono::high_resolution_clock::now();

    for (auto &batch : *data_loader) {
        auto images = batch.data.to(device);
        auto input_captions = batch.target.captions.index({torch::indexing::Slice(),
                                                           torch::indexing::Slice(0, -1)}).to(device);
        auto caption_lengths = batch.target.caption_lengths.sub(1);

        auto target_captions = batch.target.captions.index({torch::indexing::Slice(),
                                                            torch::indexing::Slice(1, torch::indexing::None)});

        auto targets = torch::nn::utils::rnn::pack_padded_sequence(target_captions, caption_lengths,
                                                                   true).data().to(device);

        auto features = encoder->forward(images);

        torch::Tensor outputs, alphas;

        std::tie(outputs, alphas) = decoder->forward(features, input_captions, caption_lengths);

        auto packed_outputs = torch::nn::utils::rnn::pack_padded_sequence(outputs, caption_lengths, true).data();

        auto loss = criterion(packed_outputs, targets, alphas);

        running_loss += loss.template item<double>() * packed_outputs.size(0);
        num_correct += packed_outputs.argmax(1).eq(targets).sum().template item<int64_t>();
        output_packed_length_sum += packed_outputs.size(0);
    }

    const auto end_time = std::chrono::high_resolution_clock::now();
    const auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

    std::cout << "Validating finished!\n";

    auto mean_loss = running_loss / output_packed_length_sum;
    auto accuracy = static_cast<double>(num_correct) / output_packed_length_sum;

    std::cout << "Validation - Loss: " << mean_loss << ", Acc: " << accuracy
              << ", elapsed time: " << elapsed_seconds / 60 << " min. " << elapsed_seconds % 60 << " sec." << "\n";

    encoder->train(encoder_is_training);
    decoder->train(decoder_is_training);
}

template<typename TEncoder, typename TDecoder, typename TDataset>
void sample_validate(torch::nn::ModuleHolder<TEncoder> encoder, torch::nn::ModuleHolder<TDecoder> decoder,
                     const ValidationLoader<TDataset> &data_loader, const torch::Device &device, int64_t eos_value) {
    const auto encoder_is_training = encoder->is_training();
    const auto decoder_is_training = decoder->is_training();

    std::cout << "Sampling...\n";

    // Test the model
    encoder->eval();
    decoder->eval();

    torch::NoGradGuard no_grad;

    score::BleuScoreLogger caption_metrics_logger;

    const auto start_time = std::chrono::high_resolution_clock::now();

    for (auto &batch : *data_loader) {
        auto images = batch.data.to(device);

        auto features = encoder->forward(images);

        torch::Tensor sampled_predictions = std::get<0>(decoder->sample(features));

        auto predictions = data_utils::unbind_caption_batch(sampled_predictions.cpu(), eos_value);
        caption_metrics_logger.update(predictions, batch.target.reference_captions);
    }

    const auto end_time = std::chrono::high_resolution_clock::now();
    const auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

    std::cout << "Sampling finished!\n";

    std::cout << "Sampling - ";

    for (size_t n = 1; n <= caption_metrics_logger.max_n(); ++n) {
        std::cout << "Bleu(" << n << "): " << caption_metrics_logger.bleu(n) << ", ";
    }

    std::cout << "elapsed time: " << elapsed_seconds / 60 << " min. " << elapsed_seconds % 60 << " sec." << "\n";

    encoder->train(encoder_is_training);
    decoder->train(decoder_is_training);
}

template<typename TEncoder, typename TDecoder>
void predict_captions(torch::nn::ModuleHolder<TEncoder> encoder, torch::nn::ModuleHolder<TDecoder> decoder,
                      const std::string &image_root_folder,
                      const std::vector<std::string> &sample_image_names,
                      const std::vector<int64_t> &image_load_size,
                      const data_utils::CaptionData &caption_data,
                      const torch::Device &device) {
    const auto encoder_is_training = encoder->is_training();
    const auto decoder_is_training = decoder->is_training();

    encoder->eval();
    decoder->eval();

    torch::NoGradGuard no_grad;

    transform::GaussBlur2d blur(33, 10);

    for (const auto &filename : sample_image_names) {
        const auto image_path = image_root_folder + "/" + filename;
        auto image = image_io::load_image(image_path, image_load_size);

        auto image_input = torch::data::transforms::Normalize<>({0.485, 0.456, 0.406},
                                                                {0.229, 0.224, 0.225})(image).unsqueeze(0).to(device);

        auto features = encoder->forward(image_input);

        torch::Tensor image_caption;
        torch::Tensor alphas;

        std::tie(image_caption, alphas) = decoder->sample(features);

        image_caption.squeeze_(0);

        auto image_caption_string = data_utils::translate_index_tensor_to_string(caption_data.vocabulary,
                                                                                 image_caption);

        alphas.squeeze_(0);

        auto upscaled_alphas = alphas
                .repeat_interleave(image_input.size(-1) / alphas.size(-1), -1)
                .repeat_interleave(image_input.size(-2) / alphas.size(-2), -2);

        auto blurred_upscaled_alphas = blur(upscaled_alphas).squeeze(0);

        std::istringstream is(image_caption_string);
        std::istream_iterator<std::string> image_caption_it(is), eos;

        auto id = 0;
        for (auto it = image_caption_it; it != eos; ++it) {
            const auto mask = blurred_upscaled_alphas[id];
            mask.div_(mask.max());
            const auto display_filename = filename.substr(0, filename.size() - 4);

            image_io::save_image(0.5 * image.to(device) + 0.5 * mask,
                                    "output/" + display_filename + "_"
                                    + std::to_string(id + 1) + "_" + *it + ".png", 1, 0);
            ++id;
        }

        std::cout << filename << ":\n";
        std::cout << "\t\t" << image_caption_string << "\n";
        std::cout << "\treferences:\n";

        const auto references = caption_data.captions.at(filename);

        for (const auto &reference : references) {
            std::cout << "\t\t" << reference << "\n";
        }

        std::cout << "\n";
    }

    std::cout << "Attention visualizations written to \"output/\"\n";

    encoder->train(encoder_is_training);
    decoder->train(decoder_is_training);
}
