// Copyright 2020-present pytorch-cpp Authors

#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <chrono>
#include "cxxopts.hpp"
#include "vocabulary.h"
#include "data_utils.h"
#include "caption_dataset.h"
#include "encoder_cnn.h"
#include "decoder_rnn.h"
#include "transform.h"
#include "score.h"
#include "scheduler.h"
#include "validate.h"

using data_utils::CaptionData;
using data_utils::load_caption_data;
using dataset::ImageCaptionDataset;
using dataset::ImageCaptionSample;
using torch::indexing::Slice;
using torch::indexing::None;
using torch::indexing::Ellipsis;

cxxopts::ParseResult parse_args(int argc, char **argv);

int main(int argc, char **argv) {
    auto results = parse_args(argc, argv);

    if (results.count("help")) {
        return 0;
    }

    std::cout << "Image Captioning\n\n";

    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << "\n";

    // Hyper parameters
    const auto batch_size = results["batch_size"].as<int64_t>();
    const auto num_epochs = results["num_epochs"].as<size_t>();
    const auto learning_rate = results["learning_rate"].as<double>();
    const auto min_word_frequency = results["min_word_frequency"].as<size_t>();
    const std::vector<int64_t> image_load_size = {256, 256};
    const std::vector<int64_t> image_crop_size = {224, 224};

    // Encoder output: [batch_size, encoder_out_size, encoder_out_wh, encoder_out_wh]
    const int64_t encoder_out_wh = results["encoder_out_wh"].as<int64_t>();
    const auto encoder_out_size = results["encoder_out_size"].as<int64_t>();

    const auto decoder_embedding_size = results["decoder_embedding_size"].as<int64_t>();
    const auto decoder_hidden_size = results["decoder_hidden_size"].as<int64_t>();
    const auto decoder_attention_size = results["decoder_attention_size"].as<int64_t>();
    const auto decoder_dropout = results["decoder_dropout"].as<double>();
    const auto teacher_forcing_p = results["teacher_forcing_p"].as<double>();
    const double ds_regularization_factor = results["ds_regularization_factor"].as<double>();

    const auto train_log_steps = results["train_log_steps"].as<size_t>();
    const auto lr_scheduler_stepsize = results["lr_scheduler_stepsize"].as<size_t>();
    const auto lr_scheduler_factor = results["lr_scheduler_factor"].as<double>();
    const auto validate_on_epoch_end = results["validate_on_epoch_end"].as<bool>();
    const auto sample_on_epoch_end = results["sample_on_epoch_end"].as<bool>();

    const auto num_sample_images = results["num_sample_images"].as<size_t>();

    // Data input paths
    const std::string flickr8k_captions_file_path = "../../../../data/flickr_8k/Flickr8k_text/Flickr8k.token.txt";
    const std::string flickr8k_image_directory_path = "../../../../data/flickr_8k/Flickr8k_Dataset/Flicker8k_Dataset";
    const std::string flickr8k_training_set_file_path =
            "../../../../data/flickr_8k/Flickr8k_text/Flickr_8k.trainImages.txt";

    const std::string flickr8k_valiadtion_set_file_path =
            "../../../../data/flickr_8k/Flickr8k_text/Flickr_8k.devImages.txt";

    // Path to prelearned encoder backbone scriptmodule file
    const std::string encoder_backbone_scriptmodule_file_path =
            "../../../../tutorials/advanced/image_captioning/model/encoder_cnn_backbone.pt";

    // Load captions from file and build caption data (vocabulary and filename -> captions map)
    auto caption_data = load_caption_data(flickr8k_captions_file_path, min_word_frequency);

    std::cout << "Vocabulary size: " << caption_data.vocabulary.size() << "\n";

    // Custom dataset to load flickr8k images and corresponding captions
    auto train_dataset = ImageCaptionDataset(flickr8k_training_set_file_path, flickr8k_image_directory_path,
                                             caption_data, image_load_size)
            .map(torch::data::transforms::Normalize<ImageCaptionSample::TargetType>({0.485, 0.456, 0.406},
                                                                                    {0.229, 0.224, 0.225}))
            .map(transform::RandomCrop<ImageCaptionSample::TargetType>(image_crop_size))
            .map(transform::RandomHorizontalFlip<ImageCaptionSample::TargetType>())
            .map(transform::ImageCaptionCollate());

    const auto num_train_samples = train_dataset.size().value();

    std::cout << "Training samples: " << num_train_samples << "\n";

    auto validation_dataset = ImageCaptionDataset(flickr8k_valiadtion_set_file_path, flickr8k_image_directory_path,
                                                  caption_data, image_crop_size)
            .map(torch::data::transforms::Normalize<ImageCaptionSample::TargetType>({0.485, 0.456, 0.406},
                                                                                    {0.229, 0.224, 0.225}))
            .map(transform::ImageCaptionCollate());

    const auto num_validation_samples = validation_dataset.size().value();

    std::cout << "Validation samples: " << num_validation_samples << "\n";

    // Data loaders
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset),
            torch::data::DataLoaderOptions(batch_size));

    auto validation_loader =
            torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(validation_dataset),
                    torch::data::DataLoaderOptions(batch_size));

    // Encoder
    EncoderCNN encoder(encoder_backbone_scriptmodule_file_path, encoder_out_wh, encoder_out_size);
    encoder->to(device);

    // Decoder
    DecoderAttentionRNN decoder(decoder_embedding_size, decoder_hidden_size, caption_data.vocabulary.size(),
                                caption_data.max_length, encoder_out_size, decoder_attention_size, decoder_dropout,
                                teacher_forcing_p);
    decoder->to(device);

    // Concatenate model parameters
    auto encoder_parameters = encoder->parameters();
    auto decoder_parameters = decoder->parameters();

    std::vector<torch::Tensor> parameters = encoder_parameters;
    parameters.insert(parameters.end(), decoder_parameters.begin(), decoder_parameters.end());

    // Loss criterion
    auto criterion = [ds_regularization_factor](const torch::Tensor &output, const torch::Tensor &target,
                                                const torch::Tensor &alphas) -> torch::Tensor {
        auto result = torch::nn::functional::cross_entropy(output, target);

        if (ds_regularization_factor > 0) {
            // Doubly stochastic regularization (https://arxiv.org/pdf/1502.03044.pdf section 4.2.1)
            result += ds_regularization_factor * (1.0 - alphas.sum(1)).pow(2).mean();
        }

        return result;
    };

    // Optimizer
    torch::optim::Adam optimizer(parameters, torch::optim::AdamOptions(learning_rate));

    // Learning rate scheduler
    scheduler::StepLR<decltype(optimizer)> scheduler(optimizer, lr_scheduler_stepsize, lr_scheduler_factor);

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Training...\n";

    // Train the model
    for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
        double running_loss = 0;
        size_t num_correct = 0;
        size_t output_packed_length_sum = 0;
        size_t batch_index = 0;

        const auto start_time = std::chrono::high_resolution_clock::now();

        for (auto &batch : *train_loader) {
            // Transfer images and encoded captions to device
            auto images = batch.data.to(device);
            auto input_captions = batch.target.captions.index({Slice(), Slice(0, -1)}).to(device);
            auto caption_lengths = batch.target.caption_lengths.sub(1);

            auto target_captions = batch.target.captions.index({Slice(), Slice(1, None)});

            auto packed_targets = torch::nn::utils::rnn::pack_padded_sequence(target_captions,
                                                                              caption_lengths, true).data().to(device);

            // Forward pass
            auto features = encoder->forward(images);

            torch::Tensor outputs, alphas;

            std::tie(outputs, alphas) = decoder->forward(features, input_captions, caption_lengths);

            auto packed_outputs = torch::nn::utils::rnn::pack_padded_sequence(outputs, caption_lengths, true).data();

            // Calculate loss
            auto loss = criterion(packed_outputs, packed_targets, alphas);

            // Backward pass and optimize
            decoder->zero_grad();
            encoder->zero_grad();
            loss.backward();
            optimizer.step();

            // Update running metrics
            running_loss += loss.item<double>() * packed_outputs.size(0);
            num_correct += packed_outputs.argmax(1).eq(packed_targets).sum().item<int64_t>();

            output_packed_length_sum += packed_outputs.size(0);

            if ((batch_index + 1) % train_log_steps == 0) {
                std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "], Step [" << batch_index + 1 << "/"
                          << num_train_samples / batch_size + (num_train_samples % batch_size != 0)
                          << "], Loss: " << loss.item<double>()
                          << ", Acc.: "
                          << packed_outputs.argmax(1).eq(packed_targets)
                                  .to(torch::kFloat32).mean().item<double>() << "\n";
            }

            ++batch_index;
        }

        const auto end_time = std::chrono::high_resolution_clock::now();
        const auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

        auto mean_loss = running_loss / output_packed_length_sum;
        auto accuracy = static_cast<double>(num_correct) / output_packed_length_sum;

        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs
                  << "], Trainset - Loss: " << mean_loss
                  << ", Acc.: " << accuracy
                  << ", elapsed time: " << elapsed_seconds / 60 << " min. " << elapsed_seconds % 60 << " sec." << "\n";

        if (validate_on_epoch_end) {
            validate(encoder, decoder, validation_loader, criterion, device);
        }

        if (sample_on_epoch_end) {
            sample_validate(encoder, decoder, validation_loader, device,
                            caption_data.vocabulary.word_to_index("<end>"));
        }

        scheduler.step();
    }

    std::cout << "Training finished!\n\n";
    std::cout << "Testing...\n";

    // Test data input path
    const std::string flickr8k_testing_set_file_path =
            "../../../../data/flickr_8k/Flickr8k_text/Flickr_8k.testImages.txt";

    // Load test data
    auto test_dataset = ImageCaptionDataset(flickr8k_testing_set_file_path, flickr8k_image_directory_path,
                                            caption_data, image_crop_size)
            .map(torch::data::transforms::Normalize<ImageCaptionSample::TargetType>({0.485, 0.456, 0.406},
                                                                                    {0.229, 0.224, 0.225}))
            .map(transform::ImageCaptionCollate());

    auto num_test_samples = test_dataset.size().value();

    std::cout << "Testing samples: " << num_test_samples << std::endl;

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_dataset),
            torch::data::DataLoaderOptions(batch_size));

    validate(encoder, decoder, test_loader, criterion, device);
    sample_validate(encoder, decoder, test_loader, device, caption_data.vocabulary.word_to_index("<end>"));

    // Show results for some test images:
    std::ifstream test_image_names_stream(flickr8k_testing_set_file_path);
    std::istream_iterator<std::string> test_image_names_it(test_image_names_stream), eos;

    std::vector<std::string> test_image_names(test_image_names_it, eos);
    std::shuffle(test_image_names.begin(), test_image_names.end(),
                 std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count()));

    std::vector<std::string> test_image_samples(test_image_names.begin(),
                                                std::next(test_image_names.begin(), num_sample_images));

    predict_captions(encoder, decoder, flickr8k_image_directory_path,
                     test_image_samples, image_crop_size, caption_data, device);
}

cxxopts::ParseResult parse_args(int argc, char **argv) {
    cxxopts::Options options("image-captioning", "Predicts image captions");

    options.add_options()
            ("batch_size", "The batch size",
             cxxopts::value<int64_t>()->default_value("32"))
            ("num_epochs", "The number of epochs",
             cxxopts::value<size_t>()->default_value("4"))
            ("learning_rate", "The learning rate",
             cxxopts::value<double>()->default_value("1e-3"))
            ("min_word_frequency", "The minimum number of times a word must appear in the corpus "
                                   "for it to be put into the vocabulary",
             cxxopts::value<size_t>()->default_value("3"))
            ("decoder_dropout", "Dropout probability to use after embedding layer in the decoder",
             cxxopts::value<double>()->default_value("0.5"))
            ("teacher_forcing_p", "Probability of using teacher forcing while training",
             cxxopts::value<double>()->default_value("1.0"))
            ("encoder_out_size", "Number of channels of encoder output",
             cxxopts::value<int64_t>()->default_value("512"))
            ("decoder_embedding_size", "The size of the embedding layer of the decoder",
             cxxopts::value<int64_t>()->default_value("256"))
            ("decoder_hidden_size", "The size of the hidden states of the decoder",
             cxxopts::value<int64_t>()->default_value("512"))
            ("decoder_attention_size", "The size of the attention layer of the decoder",
             cxxopts::value<int64_t>()->default_value("512"))
            ("ds_regularization_factor", "The doubly-stochastic regularization factor",
             cxxopts::value<double>()->default_value("1.0"))
            ("lr_scheduler_stepsize", "The stepsize of the learning rate scheduler",
             cxxopts::value<size_t>()->default_value("2"))
            ("lr_scheduler_factor", "The factor to be used in the learning rate scheduler",
             cxxopts::value<double>()->default_value("0.5"))
            ("encoder_out_wh", "Width (= height) of the encoder output",
             cxxopts::value<int64_t>()->default_value("7"))
            ("train_log_steps", "The number of steps after which to display the training metrics",
             cxxopts::value<size_t>()->default_value("5"))
            ("validate_on_epoch_end", "Whether to perform validation after each epoch",
             cxxopts::value<bool>()->default_value("true"))
            ("sample_on_epoch_end", "Whether to perform sampling after each epoch",
             cxxopts::value<bool>()->default_value("true"))
            ("num_sample_images", "The number of sample images (randomly drawn from the test set) for which"
                                  " to display predictions and generate attention visualizations",
             cxxopts::value<size_t>()->default_value("10"))
            ("h,help", "Print usage");

    auto results = options.parse(argc, argv);

    if (results.count("help")) {
        std::cout << options.help() << std::endl;
    }

    return results;
}
