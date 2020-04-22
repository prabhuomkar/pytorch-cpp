// Copyright 2020-present pytorch-cpp Authors
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <tuple>
#include "rnn_lm.h"
#include "corpus.h"

using data_utils::Corpus;
using torch::indexing::Slice;

int main() {
    std::cout << "Language Model\n\n";

    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    // Hyper parameters
    const int64_t embed_size = 128;
    const int64_t hidden_size = 1024;
    const int64_t num_layers = 1;
    const int64_t num_samples = 1000;  // the number of words to be sampled
    const int64_t batch_size = 20;
    const int64_t sequence_length = 30;
    const size_t num_epochs = 5;
    const double learning_rate = 0.002;

    // Load "Penn Treebank" dataset
    // See https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/language_model/data/
    // and https://github.com/wojzaremba/lstm/tree/master/data
    const std::string penn_treebank_data_path = "../../../../data/penntreebank/train.txt";

    Corpus corpus(penn_treebank_data_path);

    auto ids = corpus.get_data(batch_size);
    auto vocab_size = corpus.get_dictionary().size();

    // Path to the output file (All folders must exist!)
    const std::string sample_output_path = "output/sample.txt";

    // Model
    RNNLM model(vocab_size, embed_size, hidden_size, num_layers);
    model->to(device);

    // Optimizer
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Training...\n";

    // Train the model
    for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
        // Initialize running metrics
        double running_loss = 0.0;
        double running_perplexity = 0.0;
        size_t running_num_samples = 0;

        // Initialize hidden- and cell-states.
        auto h = torch::zeros({num_layers, batch_size, hidden_size}).to(device).detach();
        auto c = torch::zeros({num_layers, batch_size, hidden_size}).to(device).detach();

        for (int64_t i = 0; i < ids.size(1) - sequence_length; i += sequence_length) {
            // Transfer data and target labels to device
            auto data = ids.index({Slice(), Slice(i, i + sequence_length)}).to(device);
            auto target = ids.index({Slice(), Slice(i + 1, i + 1 + sequence_length)}).reshape(-1).to(device);

            // Forward pass
            torch::Tensor output;
            std::forward_as_tuple(output, std::tie(h, c)) = model->forward(data, std::make_tuple(h, c));

            h.detach_();
            c.detach_();

            // Calculate loss
            auto loss = torch::nn::functional::nll_loss(output, target);

            // Update running metrics
            running_loss += loss.item<double>() * data.size(0);
            running_perplexity += torch::exp(loss).item<double>() * data.size(0);
            running_num_samples += data.size(0);

            // Backward pass and optimize
            optimizer.zero_grad();
            loss.backward();
            torch::nn::utils::clip_grad_norm_(model->parameters(), 0.5);
            optimizer.step();
        }

        auto sample_mean_loss = running_loss / running_num_samples;
        auto sample_mean_perplexity = running_perplexity / running_num_samples;

        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
            << sample_mean_loss << ", Perplexity: " << sample_mean_perplexity << '\n';
    }

    std::cout << "Training finished!\n\n";
    std::cout << "Generating samples...\n";

    // Generate samples
    model->eval();
    torch::NoGradGuard no_grad;

    std::ofstream sample_output_file(sample_output_path);

    // Initialize hidden- and cell-states.
    auto h = torch::zeros({num_layers, 1, hidden_size}).to(device);
    auto c = torch::zeros({num_layers, 1, hidden_size}).to(device);

    // Select one word-id at random
    auto prob = torch::ones(vocab_size);
    auto data = prob.multinomial(1).unsqueeze(1).to(device);

    for (size_t i = 0; i != num_samples; ++i) {
        // Forward pass
        torch::Tensor output;
        std::forward_as_tuple(output, std::tie(h, c)) = model->forward(data, std::make_tuple(h, c));

        // Sample one word id
        prob = output.exp();
        auto word_id = prob.multinomial(1).item();

        // Fill input data with sampled word id for the next time step
        data.fill_(word_id);

        // Write the word corresponding to the id to the file
        auto word = corpus.get_dictionary().word_at_index(word_id.toLong());
        word = (word == "<eos>") ? "\n" : word + " ";
        sample_output_file << word;
    }
    std::cout << "Finished generating samples!\nSaved output to " << sample_output_path << "\n";
}

