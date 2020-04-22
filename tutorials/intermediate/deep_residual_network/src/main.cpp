// Copyright 2020-present pytorch-cpp Authors
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include "resnet.h"
#include "cifar10.h"
#include "transform.h"

using resnet::ResNet;
using resnet::ResidualBlock;
using transform::ConstantPad;
using transform::RandomCrop;
using transform::RandomHorizontalFlip;

int main() {
    std::cout << "Deep Residual Network\n\n";

    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    // Hyper parameters
    const int64_t num_classes = 10;
    const int64_t batch_size = 100;
    const size_t num_epochs = 20;
    const double learning_rate = 0.001;
    const size_t learning_rate_decay_frequency = 8;  // number of epochs after which to decay the learning rate
    const double learning_rate_decay_factor = 1.0 / 3.0;

    const std::string CIFAR_data_path = "../../../../data/cifar10/";

    // CIFAR10 custom dataset
    auto train_dataset = CIFAR10(CIFAR_data_path)
        .map(ConstantPad(4))
        .map(RandomHorizontalFlip())
        .map(RandomCrop({32, 32}))
        .map(torch::data::transforms::Stack<>());

    // Number of samples in the training set
    auto num_train_samples = train_dataset.size().value();

    auto test_dataset = CIFAR10(CIFAR_data_path, CIFAR10::Mode::kTest)
        .map(torch::data::transforms::Stack<>());

    // Number of samples in the testset
    auto num_test_samples = test_dataset.size().value();

    // Data loader
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), batch_size);

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), batch_size);

    // Model
    std::array<int64_t, 3> layers{2, 2, 2};
    ResNet<ResidualBlock> model(layers, num_classes);
    model->to(device);

    // Optimizer
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    auto current_learning_rate = learning_rate;

    std::cout << "Training...\n";

    // Train the model
    for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
        // Initialize running metrics
        double running_loss = 0.0;
        size_t num_correct = 0;

        for (auto& batch : *train_loader) {
            // Transfer images and target labels to device
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);

            // Forward pass
            auto output = model->forward(data);

            // Calculate loss
            auto loss = torch::nn::functional::cross_entropy(output, target);

            // Update running loss
            running_loss += loss.item<double>() * data.size(0);

            // Calculate prediction
            auto prediction = output.argmax(1);

            // Update number of correctly classified samples
            num_correct += prediction.eq(target).sum().item<int64_t>();

            // Backward pass and optimize
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }

        // Decay learning rate
        if ((epoch + 1) % learning_rate_decay_frequency == 0) {
            current_learning_rate *= learning_rate_decay_factor;
            static_cast<torch::optim::AdamOptions&>(optimizer.param_groups().front()
                .options()).lr(current_learning_rate);
        }

        auto sample_mean_loss = running_loss / num_train_samples;
        auto accuracy = static_cast<double>(num_correct) / num_train_samples;

        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
            << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
    }

    std::cout << "Training finished!\n\n";
    std::cout << "Testing...\n";

    // Test the model
    model->eval();
    torch::NoGradGuard no_grad;

    double running_loss = 0.0;
    size_t num_correct = 0;

    for (const auto& batch : *test_loader) {
        auto data = batch.data.to(device);
        auto target = batch.target.to(device);

        auto output = model->forward(data);

        auto loss = torch::nn::functional::cross_entropy(output, target);
        running_loss += loss.item<double>() * data.size(0);

        auto prediction = output.argmax(1);
        num_correct += prediction.eq(target).sum().item<int64_t>();
    }

    std::cout << "Testing finished!\n";

    auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
    auto test_sample_mean_loss = running_loss / num_test_samples;

    std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
}
