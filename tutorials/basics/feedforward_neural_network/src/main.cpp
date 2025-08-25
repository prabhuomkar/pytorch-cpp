// Copyright 2020-present pytorch-cpp Authors
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <string> // To parse command line arguments
#include "neural_net.h"

int main(int argc, char **argv)
{

    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);

    // Hyper parameters
    const int64_t input_size = 784;
    int64_t hidden_size = 500;
    const int64_t num_classes = 10;
    int64_t batch_size = 100;
    size_t num_epochs = 5;
    double learning_rate = 0.001;

    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--hidden_size" && i + 1 < argc)
        {
            hidden_size = std::atoi(argv[++i]);
        }
        if (std::string(argv[i]) == "--batch_size" && i + 1 < argc)
        {
            batch_size = std::atoi(argv[++i]);
        }
        if (std::string(argv[i]) == "--num_epochs" && i + 1 < argc)
        {
            num_epochs = std::atoi(argv[++i]);
        }
        if (std::string(argv[i]) == "--learning_rate" && i + 1 < argc)
        {
            learning_rate = std::atof(argv[++i]);
        }
        if (std::string(argv[i]) == "--help")
        {
            std::cout << "Example usage: \n";
            std::cout << "./<executable> --hidden_size 500  --batch_size 100 --num_epochs 10 --learning_rate 10e-4\n";
            std::cout << "If no arguements are passed default values will be used\n";
            std::cout << "Use the argument < --hidden_size H > to pass the number of hidden units\n";
            std::cout << "Use the argument < --batch_size B > to pass the batch size\n";
            std::cout << "Use the argument < --num_epochs E > to pass the number of epochs\n";
            std::cout << "Use the argument < --learning_rate LE >  to pass the learning rate \n";
            std::cout << "Use the argument < --help >  to print this mmessage \n";
            std::exit(-1);
        }
    }

    std::cout << "Input size: " << input_size << "\n";
    std::cout << "Number of hidden units: " << hidden_size << "\n";
    std::cout << "Number of classes: " << num_classes << "\n";
    std::cout << "batch size: " << batch_size << "\n";
    std::cout << "Number of epochs: " << num_epochs << "\n";
    std::cout << "Learning rate: " << learning_rate << "\n\n";

    std::cout << "FeedForward Neural Network\n\n";
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    const std::string MNIST_data_path = "../../../../data/mnist/";

    // MNIST Dataset
    auto train_dataset = torch::data::datasets::MNIST(MNIST_data_path)
                             .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                             .map(torch::data::transforms::Stack<>());

    // Number of samples in the training set
    auto num_train_samples = train_dataset.size().value();

    auto test_dataset = torch::data::datasets::MNIST(MNIST_data_path, torch::data::datasets::MNIST::Mode::kTest)
                            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                            .map(torch::data::transforms::Stack<>());

    // Number of samples in the testset
    auto num_test_samples = test_dataset.size().value();

    // Data loaders
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), batch_size);

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), batch_size);

    // Neural Network model
    NeuralNet model(input_size, hidden_size, num_classes);
    model->to(device);

    // Optimizer
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Training...\n";

    // Train the model
    for (size_t epoch = 0; epoch != num_epochs; ++epoch)
    {
        // Initialize running metrics
        double running_loss = 0.0;
        size_t num_correct = 0;

        for (auto &batch : *train_loader)
        {
            auto data = batch.data.view({batch_size, -1}).to(device);
            auto target = batch.target.to(device);

            // Forward pass
            auto output = model->forward(data);
            auto loss = torch::nn::functional::cross_entropy(output, target);

            // Update running loss
            running_loss += loss.item<double>() * data.size(0);

            // Calculate prediction
            auto prediction = output.argmax(1);

            // Update number of correctly classified samples
            num_correct += prediction.eq(target).sum().item<int64_t>();

            // Backward and optimize
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
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

    for (const auto &batch : *test_loader)
    {
        auto data = batch.data.view({batch_size, -1}).to(device);
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
