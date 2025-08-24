// Copyright 2020-present pytorch-cpp Authors
// Original: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include "nnet.h"
#include "cifar10.h"

int main() {
    std::cout << "Deep Learning with PyTorch: A 60 Minute Blitz\n\n";
    std::cout << "Training a Classifier\n\n";

    // Loading and normalizing CIFAR10
    const std::string CIFAR_data_path = "../../../../../data/cifar10/";

    auto train_dataset = CIFAR10(CIFAR_data_path)
        .map(torch::data::transforms::Normalize<>({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
        .map(torch::data::transforms::Stack<>());
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), 4);

    auto test_dataset = CIFAR10(CIFAR_data_path, CIFAR10::Mode::kTest)
        .map(torch::data::transforms::Normalize<>({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
        .map(torch::data::transforms::Stack<>());
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), 4);

    std::string classes[10] = {"plane", "car", "bird", "cat",
           "deer", "dog", "frog", "horse", "ship", "truck"};

    // Define a Convolutional Neural Network
    Net net = Net();
    net->to(torch::kCPU);

    // // Define a Loss function and optimizer
    torch::nn::CrossEntropyLoss criterion;
    torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(0.001).momentum(0.9));

    // Train the network
    for (size_t epoch = 0; epoch < 2; ++epoch) {
        double running_loss = 0.0;

        int i = 0;
        for (auto& batch : *train_loader) {
            // get the inputs; data is a list of [inputs, labels]
            auto inputs = batch.data.to(torch::kCPU);
            auto labels = batch.target.to(torch::kCPU);

            // zero the parameter gradients
            optimizer.zero_grad();

            // forward + backward + optimize
            auto outputs = net->forward(inputs);
            auto loss = criterion(outputs, labels);
            loss.backward();
            optimizer.step();

            // print statistics
            running_loss += loss.item<double>();
            if (i % 2000 == 1999) {  // print every 2000 mini-batches
                std::cout << "[" << epoch + 1 << ", " << i + 1 << "] loss: "
                    << running_loss / 2000 << '\n';
                running_loss = 0.0;
            }
            i++;
        }
    }
    std::cout << "Finished Training\n\n";

    std::string PATH = "./cifar_net.pth";
    torch::save(net, PATH);

    // Test the network on the test data
    net = Net();
    torch::load(net, PATH);

    int correct = 0;
    int total = 0;
    for (const auto& batch : *test_loader) {
        auto images = batch.data.to(torch::kCPU);
        auto labels = batch.target.to(torch::kCPU);

        auto outputs = net->forward(images);

        auto out_tuple = torch::max(outputs, 1);
        auto predicted = std::get<1>(out_tuple);
        total += labels.size(0);
        correct += (predicted == labels).sum().item<int>();
    }

    std::cout << "Accuracy of the network on the 10000 test images: "
        << (100 * correct / total) << "%\n\n";

    float class_correct[10];
    float class_total[10];

    torch::NoGradGuard no_grad;

    for (const auto& batch : *test_loader) {
        auto images = batch.data.to(torch::kCPU);
        auto labels = batch.target.to(torch::kCPU);

        auto outputs = net->forward(images);

        auto out_tuple = torch::max(outputs, 1);
        auto predicted = std::get<1>(out_tuple);
        auto c = (predicted == labels).squeeze();

        for (int i = 0; i < 4; ++i) {
            auto label = labels[i].item<int>();
            class_correct[label] += c[i].item<float>();
            class_total[label] += 1;
        }
    }

    for (int i = 0; i < 10; ++i) {
        std::cout << "Accuracy of " << classes[i] << " "
            << 100 * class_correct[i] / class_total[i] << "%\n";
    }
}
