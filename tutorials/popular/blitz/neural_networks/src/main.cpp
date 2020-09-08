// Copyright 2020-present pytorch-cpp Authors
// Original: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include "nnet.h"

int main() {
    std::cout << "Deep Learning with PyTorch: A 60 Minute Blitz\n\n";
    std::cout << "Neural Networks\n\n";

    std::cout << "Define the network\n\n";
    Net net = Net();
    net->to(torch::kCPU);
    std::cout << net << "\n\n";

    // The learnable parameters of a model are returned by net.parameters():
    auto params = net->parameters();
    std::cout << params.size() << '\n';
    std::cout << params.at(0).sizes() << "\n\n";  // conv1's .weight

    // Let’s try a random 32x32 input:
    auto input = torch::randn({1, 1, 32, 32});
    auto out = net->forward(input);
    std::cout << out << "\n\n";

    // Zero the gradient buffers of all parameters and backprops with random gradients:
    net->zero_grad();
    out.backward(torch::randn({1, 10}));

    std::cout << "Loss Function\n\n";

    auto output = net->forward(input);
    auto target = torch::randn(10);  // a dummy target, for example
    target = target.view({1, -1});  // make it the same shape as output
    torch::nn::MSELoss criterion;
    auto loss = criterion(output, target);
    std::cout << loss << "\n\n";

    // For illustration, let us follow a few steps backward:
    std::cout << "loss.grad_fn:\n" << loss.grad_fn() << '\n';  // MSELoss

    std::cout << "Backprop\n\n";

    // Now we shall call loss.backward(), and have a look at conv1’s bias gradients before and after the backward:
    net->zero_grad();  // zeroes the gradient buffers of all parameters
    std::cout << "conv1.bias.grad before backward:\n" << net->conv1->bias.grad() << '\n';
    loss.backward();
    std::cout << "conv1.bias.grad after backward:\n" << net->conv1->bias.grad() << "\n\n";

    std::cout << "Update the weights\n\n";

    // create your optimizer
    auto learning_rate = 0.01;
    auto optimizer = torch::optim::SGD(net->parameters(), torch::optim::SGDOptions(learning_rate));
    // in your training loop:
    optimizer.zero_grad();   // zero the gradient buffers
    output = net->forward(input);
    loss = criterion(output, target);
    loss.backward();
    optimizer.step();    // Does the update
}
