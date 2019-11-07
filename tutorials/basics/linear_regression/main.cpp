// Copyright 2019 Omkar Prabhu
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <iostream>
#include <vector>

int main() {
  std::cout << "Linear Regression" << std::endl;

  // Hyper parameters
  int input_size = 1;
  int output_size = 1;
  int num_epochs = 60;
  double learning_rate = 0.001;

  // Sample dataset
  auto x_train = torch::randint(0, 10, {15, 1});
  auto y_train = torch::randint(0, 10, {15, 1});

  // Linear regression model
  auto model = torch::nn::Linear(input_size, output_size);

  // Loss and optimizer
  auto criterion = torch::nn::L1Loss();
  auto optimizer = torch::optim::SGD(model->parameters(), torch::optim::SGDOptions(learning_rate));

  for (int epoch = 0; epoch < num_epochs; epoch++) {
    // Array to tensors
    auto inputs = x_train;
    auto targets = y_train;

    // Forward pass
    auto outputs = model(inputs);
    auto loss = criterion(outputs, targets);

    // Backward and optimize
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    if ((epoch+1) % 5 == 0) {
      std::cout << "Epoch [" << (epoch+1) << "/" << num_epochs << "], Loss: " << loss.item().toFloat() << std::endl;
    }
  } 

}
