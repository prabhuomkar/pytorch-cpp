// Copyright 2019 Omkar Prabhu
#include <torch/torch.h>
#include <iostream>

// Hyper parameters
const int input_size = 784;
const int hidden_size = 500;
const int num_classes = 10;
const int num_epochs = 5;
const int batch_size = 100;
const double learning_rate = 0.001;

struct NeuralNet: torch::nn::Module {
  // Declare all the layers of nerual network
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};

  // Construct all the layers
  NeuralNet() {
    fc1 = register_module("fc1", torch::nn::Linear(input_size, hidden_size));
    fc2 = register_module("fc2", torch::nn::Linear(hidden_size, num_classes));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    x = fc2->forward(x);
    return torch::log_softmax(x, 1);
  }
};

int main() {
  std::cout << "FeedForward Neural Network" << std::endl;

  // Device
  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available. Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  // MNIST Dataset (images and labels)
  auto train_dataset = torch::data::datasets::MNIST("../data")
                           .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                           .map(torch::data::transforms::Stack<>());
  auto test_dataset = torch::data::datasets::MNIST("../data", torch::data::datasets::MNIST::Mode::kTest)
                           .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                           .map(torch::data::transforms::Stack<>());

  // Data loader (input pipeline)
  auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), batch_size);
  auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(test_dataset), batch_size);

  // Neural Network model
  auto model = std::make_shared<NeuralNet>();

  // Loss and optimizer
  auto optimizer = torch::optim::SGD(model->parameters(), torch::optim::SGDOptions(learning_rate));

  // Train the model
  for (int epoch = 0; epoch < num_epochs; epoch++) {
    int i = 0;
    for (auto& batch : *train_loader) {
      auto data = batch.data.to(device), labels = batch.target.to(device);

      // Forward pass
      auto outputs = model->forward(data);
      auto loss = torch::nll_loss(outputs, labels);

      // Backward and optimize
      optimizer.zero_grad();
      loss.backward();
      optimizer.step();

      if ((i+1) % 5 == 0) {
        std::cout << "Epoch [" << (epoch+1) << "/" << num_epochs << "], Batch: "
          << (i+1) << ", Loss: " << loss.item().toFloat() << std::endl;
      }
    }
  }

  // Test the model
  torch::NoGradGuard no_grad;
  int correct = 0;
  int total = 0;
  for (const auto& batch : *test_loader) {
    auto data = batch.data.to(device), labels = batch.target.to(device);
    auto outputs = model->forward(data);
    auto predicted = outputs.argmax(1);
    total += labels.size(0);
    correct += predicted.eq(labels).sum().template item<int>();
  }

  std::cout << "Accuracy of the model on the 10000 test images: " <<
    static_cast<double>(100 * correct / total) << std::endl;
}
