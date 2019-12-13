// Copyright 2019 Omkar Prabhu
#include <torch/torch.h>
#include <iostream>

int main() {
  std::cout << "Logistic Regression" << std::endl;

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

  // Hyper parameters
  int input_size = 784;
  int num_classes = 10;
  int num_epochs = 5;
  int batch_size = 100;
  double learning_rate = 0.001;

  const std::string MNIST_data_path = "../../../../data/mnist/";

  // MNIST Dataset (images and labels)
  auto train_dataset = torch::data::datasets::MNIST(MNIST_data_path)
                           .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                           .map(torch::data::transforms::Stack<>());
  auto test_dataset = torch::data::datasets::MNIST(MNIST_data_path, torch::data::datasets::MNIST::Mode::kTest)
                           .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                           .map(torch::data::transforms::Stack<>());

  // Data loader (input pipeline)
  auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), batch_size);
  auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(test_dataset), batch_size);

  // Logistic regression model
  auto model = torch::nn::Sequential(
      torch::nn::Linear(input_size, num_classes),
      torch::nn::Functional([] (const torch::Tensor& x) { return torch::log_softmax(x, 1); }));

  model->to(device);

  // Loss and optimizer
  auto optimizer = torch::optim::SGD(model->parameters(), torch::optim::SGDOptions(learning_rate));

  // Train the model
  for (int epoch = 0; epoch < num_epochs; epoch++) {
    int i = 0;
    for (auto& batch : *train_loader) {
      auto data = batch.data.reshape({batch_size, -1}).to(device);
      auto labels = batch.target.to(device);

      // Forward pass
      auto outputs = model->forward(data);
      auto loss = torch::nll_loss(outputs, labels);

      // Backward and optimize
      optimizer.zero_grad();
      loss.backward();
      optimizer.step();

      if ((i+1) % 100 == 0) {
        std::cout << "Epoch [" << (epoch+1) << "/" << num_epochs << "], Batch: "
          << (i+1) << ", Loss: " << loss.item().toFloat() << std::endl;
      }
      ++i;
    }
  }

  // Test the model
  model->eval();
  torch::NoGradGuard no_grad;

  int correct = 0;
  int total = 0;
  for (const auto& batch : *test_loader) {
    auto data = batch.data.reshape({batch_size, -1}).to(device);
    auto labels = batch.target.to(device);
    auto outputs = model->forward(data);
    auto predicted = outputs.argmax(1);
    total += labels.size(0);
    correct += predicted.eq(labels).sum().item<int>();
  }

  std::cout << "Accuracy of the model on the 10000 test images: " <<
    static_cast<double>(100 * correct / total) << std::endl;
}
