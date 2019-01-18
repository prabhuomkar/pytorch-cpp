// Copyright 2019 Omkar Prabhu
#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <iostream>

int main() {
  std::cout << "PyTorch Basics" << std::endl;

  // ================================================================ //
  //                     BASIC AUTOGRAD EXAMPLE 1                     //
  // ================================================================ //

  std::cout << std::endl << "BASIC AUTOGRAD EXAMPLE 1: " << std::endl;

  // Create Tensors
  torch::Tensor x = torch::tensor(1.0, torch::requires_grad());
  torch::Tensor w = torch::tensor(2.0, torch::requires_grad());
  torch::Tensor b = torch::tensor(3.0, torch::requires_grad());

  // Build a computational graph
  auto y = w * x + b;  // y = 2 * x + 3

  // Compute the gradients
  y.backward();

  // Print out the gradients
  std::cout << x.grad() << std::endl;  // x.grad() = 2
  std::cout << w.grad() << std::endl;  // w.grad() = 1
  std::cout << b.grad() << std::endl;  // b.grad() = 1

  // ================================================================ //
  //                     BASIC AUTOGRAD EXAMPLE 2                     //
  // ================================================================ //

  std::cout << std::endl << "BASIC AUTOGRAD EXAMPLE 2: " << std::endl;

  // Create Tensors of shapes
  x = torch::randn({10, 3});
  y = torch::randn({10, 2});


  // Build a fully connected layer
  auto linear = torch::nn::Linear(3, 2);
  std::cout << "w: " << linear->weight << std::endl;
  std::cout << "b: " << linear->bias << std::endl;

  // Build loss function and optimizer
  auto criterion = torch::nn::MSELoss();
  auto optimizer = torch::optim::SGD(linear->parameters(), torch::optim::SGDOptions(0.01));

  // Forward pass
  auto pred = linear(x);

  // Compute loss
  auto loss = criterion(pred, y);
  std::cout << "loss item: " << loss.item().toFloat() << std::endl;

  // Backward pass
  loss.backward();

  // Print out the gradients
  std::cout << "dL/dw: " << linear->weight.grad() << std::endl;
  std::cout << "dL/db: " << linear->bias.grad() << std::endl;

  // 1 step gradient descent
  optimizer.step();

  // Print out the loss after 1-step gradient descent
  pred = linear(x);
  loss = criterion(pred, y);
  std::cout << "loss after 1 step optimization: " << loss.item().toFloat() << std::endl;

  // =============================================================== //
  //                          INPUT PIPELINE                         //
  // =============================================================== //

  // =============================================================== //
  //                 INPUT PIPELINE FOR CUSTOM DATASET               //
  // =============================================================== //

  // =============================================================== //
  //                        PRETRAINED MODEL                         //
  // =============================================================== //

  // =============================================================== //
  //                      SAVE AND LOAD THE MODEL                    //
  // =============================================================== //
}
