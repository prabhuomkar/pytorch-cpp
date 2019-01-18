// Copyright 2019 Omkar Prabhu
#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <iostream>

int main() {
  std::cout << "PyTorch Basics" << std::endl;

  ///////////////////////////////////////////////////////////
  //                BASIC AUTOGRAD EXAMPLE 1               //
  ///////////////////////////////////////////////////////////

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
}
