// Copyright 2020-present pytorch-cpp Authors
// Original: https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html
#include <torch/torch.h>
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "Deep Learning with PyTorch: A 60 Minute Blitz\n\n";
    std::cout << "What is PyTorch?\n\n";

    std::cout << "Tensors\n\n";

    // Construct a 5x3 matrix, uninitialized:
    auto x = torch::empty({5, 3});
    std::cout << "x:\n" << x << '\n';

    // Construct a randomly initialized matrix:
    x = torch::rand({5, 3});
    std::cout << "x:\n" << x << '\n';

    // Construct a matrix filled zeros and of dtype long:
    x = torch::zeros({5, 3}, torch::TensorOptions(torch::kLong));
    std::cout << "x:\n" << x << '\n';

    // Construct a tensor directly from data:
    x = torch::tensor({5.5, 3.0});
    std::cout << "x:\n" << x << '\n';

    // Construct a tensor based on an existing tensor:
    x = x.new_zeros({5, 3}, torch::TensorOptions(torch::kDouble));
    std::cout << "x:\n" << x << '\n';
    x = torch::rand_like(x, torch::TensorOptions(torch::kFloat));
    std::cout << "x:\n" << x << '\n';

    // Get its size:
    std::cout << "size: \n" << x.element_size() << '\n';

    std::cout << "Operations\n\n";

    // Addition: syntax 1
    auto y = torch::rand({5, 3});
    std::cout << "x+y:\n" << x+y << '\n';

    // Addition: syntax 2
    std::cout << "torch::add(x, y):\n" << torch::add(x, y) << '\n';

    // Addition: providing an output tensor as argument
    auto result = torch::empty({5, 3});
    torch::add_out(x, y, result);
    std::cout << "torch::add_out(x, y, result):\n" << result << '\n';

    // Addition: in-place
    std::cout << "y.add_(x):\n" << y.add_(x) << '\n';

    // You can use standard NumPy-like indexing with all bells and whistles!
    std::cout << "x[:, 1]:\n" << x.index({at::indexing::Slice(), 1});

    // Resizing: If you want to resize/reshape tensor, you can use torch::view:
    x = torch::randn({4, 4});
    y = x.view(16);
    auto z = x.view({-1, 8});  // the size -1 is inferred from other dimensions
    std::cout << x.element_size() << y.element_size() << z.element_size() << '\n';

    // If you have a one element tensor, use .item() to get the value as a Python number
    x = torch::randn(1);
    std::cout << "x:\n" << x << '\n';
    std::cout << "x.item():\n" << x.item<float>() << '\n';
}
