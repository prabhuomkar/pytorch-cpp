// Copyright 2020-present pytorch-cpp Authors
// Original: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
#include <torch/torch.h>
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "Deep Learning with PyTorch: A 60 Minute Blitz\n\n";
    std::cout << "Autograd: Automatic Differentiation\n\n";

    std::cout << "Tensor\n\n";

    // Create a tensor and set requires_grad=True to track computation with it:
    auto x = torch::ones({2, 2}, torch::TensorOptions().requires_grad(true));
    std::cout << "x:\n" << x << '\n';

    // Do a tensor operation:
    auto y = x + 2;
    std::cout << "y:\n" << y << '\n';

    // y was created as a result of an operation, so it has a grad_fn:
    std::cout << "y.grad_fn:\n" << y.grad_fn() << '\n';

    // Do more operations on y:
    auto z = y * y * 3;
    auto out = z.mean();
    std::cout << "z:\n" << z << "out:\n" << out << '\n';

    // .requires_grad_(...) changes an existing Tensor’s requires_grad flag in-place:
    auto a = torch::randn({2, 2});
    a = ((a * 3) / (a - 1));
    std::cout << a.requires_grad() << '\n';
    a.requires_grad_(true);
    std::cout << a.requires_grad() << '\n';
    auto b = (a * a).sum();
    std::cout << b.grad_fn() << '\n';

    std::cout << "Gradients\n\n";

    // Let’s backprop now:
    out.backward();

    // Print gradients d(out)/dx:
    std::cout << "x.grad:\n" << x.grad() << '\n';

    // Example of vector-Jacobian product:
    x = torch::randn(3, torch::TensorOptions().requires_grad(true));
    y = x * 2;
    while (y.data().norm().item<int>() < 1000) {
        y = y * 2;
    }
    std::cout << "y:\n" << y << '\n';

    // Simply pass the vector to backward as argument:
    auto v = torch::tensor({0.1, 1.0, 0.0001}, torch::TensorOptions(torch::kFloat));
    y.backward(v);
    std::cout << "x.grad:\n" << x.grad() << '\n';

    // Stop autograd from tracking history on Tensors with .requires_grad=True:
    std::cout << "x.requires_grad\n" << x.requires_grad() << '\n';
    std::cout << "(x ** 2).requires_grad\n" << (x * x).requires_grad() << '\n';
    torch::NoGradGuard no_grad;
    std::cout << "(x ** 2).requires_grad\n" << (x * x).requires_grad() << '\n';

    // Or by using .detach() to get a new Tensor with the same content but that does not require gradients:
    std::cout << "x.requires_grad:\n" << x.requires_grad() << '\n';
    y = x.detach();
    std::cout << "y.requires_grad:\n" << y.requires_grad() << '\n';
    std::cout << "x.eq(y).all():\n" << x.eq(y).all() << '\n';
}
