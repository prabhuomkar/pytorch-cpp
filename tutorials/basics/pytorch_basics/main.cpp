// Copyright 2019 Omkar Prabhu
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <iomanip>

void print_tensor_size(const torch::Tensor&);

int main() {
    std::cout << "PyTorch Basics\n\n";

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    // ================================================================ //
    //                     BASIC AUTOGRAD EXAMPLE 1                     //
    // ================================================================ //

    std::cout << "---- BASIC AUTOGRAD EXAMPLE 1 ----\n";

    // Create Tensors
    torch::Tensor x = torch::tensor(1.0, torch::requires_grad());
    torch::Tensor w = torch::tensor(2.0, torch::requires_grad());
    torch::Tensor b = torch::tensor(3.0, torch::requires_grad());

    // Build a computational graph
    auto y = w * x + b;  // y = 2 * x + 3

    // Compute the gradients
    y.backward();

    // Print out the gradients
    std::cout << x.grad() << '\n';  // x.grad() = 2
    std::cout << w.grad() << '\n';  // w.grad() = 1
    std::cout << b.grad() << "\n\n";  // b.grad() = 1

    // ================================================================ //
    //                     BASIC AUTOGRAD EXAMPLE 2                     //
    // ================================================================ //

    std::cout << "---- BASIC AUTOGRAD EXAMPLE 2 ----\n";

    // Create Tensors of shapes
    x = torch::randn({10, 3});
    y = torch::randn({10, 2});

    // Build a fully connected layer
    auto linear = torch::nn::Linear(3, 2);
    std::cout << "w:\n" << linear->weight << '\n';
    std::cout << "b:\n" << linear->bias << '\n';

    // Build loss function and optimizer
    auto criterion = [] (const torch::Tensor& pred, const torch::Tensor& target) {
        return torch::mse_loss(pred, target);
    };
    auto optimizer = torch::optim::SGD(linear->parameters(), torch::optim::SGDOptions(0.01));

    // Forward pass
    auto pred = linear(x);

    // Compute loss
    auto loss = criterion(pred, y);
    std::cout << "Loss: " << loss.item<double>() << '\n';

    // Backward pass
    loss.backward();

    // Print out the gradients
    std::cout << "dL/dw:\n" << linear->weight.grad() << '\n';
    std::cout << "dL/db:\n" << linear->bias.grad() << '\n';

    // 1 step gradient descent
    optimizer.step();

    // Print out the loss after 1-step gradient descent
    pred = linear(x);
    loss = criterion(pred, y);
    std::cout << "Loss after 1 optimization step: " << loss.item<double>() << "\n\n";

    // =============================================================== //
    //               CREATING TENSORS FROM EXISTING DATA               //
    // =============================================================== //

    std::cout << "---- CREATING TENSORS FROM EXISTING DATA ----\n";

    // WARNING: Tensors created with torch::from_blob(ptr_to_data, ...) do not own
    // the memory pointed to by ptr_to_data!
    // (see https://pytorch.org/cppdocs/notes/tensor_basics.html#using-externally-created-data)
    //
    // If you want a tensor that has its own copy of the data you can call clone() on the
    // tensor returned from torch::from_blob(), e.g.:
    // torch::Tensor t = torch::from_blob(ptr_to_data, ...).clone();
    // (see https://github.com/pytorch/pytorch/issues/12506#issuecomment-429573396)

    // Tensor From C-style array
    float data_array[] = {1, 2, 3, 4};
    torch::Tensor t1 = torch::from_blob(data_array, {2, 2});
    std::cout << "Tensor from array:\n" << t1 << '\n';

    TORCH_CHECK(data_array == t1.data_ptr<float>());

    // Tensor from vector:
    std::vector<float> data_vector = {1, 2, 3, 4};
    torch::Tensor t2 = torch::from_blob(data_vector.data(), {2, 2});
    std::cout << "Tensor from vector:\n" << t2 << "\n\n";

    TORCH_CHECK(data_vector.data() == t2.data_ptr<float>());

    // =============================================================== //
    //                         INPUT PIPELINE                          //
    // =============================================================== //

    std::cout << "---- INPUT PIPELINE ----\n";

    // Construct MNIST dataset
    const std::string MNIST_data_path = "../../../../data/mnist/";

    auto dataset = torch::data::datasets::MNIST(MNIST_data_path)
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());

    // Fetch one data pair
    auto example = dataset.get_batch(0);
    std::cout << "Sample data size: ";
    print_tensor_size(example.data);
    std::cout << "\n";
    std::cout << "Sample target: " << example.target.item<int>() << "\n";

    // Construct data loader
    auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        dataset, 64);

    // Fetch a mini-batch
    auto example_batch = *dataloader->begin();
    std::cout << "Sample batch - data size: ";
    print_tensor_size(example_batch.data);
    std::cout << "\n";
    std::cout << "Sample batch - target size: ";
    print_tensor_size(example_batch.target);
    std::cout << "\n\n";

    // Actual usage of the dataloader:
    for (auto& batch : *dataloader) {
        // Training code here
    }

    // =============================================================== //
    //               INPUT PIPELINE FOR CUSTOM DATASET                 //
    // =============================================================== //

    // See Deep Residual Network tutorial files cifar10.h and cifar10.cpp
    // for an example of a custom dataset implementation.

    // =============================================================== //
    //                        PRETRAINED MODEL                         //
    // =============================================================== //

    std::cout << "---- PRETRAINED MODEL ----\n";
    // Currently, loading a pretrained model using the C++ API must be done
    // in the following way:
    // (1) In Python: Create the (pretrained) pytorch model.
    // (2) In Python: Convert the pytorch model to a torch.jit.ScriptModule (via tracing or by using annotations)
    // (3) In Python: Serialize the scriptmodule to a file.
    // (4) In C++: Load the scriptmodule form the file using torch::jit::load()
    // See https://pytorch.org/tutorials/advanced/cpp_export.html for more infos.

    // Path to serialized ScriptModule of pretrained resnet18 model,
    // created in Python.
    // You can use the provided Python-script "create_resnet18_scriptmodule.py" in
    // tutorials/basics/pytorch-basics/models to create the necessary file.
    const std::string pretrained_model_path = "../../../../tutorials/basics/pytorch_basics/models/"
        "resnet18_scriptmodule.pt";

    torch::jit::script::Module resnet;

    try {
        resnet = torch::jit::load(pretrained_model_path);
    }
    catch (const torch::Error& error) {
        std::cerr << "Could not load scriptmodule from file " << pretrained_model_path << ".\n"
            << "You can create this file using the provided Python script 'create_resnet18_scriptmodule.py' "
            "in tutorials/basics/pytorch-basics/models/.\n";
        return -1;
    }

    std::cout << "Resnet18 model:\n";

    for (const auto& module : resnet.get_modules()) {
        std::cout << "  " << module.name().name() << "\n";

        for (const auto& sub_module : module.get_modules()) {
            std::cout << "    " << sub_module.name().name() << "\n";
        }
    }

    auto in_features = resnet.find_module("fc")->get_parameter("weight").size(1);
    auto out_features = resnet.find_module("fc")->get_parameter("weight").size(0);

    std::cout << "Fully connected layer: in_features=" << in_features << ", out_features=" << out_features << "\n";

    // Input sample
    auto sample_input = torch::randn({1, 3, 224, 224});
    std::vector<torch::jit::IValue> inputs{sample_input};

    // Forward pass
    std::cout << "Input size: ";
    print_tensor_size(sample_input);
    std::cout << "\n";
    auto output = resnet.forward(inputs).toTensor();
    std::cout << "Output size: ";
    print_tensor_size(output);
    std::cout << "\n\n";

    // =============================================================== //
    //                      SAVE AND LOAD A MODEL                      //
    // =============================================================== //

    std::cout << "---- SAVE AND LOAD A MODEL ----\n";

    // Simple example model
    torch::nn::Sequential model{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 3).stride(2).padding(1)),
        torch::nn::Functional(torch::relu)
    };

    std::cout << "Model:\n" << model << "\n";

    // Path to the model output file (all folders must exist!).
    const std::string model_save_path = "../../../../tutorials/basics/pytorch_basics/models/model.pt";

    // Save the model
    torch::save(model, model_save_path);

    // Load the model
    torch::load(model, model_save_path);
}

void print_tensor_size(const torch::Tensor& x) {
    std::cout << "[";
    for (size_t i = 0; i != x.dim() - 1; ++i) {
        std::cout << x.size(i) << " ";
    }
    std::cout << x.size(-1) << "]";
}
