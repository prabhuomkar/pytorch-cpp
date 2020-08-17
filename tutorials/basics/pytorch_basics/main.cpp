// Copyright 2020-present pytorch-cpp Authors
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <iomanip>

void print_tensor_size(const torch::Tensor&);
void print_script_module(const torch::jit::script::Module& module, size_t spaces = 0);

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
    torch::nn::Linear linear(3, 2);
    std::cout << "w:\n" << linear->weight << '\n';
    std::cout << "b:\n" << linear->bias << '\n';

    // Create loss function and optimizer
    torch::nn::MSELoss criterion;
    torch::optim::SGD optimizer(linear->parameters(), torch::optim::SGDOptions(0.01));

    // Forward pass
    auto pred = linear->forward(x);

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
    pred = linear->forward(x);
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
    //             SLICING AND EXTRACTING PARTS FROM TENSORS           //
    // =============================================================== //

    std::cout << "---- SLICING AND EXTRACTING PARTS FROM TENSORS ----\n";

    std::vector<int64_t> test_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    torch::Tensor s = torch::from_blob(test_data.data(), {3, 3}, torch::kInt64);
    std::cout << "s:\n" << s << '\n';
    // Output:
    // 1 2 3
    // 4 5 6
    // 7 8 9

    using torch::indexing::Slice;
    using torch::indexing::None;
    using torch::indexing::Ellipsis;

    // Indexing and slicing tensors is done very similarly to how
    // it is done in Python.
    //
    // For a complete side-by-side translation of all indexing types, see:
    // https://pytorch.org/cppdocs/notes/tensor_indexing.html

    // Extract a single element tensor:
    std::cout << "\"s[0,2]\" as tensor:\n" << s.index({0, 2}) << '\n';
    std::cout << "\"s[0,2]\" as value:\n" << s.index({0, 2}).item<int64_t>() << '\n';
    // Output:
    // 3

    // Slice a tensor along a dimension at a given index.
    std::cout << "\"s[:,2]\":\n" << s.index({Slice(), 2}) << '\n';
    // Output:
    // 3
    // 6
    // 9

    // Slice a tensor along a dimension at given indices from
    // a start-index up to - but not including - an end-index using a given step size.
    std::cout << "\"s[:2,:]\":\n" << s.index({Slice(None, 2), Slice()}) << '\n';
    // Output:
    // 1 2 3
    // 4 5 6
    std::cout << "\"s[:,1:]\":\n" << s.index({Slice(), Slice(1, None)}) << '\n';
    // Output:
    // 2 3
    // 5 6
    // 8 9
    std::cout << "\"s[:,::2]\":\n" << s.index({Slice(), Slice(None, None, 2)}) << '\n';
    // Output:
    // 1 3
    // 4 6
    // 7 9

    // Combination.
    std::cout << "\"s[:2,1]\":\n" << s.index({Slice(None, 2), 1}) << '\n';
    // Output:
    // 2
    // 5

    // Ellipsis (...).
    std::cout << "\"s[..., :2]\":\n" << s.index({Ellipsis, Slice(None, 2)}) << "\n\n";
    // Output:
    // 1 2
    // 4 5
    // 7 8

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
    std::cout << example.data.sizes() << "\n";
    std::cout << "Sample target: " << example.target.item<int>() << "\n";

    // Construct data loader
    auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        dataset, 64);

    // Fetch a mini-batch
    auto example_batch = *dataloader->begin();
    std::cout << "Sample batch - data size: ";
    std::cout << example_batch.data.sizes() << "\n";
    std::cout << "Sample batch - target size: ";
    std::cout << example_batch.target.sizes() << "\n\n";

    // Actual usage of the dataloader:
    // for (auto& batch : *dataloader) {
        // Training code here
    // }

    // =============================================================== //
    //               INPUT PIPELINE FOR CUSTOM DATASET                 //
    // =============================================================== //

    // See Deep Residual Network tutorial files cifar10.h and cifar10.cpp
    // for an example of a custom dataset implementation.

    // =============================================================== //
    //                        PRETRAINED MODEL                         //
    // =============================================================== //

    std::cout << "---- PRETRAINED MODEL ----\n";
    // Loading a pretrained model using the C++ API is done
    // in the following way:
    // In Python:
    // (1) Create the (pretrained) pytorch model.
    // (2) Convert the pytorch model to a torch.jit.ScriptModule (via tracing or by using annotations)
    // (3) Serialize the scriptmodule to a file.
    // In C++:
    // (4) Load the scriptmodule form the file using torch::jit::load()
    // See https://pytorch.org/tutorials/advanced/cpp_export.html for more infos.

    // Path to serialized ScriptModule of pretrained resnet18 model,
    // created in Python.
    // You can use the provided Python-script "create_resnet18_scriptmodule.py" in
    // tutorials/basics/pytorch-basics/model to create the necessary file.
    const std::string pretrained_model_path = "../../../../tutorials/basics/pytorch_basics/model/"
        "resnet18_scriptmodule.pt";

    torch::jit::script::Module resnet;

    try {
        resnet = torch::jit::load(pretrained_model_path);
    }
    catch (const torch::Error& error) {
        std::cerr << "Could not load scriptmodule from file " << pretrained_model_path << ".\n"
            << "You can create this file using the provided Python script 'create_resnet18_scriptmodule.py' "
            "in tutorials/basics/pytorch-basics/model/.\n";
        return -1;
    }

    std::cout << "Resnet18 model:\n";

    print_script_module(resnet, 2);

    std::cout << "\n";

    const auto fc_weight = resnet.attr("fc").toModule().attr("weight").toTensor();

    auto in_features = fc_weight.size(1);
    auto out_features = fc_weight.size(0);

    std::cout << "Fully connected layer: in_features=" << in_features << ", out_features=" << out_features << "\n";

    // Input sample
    auto sample_input = torch::randn({1, 3, 224, 224});
    std::vector<torch::jit::IValue> inputs{sample_input};

    // Forward pass
    std::cout << "Input size: ";
    std::cout << sample_input.sizes() << "\n";
    auto output = resnet.forward(inputs).toTensor();
    std::cout << "Output size: ";
    std::cout << output.sizes() << "\n\n";

    // =============================================================== //
    //                      SAVE AND LOAD A MODEL                      //
    // =============================================================== //

    std::cout << "---- SAVE AND LOAD A MODEL ----\n";

    // Simple example model
    torch::nn::Sequential model{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 3).stride(2).padding(1)),
        torch::nn::ReLU()
    };

    // Path to the model output file (all folders must exist!).
    const std::string model_save_path = "output/model.pt";

    // Save the model
    torch::save(model, model_save_path);

    std::cout << "Saved model:\n" << model << "\n";

    // Load the model
    torch::load(model, model_save_path);

    std::cout << "Loaded model:\n" << model;
}

void print_script_module(const torch::jit::script::Module& module, size_t spaces) {
    for (const auto& sub_module : module.named_children()) {
        if (!sub_module.name.empty()) {
            std::cout << std::string(spaces, ' ') << sub_module.value.type()->name().value().name()
                << " " << sub_module.name << "\n";
        }

        print_script_module(sub_module.value, spaces + 2);
    }
}

