// Copyright 2020-present pytorch-cpp Authors
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include "vggnet.h"
#include "load_image.h"
#include "save_image.h"

using image_utils::load_image;
using image_utils::save_image;

void print_sizes(torch::Tensor);

int main() {
    std::cout << "Neural Style Transfer\n\n";

    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    // Hyper parameters
    const int64_t max_image_size = 300;
    const double learning_rate = 3e-3;
    const double style_loss_weight = 100;
    const size_t num_total_steps = 2000;
    const size_t log_step = 10;
    const size_t sample_step = 500;

    // Paths to content and style images
    const std::string content_image_path = "../../../../data/neural_style_transfer_images/content.png";
    const std::string style_image_path = "../../../../data/neural_style_transfer_images/style.png";

    // Path to pre-learned VGG19 layers scriptmodule file.
    // Must be created using the provided python script at
    // pytorch-cpp/tutorials/advanced/neural_style_transfer/model/create_vgg19_layers_scriptmodule.py.
    const std::string vgg19_layers_scriptmodule_path =
        "../../../../tutorials/advanced/neural_style_transfer/model/vgg19_layers.pt";

    if (!std::ifstream(vgg19_layers_scriptmodule_path)) {
        std::cout << "Could not open the required VGG19 layers scriptmodule file from path: "
            << vgg19_layers_scriptmodule_path << ".\nThis file must be created using the provided python script at "
            "pytorch-cpp/tutorials/advanced/neural_style_transfer/model/create_vgg19_layers_scriptmodule.py."
            << std::endl;
        return -1;
    }

    // Create necessary normalization and denormalization transforms
    torch::data::transforms::Normalize<> normalize_transform({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225});
    torch::data::transforms::Normalize<> denormalize_transform({-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225},
        {1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225});

    // Load content and style images and resize style image to the same size as content image
    auto content = load_image(content_image_path, max_image_size, normalize_transform).unsqueeze_(0);
    auto style = load_image(style_image_path, {content.size(2), content.size(3)}, normalize_transform).unsqueeze_(0);

    // Initialize a target with the content image.
    // During training this image will be transformed to resemble
    // the style of the style image.
    auto target = content.clone();

    // Move tensors to device
    content = content.to(device);
    style = style.to(device);
    target = target.to(device);

    // Model
    VGGNet19 model(vgg19_layers_scriptmodule_path);
    model->to(device);
    model->eval();

    // Optimizer
    torch::optim::Adam optimizer(std::vector<torch::Tensor>{target.requires_grad_(true)},
        torch::optim::AdamOptions(learning_rate).betas(std::make_tuple(0.5, 0.999)));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Training...\n";

    // Generate images
    for (size_t step = 0; step != num_total_steps; ++step) {
        // Forward pass and extract feature tensors from some Conv2d layers
        auto target_features = model->forward(target);
        auto content_features = model->forward(content);
        auto style_features = model->forward(style);

        auto style_loss = torch::zeros({1}, torch::TensorOptions(device));
        auto content_loss = torch::zeros({1}, torch::TensorOptions(device));

        for (size_t f_id = 0; f_id != target_features.size(); ++f_id) {
            // Compute content loss between target and content feature images
            content_loss += torch::nn::functional::mse_loss(target_features[f_id], content_features[f_id]);

            auto c = target_features[f_id].size(1);
            auto h = target_features[f_id].size(2);
            auto w = target_features[f_id].size(3);

            // Reshape convolutional feature maps
            auto target_feature = target_features[f_id].view({c, h * w});
            auto style_feature = style_features[f_id].view({c, h * w});

            // Compute gram matrices
            target_feature = torch::mm(target_feature, target_feature.t());
            style_feature = torch::mm(style_feature, style_feature.t());

            // Compute style loss
            style_loss += torch::nn::functional::mse_loss(target_feature, style_feature) / (c * h * w);
        }

        // Compute total loss
        auto total_loss = content_loss + style_loss_weight * style_loss;

        // Backward pass and optimize
        optimizer.zero_grad();
        total_loss.backward();
        optimizer.step();

        if ((step + 1) % log_step == 0) {
            // Print losses
            std::cout << "Step [" << step + 1 << "/" << num_total_steps
                << "], Content Loss: " << content_loss.item<double>()
                << ", Style Loss: " << style_loss.item<double>() << "\n";
        }

        if ((step + 1) % sample_step == 0) {
            // Save the generated image
            auto image = denormalize_transform(target.to(torch::kCPU).clone().squeeze(0)).clamp_(0, 1);
            save_image(image, "output/output-" + std::to_string(step + 1) + ".png", 1, 0);
        }
    }

    std::cout << "Training finished!\n";
}
