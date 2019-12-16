// Copyright 2019 Markus Fleischhacker
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include "save_image.h"

using image_utils::save_image;

int main() {
    std::cout << "Generative Adversarial Network\n\n";

    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    // Hyper parameters
    const int64_t latent_size = 64;
    const int64_t hidden_size = 256;
    const int64_t image_size = 28 * 28;
    const int64_t batch_size = 100;
    const size_t num_epochs = 200;
    const double learning_rate = 0.0002;

    const std::string MNIST_data_path = "../../../../data/mnist/";

    // Path of the directory where the generated samples will be saved to (This folder must exist!)
    const std::string sample_output_dir_path = "../../../../tutorials/advanced/generative_adversarial_network/output/";

    // MNIST dataset
    auto dataset = torch::data::datasets::MNIST(MNIST_data_path)
        .map(torch::data::transforms::Normalize<>(0.5, 0.5))
        .map(torch::data::transforms::Stack<>());

    // Number of samples in the dataset
    auto num_samples = dataset.size().value();

    // Data loader
    auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(dataset), batch_size);

    // Model
    // - Discriminator
    torch::nn::Sequential D{
        torch::nn::Linear(image_size, hidden_size),
        torch::nn::Functional(torch::leaky_relu, 0.2),
        torch::nn::Linear(hidden_size, hidden_size),
        torch::nn::Functional(torch::leaky_relu, 0.2),
        torch::nn::Linear(hidden_size, 1),
        torch::nn::Functional(torch::sigmoid)
    };

    // - Generator
    torch::nn::Sequential G{
        torch::nn::Linear(latent_size, hidden_size),
        torch::nn::Functional(torch::relu),
        torch::nn::Linear(hidden_size, hidden_size),
        torch::nn::Functional(torch::relu),
        torch::nn::Linear(hidden_size, image_size),
        torch::nn::Functional(torch::tanh)
    };

    D->to(device);
    G->to(device);

    // Optimizers
    auto d_optimizer = torch::optim::Adam(D->parameters(), torch::optim::AdamOptions(learning_rate));
    auto g_optimizer = torch::optim::Adam(G->parameters(), torch::optim::AdamOptions(learning_rate));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    auto denorm = [] (torch::Tensor tensor) { return tensor.add(1).div_(2).clamp_(0, 1); };

    std::cout << "Training...\n";

    // Train the model
    for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
        torch::Tensor images;
        torch::Tensor fake_images;
        size_t batch_index = 0;

        for (auto& batch : *dataloader) {
            // Transfer images to device
            images = batch.data.reshape({batch_size, -1}).to(device);

            // Create the labels which are later used as input for the loss
            auto real_labels = torch::ones({batch_size, 1}).to(device);
            auto fake_labels = torch::zeros({batch_size, 1}).to(device);

            // ================================================================== #
            //                      Train the discriminator                       #
            // ================================================================== #

            // Compute binary cross entropy loss using real images where
            // binary_cross_entropy(x, y) = -y * log(D(x)) - (1 - y) * log(1 - D(x))
            // Second term of the loss is always zero since real_labels == 1
            auto outputs = D->forward(images);
            auto d_loss_real = torch::binary_cross_entropy(outputs, real_labels);
            auto real_score = outputs.mean().item<double>();

            // Compute binary cross entropy loss using fake images
            // First term of the loss is always zero since fake_labels == 0
            auto z = torch::randn({batch_size, latent_size}).to(device);
            fake_images = G->forward(z);
            outputs = D->forward(fake_images);
            auto d_loss_fake = torch::binary_cross_entropy(outputs, fake_labels);
            auto fake_score = outputs.mean().item<double>();

            auto d_loss = d_loss_real + d_loss_fake;

            // Backward pass and optimize
            d_optimizer.zero_grad();
            d_loss.backward();
            d_optimizer.step();

            // ================================================================== #
            //                        Train the generator                         #
            // ================================================================== #

            // Compute loss with fake images
            z = torch::randn({batch_size, latent_size}).to(device);
            fake_images = G->forward(z);
            outputs = D->forward(fake_images);

            // We train G to maximize log(D(G(z)) instead of minimizing log(1 - D(G(z)))
            // For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
            auto g_loss = torch::binary_cross_entropy(outputs, real_labels);

            // Backward pass and optimize
            g_optimizer.zero_grad();
            g_loss.backward();
            g_optimizer.step();

            if ((batch_index + 1) % 200 == 0) {
                std::cout << "Epoch [" << epoch << "/" << num_epochs << "], Step [" << batch_index + 1 << "/"
                    << num_samples / batch_size << "], d_loss: " << d_loss.item<double>() << ", g_loss: "
                    << g_loss.item<double>() << ", D(x): " << real_score
                    << ", D(G(z)): " << fake_score << "\n";
            }

            ++batch_index;
        }

        // Save real images once
        if (epoch == 0) {
            images = denorm(images.reshape({images.size(0), 1, 28, 28}));
            save_image(images, sample_output_dir_path + "real_images.png");
        }

        // Save generated fake images
        fake_images = denorm(fake_images.reshape({fake_images.size(0), 1, 28, 28}));
        save_image(fake_images, sample_output_dir_path + "fake_images-" + std::to_string(epoch + 1) + ".png");
    }

    std::cout << "Training finished!\n";
}
