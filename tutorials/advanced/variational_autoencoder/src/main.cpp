// Copyright 2020-present pytorch-cpp Authors
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include "variational_autoencoder.h"
#include "image_io.h"

using image_io::save_image;

int main() {
    std::cout << "Variational Autoencoder\n\n";

    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    // Hyper parameters
    const int64_t h_dim = 400;
    const int64_t z_dim = 20;
    const int64_t image_size = 28 * 28;
    const int64_t batch_size = 100;
    const size_t num_epochs = 15;
    const double learning_rate = 1e-3;

    const std::string MNIST_data_path = "../../../../data/mnist/";

    // Path of the directory where the sampled and reconstructed images will be saved to (This folder must exist!)
    const std::string sample_output_dir_path = "output/";

    // MNIST dataset
    auto dataset = torch::data::datasets::MNIST(MNIST_data_path)
        .map(torch::data::transforms::Stack<>());

    // Number of samples in the dataset
    auto num_samples = dataset.size().value();

    // Data loader
    auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(dataset), batch_size);

    // Model
    VAE model(image_size, h_dim, z_dim);
    model->to(device);

    // Optimizer
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Training...\n";

    // Train the model
    for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
        torch::Tensor images;
        size_t batch_index = 0;

        model->train();

        for (auto& batch : *dataloader) {
            // Transfer images to device
            images = batch.data.reshape({-1, image_size}).to(device);

            // Forward pass
            auto output = model->forward(images);

            // Compute reconstruction loss and kl divergence
            // For KL divergence, see Appendix B in VAE paper https://arxiv.org/pdf/1312.6114.pdf
            auto reconstruction_loss = torch::nn::functional::binary_cross_entropy(output.reconstruction, images,
                torch::nn::functional::BinaryCrossEntropyFuncOptions().reduction(torch::kSum));
            auto kl_divergence = -0.5 * torch::sum(1 + output.log_var - output.mu.pow(2) - output.log_var.exp());

            // Backward pass and optimize
            auto loss = reconstruction_loss + kl_divergence;
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            if ((batch_index + 1) % 100 == 0) {
                std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "], Step [" << batch_index + 1 << "/"
                    << num_samples / batch_size << "], Reconstruction loss: "
                    << reconstruction_loss.item<double>() / batch.data.size(0)
                    << ", KL-divergence: " << kl_divergence.item<double>() / batch.data.size(0)
                    << "\n";
            }
            ++batch_index;
        }

        model->eval();
        torch::NoGradGuard no_grad;

        // Sample a batch of codings from the unit Gaussian Distribution, then decode them using the Decoder
        // and save the resulting images.
        auto z = torch::randn({batch_size, z_dim}).to(device);
        auto images_decoded = model->decode(z).view({-1, 1, 28, 28});
        save_image(images_decoded, sample_output_dir_path + "sampled-" + std::to_string(epoch + 1) + ".png");

        // Save the target and reconstructed images from the last batch in this epoch.
        // The saved png image contains (target | reconstruction)-pairs of columns of digits
        auto output = model->forward(images);
        auto images_concatenated = torch::cat({images.view({-1, 1, 28, 28}),
            output.reconstruction.view({-1, 1, 28, 28})}, 3);
        save_image(images_concatenated, sample_output_dir_path + "reconstruction-"
            + std::to_string(epoch + 1) + ".png");
    }

    std::cout << "Training finished!\n";
}
