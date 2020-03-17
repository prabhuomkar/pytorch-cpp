// Copyright 2020-present pytorch-cpp Authors
#include "vggnet.h"
#include <utility>

namespace {
    void initialize_weights(const torch::nn::Module& module) {
        torch::NoGradGuard no_grad;

        if (auto conv2d = module.as<torch::nn::Conv2d>()) {
            torch::nn::init::kaiming_normal_(conv2d->weight, 0.0, torch::kFanOut, torch::kReLU);
        } else if (auto bn2d = module.as<torch::nn::BatchNorm2d>()) {
            torch::nn::init::constant_(bn2d->weight, 1);
            torch::nn::init::constant_(bn2d->bias, 0);
        } else if (auto linear = module.as<torch::nn::Linear>()) {
            torch::nn::init::normal_(linear->weight, 0, 0.01);
            torch::nn::init::constant_(linear->bias, 0);
        }
    }
}  // namespace

inline VGGNetImpl::VGGNetImpl(const std::vector<Layer>& config, const std::set<size_t>& selected,
    bool batch_norm, const std::string& scriptmodule_file_path)
    : layers(make_layers(config, batch_norm)), selected_layer_idxs_(selected) {
    register_module("layers", layers);

    if (scriptmodule_file_path.empty()) {
        layers->apply(initialize_weights);
    } else {
        torch::load(layers, scriptmodule_file_path);
    }
}

std::vector<torch::Tensor> VGGNetImpl::forward(torch::Tensor x) {
    std::vector<torch::Tensor> tensors;

    size_t layer_id = 0;
    for (auto m : *layers) {
        x = m.forward<>(x);

        if (selected_layer_idxs_.find(layer_id) != selected_layer_idxs_.end()) {
            tensors.push_back(x);
        }

        ++layer_id;
    }

    return tensors;
}

// Create layers
//
// Translated from python code at
// https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
torch::nn::Sequential VGGNetImpl::make_layers(const std::vector<Layer>& config, bool batch_norm) {
    torch::nn::Sequential layers;
    int64_t in_channels = 3;

    for (auto layer_type : config) {
        if (layer_type == Layer::MAXPOOL) {
            layers->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
        } else {
            int64_t out_channels = 0;

            switch (layer_type) {
                case Layer::CONV64:
                    out_channels = 64;
                    break;
                case Layer::CONV128:
                    out_channels = 128;
                    break;
                case Layer::CONV256:
                    out_channels = 256;
                    break;
                case Layer::CONV512:
                    out_channels = 512;
                    break;
                default:
                    throw std::runtime_error("Invalid layer type.");
            }

            layers->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1)));

            if (batch_norm) {
                layers->push_back(torch::nn::BatchNorm2d(out_channels));
            }

            layers->push_back(torch::nn::ReLU());

            in_channels = out_channels;
        }
    }

    return layers;
}

VGGNet19Impl::VGGNet19Impl(const std::string& scriptmodule_file_path,
    const std::set<size_t>& selected_layer_idxs)
    : VGGNetImpl({
            Layer::CONV64, Layer::CONV64, Layer::MAXPOOL, Layer::CONV128, Layer::CONV128,
            Layer::MAXPOOL, Layer::CONV256, Layer::CONV256, Layer::CONV256, Layer::CONV256,
            Layer::MAXPOOL, Layer::CONV512, Layer::CONV512, Layer::CONV512, Layer::CONV512,
            Layer::MAXPOOL, Layer::CONV512, Layer::CONV512, Layer::CONV512, Layer::CONV512,
            Layer::MAXPOOL
        },
        selected_layer_idxs,
        false,
        scriptmodule_file_path) {}
