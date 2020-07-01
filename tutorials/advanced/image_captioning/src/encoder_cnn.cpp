// Copyright 2020-present pytorch-cpp Authors
#include "encoder_cnn.h"
#include <torch/script.h>

EncoderCNNImpl::EncoderCNNImpl(const std::string &backbone_scriptmodule_file_path, int64_t out_wh, int64_t out_size)
        : pool(torch::nn::AdaptiveAvgPool2dOptions({out_wh, out_wh})), bn(out_size) {
    try {
        backbone = torch::jit::load(backbone_scriptmodule_file_path);
    }
    catch (const torch::Error &error) {
        throw std::runtime_error("Could not load scriptmodule from file "
        + backbone_scriptmodule_file_path +
        ".\nYou can create this file using the provided Python script 'create_encoder_cnn_backbone_scriptmodule.py' "
        "in tutorials/advanced/image_captioning/model/.");
    }

    if (!backbone.hasattr("out_channels")) {
        throw std::runtime_error("Could not read 'out_channels' attribute from encoder backbone scriptmodule, "
                                 "make sure it is registered as a buffer in Python!");
    }

    auto backbone_out_size = backbone.attr("out_channels").toTensor().item<int64_t>();

    conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(backbone_out_size, out_size, 1).bias(false));

    register_module("pool", pool);
    register_module("conv", conv);
    register_module("bn", bn);
}

torch::Tensor EncoderCNNImpl::forward(torch::Tensor x) {
    x = backbone.forward({x}).toTensor();
    x = pool->forward(x);
    x = conv->forward(x);
    x = bn->forward(x);
    x = torch::relu(x);
    return x;
}

void EncoderCNNImpl::to(torch::Device device, bool non_blocking) {
    torch::nn::Module::to(device, non_blocking);
    backbone.to(device, non_blocking);
}

void EncoderCNNImpl::to(torch::ScalarType dtype, bool non_blocking) {
    torch::nn::Module::to(dtype, non_blocking);
    backbone.to(dtype, non_blocking);
}

void EncoderCNNImpl::to(torch::Device device, torch::ScalarType dtype, bool non_blocking) {
    torch::nn::Module::to(device, dtype, non_blocking);
    backbone.to(device, dtype, non_blocking);
}
