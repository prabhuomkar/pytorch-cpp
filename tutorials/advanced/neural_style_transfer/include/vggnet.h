// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>
#include <utility>
#include <set>

enum class Layer { CONV64, CONV128, CONV256, CONV512, MAXPOOL };

class VGGNetImpl : public torch::nn::Module {
 public:
    VGGNetImpl(const std::vector<Layer>& config, const std::set<size_t>& selected,
        bool batch_norm, const std::string& scriptmodule_file_path);

    std::vector<torch::Tensor> forward(torch::Tensor x);
    void set_selected_layer_idxs(const std::set<size_t>& idxs) { selected_layer_idxs_ = idxs; }
    std::set<size_t> get_selected_layer_idxs() const { return selected_layer_idxs_; }
    size_t get_num_layers() const { return layers->size(); }
 private:
    torch::nn::Sequential make_layers(const std::vector<Layer>& config, bool batch_norm);

    torch::nn::Sequential layers;
    std::set<size_t> selected_layer_idxs_;
};

TORCH_MODULE(VGGNet);

class VGGNet19Impl : public VGGNetImpl {
 public:
     VGGNet19Impl(const std::string& scriptmodule_file_path = {},
        const std::set<size_t>& selected_layer_idxs = {0, 5, 10, 19, 28});
};

TORCH_MODULE(VGGNet19);
