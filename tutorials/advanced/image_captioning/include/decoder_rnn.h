// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>

/**
 * "Soft" attention network.
 *
 * Based on the paper "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"
 * (see https://arxiv.org/pdf/1502.03044.pdf)
 */
class AttentionBlockImpl : public torch::nn::Module {
 public:
    AttentionBlockImpl(int64_t encoder_features, int64_t decoder_features, int64_t attention_features);

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor encoder_output, torch::Tensor decoder_output);

 private:
    torch::nn::Linear linear_encoder;
    torch::nn::Linear linear_decoder;
    torch::nn::Linear linear_attention;
};

TORCH_MODULE(AttentionBlock);

/**
 * Attention based recurrent decoder network.
 *
 * Based on the paper "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"
 * (see https://arxiv.org/pdf/1502.03044.pdf)
 */
class DecoderAttentionRNNImpl : public torch::nn::Module {
 public:
    enum class SampleMode {
        GREEDY, MULTINOMIAL
    };

    DecoderAttentionRNNImpl(int64_t embedding_size, int64_t hidden_size,
                            int64_t vocab_size, int64_t max_seq_length,
                            int64_t encoder_out_size, int64_t attention_size,
                            double dropout_p, double teacher_forcing_p);

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor encoder_output,
                                                    torch::Tensor captions, torch::Tensor lengths);

    std::pair<torch::Tensor, torch::Tensor> sample(const torch::Tensor &features,
                                                   SampleMode sample_mode = SampleMode::GREEDY);

 private:
    torch::nn::Embedding embedding;
    torch::nn::LSTMCell lstm_cell;
    torch::nn::Linear linear;
    torch::nn::Linear linear_h;
    torch::nn::Linear linear_c;
    AttentionBlock attention_block;
    torch::nn::Linear f_beta;
    torch::nn::Dropout embedding_dropout;
    torch::nn::Dropout lstm_output_dropout;
    int64_t max_seq_length_;
    double teacher_forcing_p_;
};

TORCH_MODULE(DecoderAttentionRNN);

