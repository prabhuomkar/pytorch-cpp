// Copyright 2020-present pytorch-cpp Authors
#include "decoder_rnn.h"
#include <torch/types.h>

using torch::indexing::Slice;
using torch::indexing::Ellipsis;
using torch::indexing::None;

AttentionBlockImpl::AttentionBlockImpl(int64_t encoder_features, int64_t decoder_features, int64_t attention_features) :
        linear_encoder(encoder_features, attention_features),
        linear_decoder(decoder_features, attention_features),
        linear_attention(attention_features, 1) {
    register_module("linear_encoder", linear_encoder);
    register_module("linear_decoder", linear_decoder);
    register_module("linear_attention", linear_attention);
}

std::pair<torch::Tensor, torch::Tensor> AttentionBlockImpl::forward(torch::Tensor encoder_output,
                                                                    torch::Tensor decoder_output) {
    auto encoder_attention_input = linear_encoder->forward(encoder_output.permute({0, 2, 1}));
    auto decoder_attention_input = linear_decoder->forward(decoder_output).unsqueeze_(1);

    auto attention_input = torch::tanh(encoder_attention_input.add(decoder_attention_input));

    auto attention_output = linear_attention->forward(attention_input).squeeze_(2);

    auto alpha = torch::softmax(attention_output, 1);

    auto encoder_attention_output = encoder_output.mul(alpha.unsqueeze(1)).sum(2);

    return {encoder_attention_output, alpha};
}

DecoderAttentionRNNImpl::DecoderAttentionRNNImpl(int64_t embedding_size, int64_t hidden_size, int64_t vocab_size,
                                                 int64_t max_seq_length, int64_t encoder_features,
                                                 int64_t attention_size,
                                                 double dropout_p, double teacher_forcing_p)
        : embedding(vocab_size, embedding_size),
          lstm_cell(torch::nn::LSTMCellOptions(encoder_features + embedding_size, hidden_size).bias(true)),
          linear(hidden_size, vocab_size),
          linear_h(encoder_features, hidden_size),
          linear_c(encoder_features, hidden_size),
          attention_block(encoder_features, hidden_size, attention_size),
          f_beta(hidden_size, encoder_features),
          embedding_dropout(dropout_p),
          lstm_output_dropout(dropout_p),
          max_seq_length_(max_seq_length),
          teacher_forcing_p_(teacher_forcing_p) {
    register_module("embedding", embedding);
    register_module("lstm_cell", lstm_cell);
    register_module("linear", linear);
    register_module("linear_h", linear_h);
    register_module("linear_c", linear_c);
    register_module("attention_block", attention_block);
    register_module("f_beta", f_beta);
    register_module("embedding_dropout", embedding_dropout);
    register_module("lstm_output_dropout", lstm_output_dropout);
}

/**
 * Forward method.
 *
 * @param encoder_output [batch_size, encoder_out_size, encoder_out_wh, encoder_out_wh]
 * @param captions [batch_size, max_batch_caption_length]
 * @param lengths [batch_size]
 * @return {predictions, alphas} {[batch_size, max_batch_caption_length, vocab_size],
 *                                [batch_size, max_batch_caption_length, encoder_out_wh * encoder_out_wh]}
 */
std::pair<torch::Tensor, torch::Tensor> DecoderAttentionRNNImpl::forward(torch::Tensor encoder_output,
                                                                         torch::Tensor captions,
                                                                         torch::Tensor lengths) {
    // [batch_size, max_batch_caption_length, embedding_size]
    auto embeddings = embedding_dropout->forward(embedding->forward(captions));

    // [batch_size, encoder_out_size, encoder_out_wh * encoder_out_wh]
    auto features_reshaped = encoder_output.view({encoder_output.size(0), encoder_output.size(1), -1});

    auto features_mean = features_reshaped.mean(2);  // [batch_size, encoder_out_size]

    auto h = torch::tanh(linear_h->forward(features_mean));  // [batch_size, hidden_size]
    auto c = torch::tanh(linear_c->forward(features_mean));  // [batch_size, hidden_size]

    auto max_caption_size = lengths[0].item<int64_t>();

    // [batch_size, max_batch_caption_length, vocab_size]
    auto predictions = torch::zeros({features_reshaped.size(0),
                                     max_caption_size, linear->options.out_features()}, features_reshaped.device());
    // [batch_size, max_batch_caption_length, encoder_out_wh * encoder_out_wh]
    auto alphas = torch::zeros({features_reshaped.size(0), max_caption_size,
                                features_reshaped.size(2)}, features_reshaped.device());

    torch::Tensor embedding_batch_t;
    torch::Tensor prediction_batch_t;

    for (decltype(max_caption_size) i = 0; i != max_caption_size; ++i) {
        const auto batch_size_t = (lengths > i).sum().item<int64_t>();

        auto encoder_out_batch_t = features_reshaped.index({Slice(None, batch_size_t), Ellipsis});

        auto h_batch_t = h.index({Slice(None, batch_size_t), Ellipsis});

        auto c_batch_t = c.index({Slice(None, batch_size_t), Ellipsis});

        torch::Tensor attention_weighted_encoding;
        torch::Tensor alpha;

        std::tie(attention_weighted_encoding, alpha) = attention_block->forward(encoder_out_batch_t, h_batch_t);

        auto gate = torch::sigmoid(f_beta->forward(h_batch_t));
        attention_weighted_encoding.mul_(gate);

        if (i == 0 || torch::rand(1)[0].item<double>() < teacher_forcing_p_) {
            embedding_batch_t = embeddings.index({Slice(None, batch_size_t), i, Slice()});
        } else {
            embedding_batch_t = embedding->forward(prediction_batch_t.argmax(1))
                    .index({Slice(None, batch_size_t), Slice()});
        }

        auto input_batch_t = torch::cat({embedding_batch_t, attention_weighted_encoding}, 1);

        std::tie(h, c) = lstm_cell->forward(input_batch_t, std::make_tuple(h_batch_t, c_batch_t));

        prediction_batch_t = linear->forward(lstm_output_dropout->forward(h));

        predictions.index_put_({Slice(None, batch_size_t), i, Slice()}, prediction_batch_t);
        alphas.index_put_({Slice(None, batch_size_t), i, Slice()}, alpha);
    }

    return {predictions, alphas};
}

std::pair<torch::Tensor, torch::Tensor>
DecoderAttentionRNNImpl::sample(const torch::Tensor &features, DecoderAttentionRNNImpl::SampleMode sample_mode) {
    auto start_input = torch::ones({features.size(0)},
                                   torch::TensorOptions(torch::kLong).device(features.device()));

    auto inputs = embedding_dropout->forward(embedding->forward(start_input));

    auto features_reshaped = features.view({features.size(0), features.size(1), -1});

    auto features_mean = features_reshaped.mean(2);

    auto h = torch::tanh(linear_h->forward(features_mean));
    auto c = torch::tanh(linear_c->forward(features_mean));

    std::vector<torch::Tensor> sampled_ids;
    std::vector<torch::Tensor> alphas;

    for (decltype(max_seq_length_) i = 0; i != max_seq_length_; ++i) {
        torch::Tensor attention_weighted_encoding;
        torch::Tensor alpha;

        std::tie(attention_weighted_encoding, alpha) = attention_block->forward(features_reshaped, h);

        auto gate = torch::sigmoid(f_beta->forward(h));
        attention_weighted_encoding.mul_(gate);

        auto input_t = torch::cat({inputs, attention_weighted_encoding}, 1);

        std::tie(h, c) = lstm_cell->forward(input_t, std::make_tuple(h, c));

        auto output = linear->forward(lstm_output_dropout->forward(h));

        torch::Tensor predicted;

        if (sample_mode == SampleMode::GREEDY) {
            predicted = output.argmax(1);
        } else if (sample_mode == SampleMode::MULTINOMIAL) {
            predicted = output.softmax(1).multinomial(1).squeeze_(1);
        }

        sampled_ids.push_back(predicted);
        alphas.push_back(alpha);

        inputs = embedding->forward(predicted);
    }
    auto alpha_stack = torch::stack(alphas, 1);

    const auto attention_h = static_cast<int64_t>(sqrt(static_cast<double>(alpha_stack.size(2))
                                                       * features.size(-2) / features.size(-1)));
    const auto attention_w = static_cast<int64_t>(sqrt(static_cast<double>(alpha_stack.size(2))
                                                       * features.size(-1) / features.size(-2)));

    return {torch::stack(sampled_ids, 1), alpha_stack.reshape({-1, alpha_stack.size(1),
                                                               attention_h, attention_w})};
}
