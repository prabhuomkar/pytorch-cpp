// Copyright 2020-present pytorch-cpp Authors
#include <torch/torch.h>
#include "transformer.h"

TransformerImpl::TransformerImpl(int64_t ntoken,
                                 int64_t ninp,
                                 int64_t nhead,
                                 int64_t nhid,
                                 int64_t nlayers,
                                 double dropout)
    :
        pos_encoder(ninp, dropout),
        encoder_layers(ninp, nhead, nhid, dropout),
        transformer_encoder(encoder_layers, nlayers),
        encoder(ntoken, ninp),
        decoder(ninp, ntoken) {
    _ninp = ninp;
    // TODO(omkar, markus): init src_mask

    init_weights();

    register_module("pos_encoder", pos_encoder);
    register_module("encoder_layers", encoder_layers);
    register_module("transformer_encoder", transformer_encoder);
    register_module("encoder", encoder);
    register_module("decoder", decoder);
}

torch::Tensor TransformerImpl::_generate_square_subsequent_mask(int sz) {
    auto mask = (torch::triu(torch::ones({sz, sz})) == 1).transpose_(0, 1);
    // TODO(omkar, markus): convert to c++
    // mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask;
}

void TransformerImpl::init_weights() {
    double initrange = 0.1;
    // TODO(omkar, markus): convert to c++
    // self.encoder.weight.data.uniform_(-initrange, initrange)
    // self.decoder.bias.data.zero_()
    // self.decoder.weight.data.uniform_(-initrange, initrange)
}

torch::Tensor TransformerImpl::forward(torch::Tensor src) {
    // TODO(omkar, markus): convert to c++
    // if self.src_mask is None or self.src_mask.size(0) != src.size(0):
    //     device = src.device
    //     mask = self._generate_square_subsequent_mask(src.size(0)).to(device)
    //     self.src_mask = mask

    auto out = encoder->forward(src) * sqrt(ninp);
    out = pos_encoder->forward(out);
    out = transformer_encoder->forward(out, src_mask);
    out = decoder->forward(out);
    return out;
}
