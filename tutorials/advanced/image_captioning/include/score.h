// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>
#include <iterator>

namespace score {
struct CountFraction {
    size_t numerator = 0;
    size_t denominator = 0;

    CountFraction &operator+=(const CountFraction &other);
};

CountFraction modified_precision(const torch::Tensor &hypothesis,
                                 const std::vector<torch::Tensor> &references, size_t n);

/**
 * Provides Bleu score calculation using accumulated data
 *
 * see https://www.aclweb.org/anthology/P02-1040.pdf (BLEU: a Method for Automatic Evaluation of Machine Translation)
 * and https://www.nltk.org/_modules/nltk/translate/bleu_score.html (nltk Python implementation)
 */
class BleuScoreLogger {
 public:
    explicit BleuScoreLogger(size_t max_n = 4) : max_n_(max_n), modified_precisions_(max_n) {}

    void update(const std::vector<torch::Tensor> &hypotheses,
                const std::vector<std::vector<torch::Tensor>> &references_list);

    double bleu(size_t n) const;

    void reset();

    size_t max_n() const {
        return max_n_;
    }

 private:
    size_t max_n_;
    std::vector<CountFraction> modified_precisions_;
    size_t cum_hypothesis_length_ = 0;
    size_t cum_closest_reference_length_ = 0;
};
}  // namespace score
