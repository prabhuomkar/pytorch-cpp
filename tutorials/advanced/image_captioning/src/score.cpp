// Copyright 2020-present pytorch-cpp Authors
#include "score.h"
#include <vector>
#include <cmath>

namespace score {
namespace {
torch::Tensor n_grams(const torch::Tensor &sequence, size_t n) {
    return static_cast<size_t>(sequence.size(0)) < n ?
           torch::empty({0}, sequence.options()) : sequence.unfold(-1, n, 1);
}

int64_t closest_reference_size(int64_t hypothesis_size, const std::vector<torch::Tensor> &references) {
    std::vector<int64_t> reference_sizes(references.size());

    std::transform(references.cbegin(), references.cend(), reference_sizes.begin(),
                   [](const auto &reference) { return reference.size(0); });

    return *std::min_element(reference_sizes.cbegin(), reference_sizes.cend(),
                             [hypothesis_size](int64_t l, int64_t r) {
                                 return std::abs(l - hypothesis_size) <=
                                        std::abs(r - hypothesis_size);
                             });
}

double brevity_penalty(int64_t hypothesis_size, int64_t closest_ref_size) {
    return (hypothesis_size > closest_ref_size) ? 1.0 :
           std::exp(1.0 - static_cast<double>(closest_ref_size) / hypothesis_size);
}
}  // namespace

CountFraction &CountFraction::operator+=(const CountFraction &other) {
    numerator += other.numerator;
    denominator += other.denominator;

    return *this;
}

CountFraction modified_precision(const torch::Tensor &hypothesis,
                                 const std::vector<torch::Tensor> &references, size_t n) {
    auto actual_n_grams = n_grams(hypothesis, n);

    std::vector<torch::Tensor> reference_n_gram_sequences;
    reference_n_gram_sequences.reserve(references.size());

    std::transform(references.cbegin(), references.cend(), std::back_inserter(reference_n_gram_sequences),
                   [n](const auto &ref) { return n_grams(ref, n); });

    torch::Tensor unique_hypo_n_grams;
    torch::Tensor counts_in_hypo;

    std::tie(unique_hypo_n_grams, std::ignore, counts_in_hypo) =
            torch::unique_dim(actual_n_grams, 0, false, false, true);

    auto counts_in_refs = torch::zeros(unique_hypo_n_grams.size(0), torch::kInt64);

    if (unique_hypo_n_grams.ndimension() == 1) {
        unique_hypo_n_grams.unsqueeze_(1);
    }

    unique_hypo_n_grams.unsqueeze_(1);

    for (auto reference : reference_n_gram_sequences) {
        if (reference.ndimension() == 1) {
            reference = reference.unsqueeze(1);
        }

        counts_in_refs = torch::max(counts_in_refs,
                                    unique_hypo_n_grams.expand({-1, reference.size(0), -1})
                                            .eq(reference).all(-1).sum(-1));
    }

    size_t count = std::max<size_t>(1, counts_in_hypo.sum().item<int64_t>());
    size_t clipped_count = torch::min(counts_in_refs, counts_in_hypo).sum().item<int64_t>();

    return {clipped_count, count};
}

void BleuScoreLogger::reset() {
    modified_precisions_.clear();
    cum_closest_reference_length_ = 0;
    cum_closest_reference_length_ = 0;
}

double BleuScoreLogger::bleu(size_t n) const {
    if (n > max_n_) {
        throw std::invalid_argument("Invalid n for bleu score.");
    }

    if (modified_precisions_[0].numerator == 0) {
        return 0;
    }

    double score = 0;

    for (size_t j = 1; j <= n; ++j) {
        score += std::log(static_cast<double>(modified_precisions_[j - 1].numerator)
                          / modified_precisions_[j - 1].denominator);
    }

    return brevity_penalty(cum_hypothesis_length_, cum_closest_reference_length_) * std::exp(score / n);
}

void BleuScoreLogger::update(const std::vector<torch::Tensor> &hypotheses,
                             const std::vector<std::vector<torch::Tensor>> &references_list) {
    for (decltype(hypotheses.size()) i = 0; i != hypotheses.size(); ++i) {
        for (size_t j = 1; j <= max_n_; ++j) {
            modified_precisions_[j - 1] += score::modified_precision(hypotheses[i], references_list[i], j);
        }

        cum_hypothesis_length_ += hypotheses[i].size(0);
        cum_closest_reference_length_ += closest_reference_size(hypotheses[i].size(0),
                                                                references_list[i]);
    }
}
}  // namespace score
