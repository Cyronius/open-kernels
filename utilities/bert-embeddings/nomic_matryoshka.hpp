// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace npue::enc {

inline std::vector<float> nomic_matryoshka(std::span<const float> full, std::size_t dimensions) {
    if (dimensions == 0 || dimensions > full.size())
        throw std::invalid_argument("invalid Nomic embedding dimensions");

    // Nomic applies LayerNorm over the full embedding before taking a prefix.
    // The final L2 normalization cancels the positive scale from LayerNorm and
    // from any L2 normalization already applied to the full-width input.
    const double mean = std::accumulate(full.begin(), full.end(), 0.0) / full.size();
    auto prefix = full.first(dimensions);
    const double sum_squares = std::accumulate(
        prefix.begin(), prefix.end(), 0.0, [mean](double sum, double value) {
            const double centered = value - mean;
            return sum + centered * centered;
        });
    const double scale = 1.0 / std::max(std::sqrt(sum_squares), 1.0e-12);
    std::vector<float> output(dimensions);
    std::transform(prefix.begin(), prefix.end(), output.begin(),
                   [mean, scale](double value) { return static_cast<float>((value - mean) * scale); });
    return output;
}

}  // namespace npue::enc
