// SPDX-License-Identifier: MIT
#pragma once

#include "npue.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace npue::enc {

class GteSparseHead {
public:
    GteSparseHead(const npue::File& model, std::array<int32_t, 4> specials)
        : weights(read(model, "sparse.weight", {1, width})),
          bias(*read(model, "sparse.bias", {1})), special_ids(specials) {}

    std::map<int32_t, float> score(std::span<const float> hidden,
                                   std::span<const int32_t> ids) const {
        if (hidden.size() != ids.size() * width)
            throw std::runtime_error("GTE sparse states and token IDs disagree");
        std::map<int32_t, float> result;
        for (size_t i = 0; i < ids.size(); ++i) {
            const auto id = ids[i];
            if (std::find(special_ids.begin(), special_ids.end(), id) != special_ids.end())
                continue;
            auto row = hidden.subspan(i * width, width);
            const float score = static_cast<float>(std::inner_product(
                row.begin(), row.end(), weights, static_cast<double>(bias)));
            if (!std::isfinite(score))
                throw std::runtime_error("GTE sparse output is not finite");
            if (score > 0) {
                auto& best = result[id];
                best = std::max(best, score);
            }
        }
        return result;
    }

private:
    static constexpr size_t width = 768;
    const float* weights;
    float bias;
    std::array<int32_t, 4> special_ids;

    static const float* read(const npue::File& model, const std::string& key,
                             const std::vector<int64_t>& shape) {
        const auto& info = model.info(key);
        const auto count = std::accumulate(shape.begin(), shape.end(), size_t{1},
                                           [](size_t n, int64_t dim) { return n * dim; });
        const auto data = model.raw(key);
        if (info.dtype != "F32" || info.logical_shape != shape ||
            data.bytes != count * sizeof(float))
            throw std::runtime_error("invalid GTE sparse tensor " + key);
        const auto* values = data.as<float>();
        if (!std::all_of(values, values + count, [](float v) { return std::isfinite(v); }))
            throw std::runtime_error("non-finite GTE sparse tensor " + key);
        return values;
    }
};

}  // namespace npue::enc
