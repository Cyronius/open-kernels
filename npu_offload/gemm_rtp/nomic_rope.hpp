// SPDX-License-Identifier: MIT
#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace npue::npu {

// Nomic Embed Text v1.5 uses the original base through 2,048 positions and
// dynamic NTK scaling per input beyond that boundary.
inline double nomic_v15_rope_base(int64_t tokens) {
    if (tokens < 0 || tokens > 8192)
        throw std::invalid_argument("Nomic v1.5 rotary input length is outside 0–8192");
    constexpr double base = 1000.0;
    constexpr int64_t original_context = 2048;
    if (tokens <= original_context) return base;
    return base * std::pow(2.0 * tokens / original_context - 1.0, 64.0 / 62.0);
}

}  // namespace npue::npu
