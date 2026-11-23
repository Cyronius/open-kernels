// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>

namespace npue::npu {

inline bool bert_attention_geometry(int64_t sequence, int64_t heads, int64_t width) {
    if (heads == 12 && width == 32) return sequence == 256 || sequence == 512;
    if (heads == 12 && width == 64) return sequence == 512 || sequence == 2048 || sequence == 8192;
    return heads == 16 && width == 64 && sequence == 512;
}

}  // namespace npue::npu
