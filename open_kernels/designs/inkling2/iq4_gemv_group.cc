#include "gemv_q8.h"
#include "iq4_lut.h"

namespace {
constexpr unsigned compact_tile = 2560, compact_group = 576;

// Only the L1 scratch tile expands to int8. DDR carries the nibbles and the
// BF16 scales in the layout also used by the compact MXFP4 GEMV.
__attribute__((noinline)) void expand(const uint8_t* weights, uint8_t* expanded, const Iq4Decode& lookup) {
    auto* codes = reinterpret_cast<int8_t*>(expanded + kQ8ScaleBytes);
#pragma clang loop unroll(disable)
    for (unsigned g = 0; g < 4; ++g) {
        const uint8_t* src = weights + g * compact_group;
        aie::store_v(expanded + g * 64, aie::load_v<64>(src));
        src += 64;
        for (unsigned octet = 0; octet < 8; ++octet) {
            const auto lo = lookup(src + octet * 64);
            const auto hi = lookup(src + octet * 64 + 32);
            const auto both = aie::interleave_zip(lo, hi, 8);
            aie::store_v(codes + g * 1024 + octet * 128, both.first);
            aie::store_v(codes + g * 1024 + octet * 128 + 64, both.second);
        }
    }
}
}

extern "C" void inkling_iq4_group(const uint8_t* __restrict weights, const uint8_t* __restrict tab,
                                   const int16_t* bank_a, const int16_t* bank_b,
                                   uint8_t* __restrict expanded, float* __restrict output,
                                   int32_t group, int32_t width) {
    const Iq4Decode lookup(bank_a, bank_b);
#pragma clang loop unroll(disable)
    for (unsigned part = 0; part < kQ8RowSplit; ++part) {
        expand(weights + part * compact_tile, expanded, lookup);
        gemv_q8_half_tile(expanded, tab, width, group, group == 0, output + part * kQ8Rows);
    }
}
