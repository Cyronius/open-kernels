#include "iq4_lut.h"

#ifndef BLOCK_ROWS
#define BLOCK_ROWS 16
#endif
#ifndef BLOCK_COLS
#define BLOCK_COLS 96
#endif
#ifndef IQ4_WEIGHTS
#define IQ4_WEIGHTS 1
#endif
#ifndef EXPERT_ROW_UNROLL
#define EXPERT_ROW_UNROLL 0
#endif
#ifndef EXPERT_SKIP_EMPTY
#define EXPERT_SKIP_EMPTY 0
#endif
#ifndef SPLIT_SCALE
#define SPLIT_SCALE 0
#endif
// Read the split-scale layout but keep the one-scale arithmetic of grouped prefill.
#ifndef SPLIT_SCALE_HIGH_ONLY
#define SPLIT_SCALE_HIGH_ONLY 0
#endif

namespace {
constexpr unsigned m = BLOCK_ROWS;
constexpr unsigned n = BLOCK_COLS;
constexpr unsigned k = 32;
constexpr unsigned micro_m = 4;
constexpr unsigned micro_n = 8;
static_assert(m % micro_m == 0 && n % micro_n == 0);

using Dot = aie::mmul<micro_m, 8, micro_n, int16_t, int8_t>;
using Output = aie::accum<accfloat, micro_m * micro_n>;

void apply(const uint8_t* __restrict activations, const uint8_t* __restrict weights,
           const int16_t* lut0, const int16_t* lut1, float* __restrict y, bool first) {
#ifdef EXPERT_NULL
    return;
#endif
    aie::set_rounding(aie::rounding_mode::conv_even);
    const auto* rescale = reinterpret_cast<const bfloat16*>(activations);
    const auto* a = reinterpret_cast<const int16_t*>(activations + m * micro_n * sizeof(bfloat16));
    const auto* scales = reinterpret_cast<const bfloat16*>(weights);
    const auto* codes = weights + n * (1 + SPLIT_SCALE) * sizeof(bfloat16);
#if IQ4_WEIGHTS
    const Iq4Decode lookup(lut0, lut1);
#else
    (void)lut0;
    (void)lut1;
#endif
#if EXPERT_SKIP_EMPTY
    // A row tile whose codes are all zero (padding) adds exactly +0, so it is skipped.
    bool live[m / micro_m];
    for (unsigned row = 0; row < m / micro_m; ++row) {
        const auto* tile = a + row * micro_m * k;
        auto bits = aie::bit_or(aie::load_v<32>(tile), aie::load_v<32>(tile + 32));
        bits = aie::bit_or(bits, aie::bit_or(aie::load_v<32>(tile + 64), aie::load_v<32>(tile + 96)));
        live[row] = !aie::eq(bits, aie::zeros<int16_t, 32>()).full();
    }
#endif

#if EXPERT_ROW_UNROLL > 1
#pragma clang loop unroll_count(2)
#else
#pragma clang loop unroll(disable)
#endif
    for (unsigned col = 0; col < n / micro_n; ++col) {
        const auto scale = aie::load_v<micro_n>(scales + col * micro_n).grow_replicate<micro_m * micro_n>();
#if SPLIT_SCALE && !SPLIT_SCALE_HIGH_ONLY
        const auto scale_low = aie::load_v<micro_n>(scales + n + col * micro_n).grow_replicate<micro_m * micro_n>();
#endif
        aie::vector<int8_t, 64> b[k / 8];
#pragma clang loop unroll(full)
        for (unsigned octet = 0; octet < k / 8; ++octet) {
            const unsigned index = col * (k / 8) + octet;
#if IQ4_WEIGHTS
            b[octet] = lookup(codes + index * 32);
#else
            b[octet] = aie::load_v<64>(reinterpret_cast<const int8_t*>(codes) + index * 64);
#endif
        }
#if EXPERT_ROW_UNROLL
#pragma clang loop unroll(full)
#else
#pragma clang loop unroll(disable)
#endif
        for (unsigned row = 0; row < m / micro_m; ++row) {
#if EXPERT_SKIP_EMPTY
            if (!live[row]) {
                if (first)
                    aie::store_v(y + (row * (n / micro_n) + col) * micro_m * micro_n,
                                 aie::zeros<float, micro_m * micro_n>());
                continue;
            }
#endif
            const auto* a_tile = a + row * micro_m * k;
            Dot dot;
            dot.mul(aie::load_v<32>(a_tile), b[0]);
#pragma clang loop unroll(full)
            for (unsigned octet = 1; octet < k / 8; ++octet)
                dot.mac(aie::load_v<32>(a_tile + octet * 32), b[octet]);

            const Output part(aie::to_float<float>(dot.to_vector<int32_t>(), 0));
            const auto hi = part.to_vector<bfloat16>();
            const auto lo = aie::sub(part, hi).to_vector<bfloat16>();
            // Multiplication by the activation's power of two is exact for
            // normal BF16 scales and keeps the integer dot product unchanged.
            const auto step = aie::load_v<micro_m * micro_n>(rescale + row * micro_m * micro_n);
            const auto factor = aie::mul(step, scale).to_vector<bfloat16>();
            auto* dest = y + (row * (n / micro_n) + col) * micro_m * micro_n;
            Output out;
            if (first) out = aie::zeros<accfloat, micro_m * micro_n>();
            else out.from_vector(aie::load_v<micro_m * micro_n>(dest));
            out = aie::mac(out, hi, factor);
            out = aie::mac(out, lo, factor);
#if SPLIT_SCALE && !SPLIT_SCALE_HIGH_ONLY
            const auto correction = aie::mul(step, scale_low).to_vector<bfloat16>();
            out = aie::mac(out, hi, correction);
            out = aie::mac(out, lo, correction);
#endif
            aie::store_v(dest, out.to_vector<float>());
        }
    }
}
}

#ifndef INKLING2_EXPERT_IMPL_ONLY
extern "C" void inkling_block_expert(const uint8_t* __restrict a, const uint8_t* __restrict w,
                                     const int16_t* lut0, const int16_t* lut1,
                                     float* __restrict y, int32_t first) {
    apply(a, w, lut0, lut1, y, first != 0);
}
#endif
