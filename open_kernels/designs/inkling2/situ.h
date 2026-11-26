#pragma once
// Inkling SiTU and the down-input row pack, shared by the singleton and grouped expert kernels so that
// verify rows and decode rows get the same bits. The caller sets conv_even rounding.
#include "vecmath.h"

namespace inkling_situ {

__attribute__((noinline)) inline v32f mul32(v32f a, v32f b) { return fmulN<32>(a, b); }

// One row of one K32 tile: gate and up each hold 4 x 8 values at a 32-float stride. Returns the 32 int16
// codes in lane order and sets the row's BF16 power-of-two scale.
inline aie::vector<int16_t, 32> row(const float* gate, const float* up, int16_t& scale) {
    v32f g, u;
#pragma clang loop unroll(full)
    for (unsigned c = 0; c < 4; ++c) {
        g.insert(c, aie::load_v<8>(gate + 32 * c));
        u.insert(c, aie::load_v<8>(up + 32 * c));
    }
    const auto exponent = aie::broadcast<int32_t, 32>(0x7f800000);
    const int32_t inputs = aie::reduce_max(aie::max(aie::bit_and(g.cast_to<int32_t>(), exponent),
                                                    aie::bit_and(u.cast_to<int32_t>(), exponent)));
    // 4 tanh(g/4) sigmoid(g) 25 tanh(u/25) = 100 (1-e)(1-f) / ((1+e)(1+e^2)(1+f)); the clamps keep it finite.
    const auto one = aie::broadcast<float, 32>(1.0f);
    g = aie::max(g, aie::broadcast<float, 32>(-30.0f));
    u = aie::min(aie::max(u, aie::broadcast<float, 32>(-500.0f)), aie::broadcast<float, 32>(500.0f));
    const v32f e = vexpN<32>(mul32(g, aie::broadcast<float, 32>(-0.5f)));
    const v32f f = vexpN<32>(mul32(u, aie::broadcast<float, 32>(-0.08f)));
    const v32f d = mul32(mul32(faddN<32>(one, e), faddN<32>(one, mul32(e, e))), faddN<32>(one, f));
    // Bit-pattern seed for 1/d, then Newton steps (the hardware inv expands to 32 scalar ops).
    v32f r = aie::sub(aie::broadcast<int32_t, 32>(0x7ef311c7), d.cast_to<int32_t>()).cast_to<float>();
#pragma clang loop unroll(disable)
    for (unsigned i = 0; i < 3; ++i) r = mul32(r, fsubN<32>(aie::broadcast<float, 32>(2.0f), mul32(d, r)));
    const v32f n = mul32(mul32(fsubN<32>(one, e), fsubN<32>(one, f)), aie::broadcast<float, 32>(100.0f));
    aie::accum<accfloat, 32> h;
    h.from_vector(mul32(n, r));
    const auto rounded = h.to_vector<bfloat16>();
    const int top = aie::reduce_max(aie::bit_and(rounded.cast_to<uint16_t>(),
                                                 aie::broadcast<uint16_t, 32>(0x7f80))) >> 7;
    const int shift = 141 - top < 126 ? 141 - top : 126;
    // A NaN scale makes the down output non-finite, which the host rejects.
    scale = inputs == 0x7f800000 || top == 255 ? 0x7fc0 : static_cast<int16_t>((127 - shift) << 7);
    return aie::to_fixed<int16_t>(rounded, shift);
}

}  // namespace inkling_situ
