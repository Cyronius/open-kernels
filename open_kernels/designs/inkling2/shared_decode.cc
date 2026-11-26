#ifdef INKLING2_SINGLETON
#include "iq4_lut.h"
#include "vecmath.h"

// Both expert formats and the dense projections share one worker image.
#define INKLING2_EXPERT_IMPL_ONLY
#define IQ4_WEIGHTS 1
namespace iq4 {
#include "block_expert.cc"
}
#undef IQ4_WEIGHTS
#undef SPLIT_SCALE
#define IQ4_WEIGHTS 0
#define SPLIT_SCALE 1
namespace q8 {
#include "block_expert.cc"
}
#else
#include "block_expert.cc"
#endif

namespace {
constexpr unsigned activation_bytes = 320;
constexpr unsigned activation_group = 4;
#ifdef INKLING2_SINGLETON
constexpr unsigned weight_bytes = 96 * 36;
#else
constexpr unsigned weight_bytes = 96 * (IQ4_WEIGHTS ? 18 : 34);
#endif
constexpr unsigned query_bytes = 9344;
constexpr unsigned result_values = 2080;
constexpr unsigned frame_values = 384;

void copy_bytes(const uint8_t* src, uint8_t* dst, unsigned bytes) {
    for (unsigned i = 0; i < bytes; i += 64) {
        const auto x = aie::load_v<32>(reinterpret_cast<const uint16_t*>(src + i));
        aie::store_v(reinterpret_cast<uint16_t*>(dst + i), x);
    }
}
} // namespace

#ifdef INKLING2_SINGLETON
// flags bit 0: first record of a band; bit 1: IQ4_NL, whose record holds two K tiles; bit 2: write to `scratch`.
extern "C" void inkling_shared_project(const uint8_t* __restrict a, const uint8_t* __restrict w,
                                       const int16_t* lut0, const int16_t* lut1, float* y, float* scratch,
                                       int32_t record, int32_t flags) {
    const bool first = flags & 1;
    if (flags & 4) y = scratch;
    if (flags & 2) {
        a += 2 * record * activation_bytes;
        iq4::apply(a, w, lut0, lut1, y, first);
        iq4::apply(a + activation_bytes, w + weight_bytes / 2, lut0, lut1, y, false);
    } else {
        q8::apply(a + record * activation_bytes, w, lut0, lut1, y, first);
    }
}

namespace {
__attribute__((noinline)) v32f mul32(v32f a, v32f b) { return fmulN<32>(a, b); }
} // namespace

// Row 0 of K32 tile `t` of a gate band and its up band (4 x 8 values at a 32-float stride) to one
// 320-byte down-input tile written over `frame`. The tile's up values are loaded before its wire is
// stored, and the wire never reaches a later tile's up values.
extern "C" void inkling_shared_situ(const float* gate, float* frame, int32_t t) {
    aie::set_rounding(aie::rounding_mode::conv_even);
    gate += 128 * t;
    const float* up = frame + 128 * t;
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
    const int16_t scale = inputs == 0x7f800000 || top == 255 ? 0x7fc0 : static_cast<int16_t>((127 - shift) << 7);
    const auto codes = aie::to_fixed<int16_t>(rounded, shift);
    auto* tile = reinterpret_cast<int16_t*>(frame) + 160 * t;
    auto scales = aie::broadcast<int16_t, 32>(128);
    scales.insert(0, aie::broadcast<int16_t, 8>(scale));
    aie::store_v(tile, scales);
#pragma clang loop unroll(full)
    for (unsigned c = 0; c < 4; ++c) {
        auto lane = aie::zeros<int16_t, 32>();
        lane.insert(0, codes.extract<8>(c));
        aie::store_v(tile + 32 + 32 * c, lane);
    }
}
#else
extern "C" void inkling_shared_expert(const uint8_t* __restrict a, const uint8_t* __restrict w,
                                      const int16_t* lut0, const int16_t* lut1, float* __restrict y,
                                      int32_t within, int32_t first) {
    inkling_block_expert(a + within * activation_bytes, w, lut0, lut1, y, first);
}
#endif

extern "C" void inkling_shared_query(const uint8_t* __restrict src, uint8_t* __restrict query, int32_t frame) {
    const unsigned at = frame * weight_bytes;
    const unsigned remaining = query_bytes - at;
    copy_bytes(src, query + at, remaining < weight_bytes ? remaining : weight_bytes);
}

extern "C" void inkling_shared_kv(const uint8_t* __restrict src, uint8_t* __restrict kv, int32_t frame) {
    constexpr unsigned bytes = activation_group * activation_bytes;
    copy_bytes(src, kv + frame * bytes, bytes);
}

extern "C" void inkling_shared_output(const float* __restrict result, float* __restrict out, int32_t frame) {
    const unsigned at = frame * frame_values;
    const unsigned remaining = result_values - at;
    const unsigned copy = remaining < frame_values ? remaining : frame_values;
    for (unsigned i = 0; i < copy; i += 32) aie::store_v(out + i, aie::load_v<32>(result + at + i));
    for (unsigned i = copy; i < frame_values; i += 32) aie::store_v(out + i, aie::zeros<float, 32>());
}

extern "C" void inkling_shared_zero(float* out) {
    for (unsigned i = 0; i < frame_values; i += 32) aie::store_v(out + i, aie::zeros<float, 32>());
}
