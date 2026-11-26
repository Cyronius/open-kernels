// Inkling 2 loop controller for 8 token rows a core. The GGUF's F32 controller weights are all bf16
// values, so they stay exact; F32 activations are split into bf16 hi + lo halves.
#include <aie_api/aie.hpp>

#include "vecmath.h"

#ifdef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
#error "the controller needs the native bf16 product"
#endif

namespace {
using Dot = aie::mmul<4, 8, 8, bfloat16, bfloat16>;
constexpr unsigned frame_cols = 64;   // x then m of 32 hidden indices, 8 rows
constexpr unsigned tile = 32;         // [4 rows x 8]

v32f gather(const float* frame, unsigned rt, unsigned kt) {
    const float* p = frame + rt * 4 * frame_cols + kt * 8;
    return aie::concat(aie::load_v<8>(p), aie::load_v<8>(p + frame_cols),
                       aie::load_v<8>(p + 2 * frame_cols), aie::load_v<8>(p + 3 * frame_cols));
}

void mac2(Dot& dot, const bfloat16* a, const bfloat16* b) {
    const auto w = aie::load_v<64>(b);
    dot.mac(aie::load_v<32>(a), w);
    dot.mac(aie::load_v<32>(a + 512), w);
}

v32f rsqrt(const v32f& a) {
    const auto bits = a.cast_to<int32_t>();
    v32f y = aie::sub(aie::broadcast<int32_t, 32>(0x5f3759df), aie::downshift(bits, 1)).cast_to<float>();
    const v32f half = fmulN<32>(a, aie::broadcast<float, 32>(0.5f));
    const v32f three_halves = aie::broadcast<float, 32>(1.5f);
#pragma clang loop unroll(disable)
    for (unsigned i = 0; i < 4; ++i)
        y = fmulN<32>(y, fsubN<32>(three_halves, fmulN<32>(fmulN<32>(half, y), y)));
    return y;
}

v32f vtanh(const v32f& v) {
    const v32f s = vsigmoidN<32>(faddN<32>(v, v));
    return fsubN<32>(faddN<32>(s, s), aie::broadcast<float, 32>(1.0f));
}

// out tiles [rt][h tile] = z [8 rows x 64] times one [64 x 32] weight plane.
void project(const bfloat16* zs, const bfloat16* w, float* out) {
#pragma clang loop unroll(disable)
    for (unsigned rt = 0; rt < 2; ++rt)
#pragma clang loop unroll(disable)
        for (unsigned ht = 0; ht < 4; ++ht) {
            Dot dot(aie::zeros<float, tile>());
#pragma clang loop unroll(disable)
            for (unsigned zt = 0; zt < 8; ++zt)
                mac2(dot, zs + (rt * 8 + zt) * tile, w + (zt * 4 + ht) * 64);
            aie::store_v(out + (rt * 4 + ht) * tile, dot.to_vector<float>());
        }
}
} // namespace

// Phase 1, one frame: z_x and z_m partial sums and the squares of x and m.
extern "C" void inkling_ctrl_down(const float* __restrict frame, const uint8_t* __restrict weights,
                                  float* __restrict zx, float* __restrict zm, float* __restrict squares,
                                  bfloat16* __restrict tiles, int32_t index) {
#ifdef CONTROLLER_NULL
    return;
#endif
    aie::set_rounding(aie::rounding_mode::conv_even);
    const auto* w = reinterpret_cast<const bfloat16*>(weights);
    if (index == 0) {
        for (unsigned i = 0; i < 512; i += tile) {
            aie::store_v(zx + i, aie::zeros<float, tile>());
            aie::store_v(zm + i, aie::zeros<float, tile>());
        }
        for (unsigned i = 0; i < 128; i += tile) aie::store_v(squares + i, aie::zeros<float, tile>());
    }
#pragma clang loop unroll(disable)
    for (unsigned rt = 0; rt < 2; ++rt)
#pragma clang loop unroll(disable)
        for (unsigned kt = 0; kt < 8; ++kt) {
            v32b hi, lo;
            split32(gather(frame, rt, kt), hi, lo);
            aie::store_v(tiles + (rt * 8 + kt) * tile, hi);
            aie::store_v(tiles + 512 + (rt * 8 + kt) * tile, lo);
            float* sum = squares + ((kt >> 2) * 2 + rt) * tile;
            accf32 acc;
            acc.from_vector(aie::load_v<tile>(sum));
            acc = aie::mac(acc, hi, hi);
            acc = aie::mac(acc, hi, lo);
            acc = aie::mac(acc, hi, lo);
            aie::store_v(sum, acc.to_vector<float>());
        }
#pragma clang loop unroll(disable)
    for (unsigned rt = 0; rt < 2; ++rt)
#pragma clang loop unroll(disable)
        for (unsigned zt = 0; zt < 8; ++zt) {
            float* px = zx + (rt * 8 + zt) * tile;
            float* pm = zm + (rt * 8 + zt) * tile;
            Dot dx(aie::load_v<tile>(px)), dm(aie::load_v<tile>(pm));
#pragma clang loop unroll(disable)
            for (unsigned kt = 0; kt < 4; ++kt) {
                const bfloat16* a = tiles + (rt * 8 + kt) * tile;
                mac2(dx, a, w + (kt * 8 + zt) * 64);
                mac2(dm, a + 4 * tile, w + ((kt + 4) * 8 + zt) * 64);
            }
            aie::store_v(px, dx.to_vector<float>());
            aie::store_v(pm, dm.to_vector<float>());
        }
}

// z = z_x / rms(x) + z_m / rms(m), as bf16 hi and lo tiles for phase 2.
extern "C" void inkling_ctrl_scale(const float* __restrict zx, const float* __restrict zm,
                                   const float* __restrict squares, bfloat16* __restrict zs,
                                   const int32_t* __restrict params) {
#ifdef CONTROLLER_NULL
    return;
#endif
    aie::set_rounding(aie::rounding_mode::conv_even);
    const v32f eps = aie::broadcast<int32_t, 32>(params[1]).cast_to<float>();
    const v32f inverse = aie::broadcast<float, 32>(1.0f / 2304.0f);
    const auto ones = aie::broadcast<bfloat16, 64>(bfloat16(1.0f));
#pragma clang loop unroll(disable)
    for (unsigned rt = 0; rt < 2; ++rt) {
        v32f scale[2];
#pragma clang loop unroll(disable)
        for (unsigned part = 0; part < 2; ++part) {
            v32b hi, lo;
            split32(aie::load_v<tile>(squares + (part * 2 + rt) * tile), hi, lo);
            // A ones matrix adds each row's 8 lanes and keeps the [4 rows x 8] layout.
            Dot dot;
            dot.mul(hi, ones);
            dot.mac(lo, ones);
            scale[part] = rsqrt(faddN<32>(fmulN<32>(dot.to_vector<float>(), inverse), eps));
        }
#pragma clang loop unroll(disable)
        for (unsigned zt = 0; zt < 8; ++zt) {
            const unsigned at = (rt * 8 + zt) * tile;
            const v32f z = faddN<32>(fmulN<32>(aie::load_v<tile>(zx + at), scale[0]),
                                     fmulN<32>(aie::load_v<tile>(zm + at), scale[1]));
            v32b hi, lo;
            split32(z, hi, lo);
            aie::store_v(zs + at, hi);
            aie::store_v(zs + 512 + at, lo);
        }
    }
}

// Phase 2, one chunk of 32 hidden indices: u = x + tanh(g) m - tanh(f) x in row-major [8 rows x 32].
extern "C" void inkling_ctrl_gate(const bfloat16* __restrict zs, const uint8_t* __restrict weights,
                                  float* __restrict g, float* __restrict f,
                                  const float* __restrict frame, float* __restrict y) {
#ifdef CONTROLLER_NULL
    return;
#endif
    aie::set_rounding(aie::rounding_mode::conv_even);
    const auto* w = reinterpret_cast<const bfloat16*>(weights);
    project(zs, w, g);
    project(zs, w + 2048, f);
#pragma clang loop unroll(disable)
    for (unsigned rt = 0; rt < 2; ++rt)
#pragma clang loop unroll(disable)
        for (unsigned ht = 0; ht < 4; ++ht) {
            const unsigned at = (rt * 4 + ht) * tile;
            const v32f x = gather(frame, rt, ht), m = gather(frame, rt, ht + 4);
            const v32f u = faddN<32>(x, fsubN<32>(fmulN<32>(vtanh(aie::load_v<tile>(g + at)), m),
                                                  fmulN<32>(vtanh(aie::load_v<tile>(f + at)), x)));
            float* row = y + rt * 4 * 32 + ht * 8;
            aie::store_v(row, u.extract<8>(0));
            aie::store_v(row + 32, u.extract<8>(1));
            aie::store_v(row + 64, u.extract<8>(2));
            aie::store_v(row + 96, u.extract<8>(3));
        }
}
