#pragma once
// Two token rows through one q8 half-tile (gx_rows.h's twin for the attention GEMVs): one unzip and one mmul for both
// rows, the epilogue per token as gemv_q8_half_tile's. Table: gx_rows.h's 2-token layout.

#include "gemv_q8.h"

__attribute__((noinline)) inline void gx_q8_half_tile2(const uint8_t *__restrict tile, const uint8_t *__restrict tab,
                                                       unsigned K, unsigned kt, bool first, float *__restrict y0,
                                                       float *__restrict y1) {
  event0();
#ifdef GEMV_NULL
  if (first) {
    aie::store_v(y0, aie::zeros<float, kQ8Rows>());
    aie::store_v(y1, aie::zeros<float, kQ8Rows>());
  }
  event1();
  return;
#else
  aie::set_rounding(aie::rounding_mode::conv_even);
  const unsigned NB = K / 32;
  const bfloat16 *__restrict s = (const bfloat16 *)tile;
  const uint8_t *__restrict c0 = tile + kQ8ScaleBytes;
  const int16_t *__restrict xg = (const int16_t *)tab + kt * kQ8TileK * 2;
  const int32_t *__restrict sh[2];
  sh[0] = (const int32_t *)(tab + 4 * K) + kt * kQ8KBlocks;
  sh[1] = sh[0] + NB;

  aie::accum<accfloat, kQ8Rows> acc[2];
  if (first) {
#if GEMV_Q8_PAD_BIAS
    acc[0].from_vector(aie::load_v<kQ8Rows>((const float *)(tile + kQ8PadBias)));
    acc[1].from_vector(aie::load_v<kQ8Rows>((const float *)(tile + kQ8PadBias)));
#else
    acc[0] = aie::zeros<accfloat, kQ8Rows>();
    acc[1] = aie::zeros<accfloat, kQ8Rows>();
#endif
  } else {
    acc[0].from_vector(aie::load_v<kQ8Rows>(y0));
    acc[1].from_vector(aie::load_v<kQ8Rows>(y1));
  }

#pragma clang loop unroll(disable)
  for (unsigned kb = 0; kb < kQ8KBlocks; ++kb) {
    const uint8_t *__restrict src = c0 + kb * kQ8KInBlock * kQ8Rows;
    aie::mmul<4, 8, 8, int16_t, int8_t> Clo, Chi;
#pragma clang loop unroll(full)
    for (unsigned oc = 0; oc < 4; ++oc) {
      const aie::vector<int16_t, 32> A =
          aie::concat(aie::load_v<16>(xg + (kb * 4 + oc) * 16), aie::zeros<int16_t, 16>());
      const aie::vector<int8_t, 64> q0 = aie::load_v<64>((const int8_t *)src + oc * 128);
      const aie::vector<int8_t, 64> q1 = aie::load_v<64>((const int8_t *)src + oc * 128 + 64);
      auto [lo, hi] = aie::interleave_unzip(q0, q1, 8);
      if (oc == 0) {
        Clo.mul(A, lo);
        Chi.mul(A, hi);
      } else {
        Clo.mac(A, lo);
        Chi.mac(A, hi);
      }
    }
    const aie::vector<int32_t, 32> Clov = Clo.template to_vector<int32_t>();
    const aie::vector<int32_t, 32> Chiv = Chi.template to_vector<int32_t>();
    const aie::vector<bfloat16, kQ8Rows> sv = aie::load_v<kQ8Rows>(s + kb * kQ8Rows);
#pragma clang loop unroll(full)
    for (unsigned t = 0; t < 2; ++t) {
      const aie::vector<int32_t, kQ8Rows> vi = aie::concat(Clov.template extract<8>(t), Chiv.template extract<8>(t));
      aie::accum<accfloat, kQ8Rows> part;
      part.from_vector(aie::to_float<float>(vi, sh[t][kb]));
      const aie::vector<bfloat16, kQ8Rows> hi = part.template to_vector<bfloat16>();
      const aie::vector<bfloat16, kQ8Rows> lo = aie::sub(part, hi).template to_vector<bfloat16>();
      acc[t] = aie::mac(acc[t], hi, sv);
      acc[t] = aie::mac(acc[t], lo, sv);
    }
  }

  aie::store_v(y0, acc[0].template to_vector<float>());
  aie::store_v(y1, acc[1].template to_vector<float>());
  event1();
#endif
}

static inline void gx_q8_walk2(const uint8_t *__restrict chunks, const uint8_t *__restrict tab, unsigned group,
                               float *__restrict y0, float *__restrict y1, unsigned per_band, unsigned rs) {
  const unsigned K = kQ8TileK * per_band / rs;
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < kQ8PerCall; ++i) {
    const unsigned c = group * kQ8PerCall + i;
    const unsigned part = c % rs;
    const unsigned kt = c / rs;
    gx_q8_half_tile2(chunks + i * kQ8TileBytes, tab, K, kt, kt == 0, y0 + part * kQ8Rows, y1 + part * kQ8Rows);
  }
}
