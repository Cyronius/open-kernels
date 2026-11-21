#pragma once
// Two token rows through one mx4 half-tile: one nibble decode and one mmul for both rows (A rows 0 and 1 hold the
// tokens, rows 2 and 3 are zero), the epilogue per token as gemv_mx4_half_tile's, so each row is bit-identical to decode.
// 2-token table: int16 xi[K/32][4 oc][2 tokens][8] | int32 sh[2][K/32]

#include "gemv_mx4.h"

static constexpr unsigned gx_tab2_bytes(unsigned K) { return 4 * K + 8 * (K / 32); }

template <bool F32>
__attribute__((noinline)) inline void gx_prep2_tok(const void *__restrict xp, uint8_t *__restrict tab, unsigned K,
                                                   unsigned tok, unsigned b0, unsigned nb) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const unsigned NB = K / 32;
  int16_t *__restrict xg = (int16_t *)tab;
  int32_t *__restrict sh = (int32_t *)(tab + 4 * K + tok * 4 * NB);
  const aie::vector<uint8_t, 64> absmask = gemv_q4_absmask();
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < nb; ++j) {
    aie::vector<bfloat16, 32> xv;
    if constexpr (F32) {
      aie::accum<accfloat, 32> a;
      a.from_vector(aie::load_v<32>((const float *)xp + j * 32));
      xv = a.template to_vector<bfloat16>();
    } else {
      xv = aie::load_v<32>((const bfloat16 *)xp + j * 32);
    }
    const unsigned kb = b0 + j;
    const aie::vector<int16_t, 32> ab =
        aie::bit_and(xv.template cast_to<uint8_t>(), absmask).template cast_to<int16_t>();
    const int mx = aie::reduce_max(ab);
    int s = 141 - (mx >> 7);
    if (s > 126) s = 126;
    sh[kb] = s;
    const aie::vector<bfloat16, 32> sc =
        aie::broadcast<int16_t, 32>((int16_t)((127 + s) << 7)).template cast_to<bfloat16>();
    const aie::accum<accfloat, 32> a = aie::mul(xv, sc);
    const aie::vector<int16_t, 32> r = aie::to_fixed<int16_t>(a, 0);
#pragma clang loop unroll(full)
    for (unsigned oc = 0; oc < 4; ++oc) aie::store_v(xg + (kb * 4 + oc) * 16 + tok * 8, r.template extract<8>(oc));
  }
}

__attribute__((noinline)) inline void gx_half_tile2(const uint8_t *__restrict tile, const uint8_t *__restrict tab,
                                                    unsigned K, unsigned kt, bool first, float *__restrict y0,
                                                    float *__restrict y1) {
  event0();
#ifdef GEMV_NULL
  if (first) {
    aie::store_v(y0, aie::zeros<float, kMx4Rows>());
    aie::store_v(y1, aie::zeros<float, kMx4Rows>());
  }
  event1();
  return;
#else
  aie::set_rounding(aie::rounding_mode::conv_even);
  const unsigned NB = K / 32;
  const int16_t *__restrict xg = (const int16_t *)tab + kt * kMx4TileK * 2;
  const int32_t *__restrict sh[2];
  sh[0] = (const int32_t *)(tab + 4 * K) + kt * kMx4KBlocks;
  sh[1] = sh[0] + NB;

  aie::accum<accfloat, kMx4Rows> acc[2];
  if (first) {
#if GEMV_Q8_PAD_BIAS
    acc[0].from_vector(aie::load_v<kMx4Rows>((const float *)(tile + kMx4PadBias)));
    acc[1].from_vector(aie::load_v<kMx4Rows>((const float *)(tile + kMx4PadBias)));
#else
    acc[0] = aie::zeros<accfloat, kMx4Rows>();
    acc[1] = aie::zeros<accfloat, kMx4Rows>();
#endif
  } else {
    acc[0].from_vector(aie::load_v<kMx4Rows>(y0));
    acc[1].from_vector(aie::load_v<kMx4Rows>(y1));
  }

#pragma clang loop unroll_count(2)
  for (unsigned kb = 0; kb < kMx4KBlocks; ++kb) {
    const uint8_t *__restrict grp = tile + (kb / 2) * kMx4GroupBytes;
    const uint8_t *__restrict src = grp + 64 + (kb % 2) * 256;
    aie::mmul<4, 8, 8, int16_t, int8_t> Clo, Chi;
#pragma clang loop unroll(full)
    for (unsigned oc = 0; oc < 4; ++oc) {
      const aie::vector<int16_t, 32> A =
          aie::concat(aie::load_v<16>(xg + (kb * 4 + oc) * 16), aie::zeros<int16_t, 16>());
      const aie::vector<int8_t, 64> lo = mx4_codes(src + oc * 64);
      const aie::vector<int8_t, 64> hi = mx4_codes(src + oc * 64 + 32);
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
    const aie::vector<bfloat16, kMx4Rows> sv = aie::load_v<kMx4Rows>((const bfloat16 *)grp + (kb % 2) * kMx4Rows);
#pragma clang loop unroll(full)
    for (unsigned t = 0; t < 2; ++t) {
      const aie::vector<int32_t, kMx4Rows> vi = aie::concat(Clov.template extract<8>(t), Chiv.template extract<8>(t));
      aie::accum<accfloat, kMx4Rows> part;
      part.from_vector(aie::to_float<float>(vi, sh[t][kb]));
      const aie::vector<bfloat16, kMx4Rows> hi = part.template to_vector<bfloat16>();
      const aie::vector<bfloat16, kMx4Rows> lo = aie::sub(part, hi).template to_vector<bfloat16>();
      acc[t] = aie::mac(acc[t], hi, sv);
      acc[t] = aie::mac(acc[t], lo, sv);
    }
  }

  aie::store_v(y0, acc[0].template to_vector<float>());
  aie::store_v(y1, acc[1].template to_vector<float>());
  event1();
#endif
}

// gemv_mx4_pool_group_rt / _acc for two rows: y0 and y1 are the rows' accumulators.
static inline void gx_mx4_walk2(const uint8_t *__restrict chunks, const uint8_t *__restrict tab, unsigned group,
                                float *__restrict y0, float *__restrict y1, unsigned per_band, bool fresh) {
  const unsigned K = kMx4TileK * per_band / kMx4RowSplit;
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < kMx4PerCall; ++i) {
    const unsigned c = group * kMx4PerCall + i;
    const unsigned part = c % kMx4RowSplit;
    const unsigned kt = c / kMx4RowSplit;
    gx_half_tile2(chunks + i * kMx4TileBytes, tab, K, kt, fresh && kt == 0, y0 + part * kMx4Rows,
                  y1 + part * kMx4Rows);
  }
}
