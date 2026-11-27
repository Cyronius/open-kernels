#pragma once
//===- lm_head_q8.h ----------------------------------------*- C++ -*-===//
//
// W8A16 GEMV for the lm_head from phlegm's POOL-ORDER q8 chunks:
//   logits[248320] = W[248320, 2048] @ x[2048]
//
// Inner arithmetic ported from vegah/LLMNpuTest designs/lm_head (Apache-2.0,
// ../../LICENSE.LLMNpuTest); the chunk layout is OFLM's q8 (q4nx.rs):
//
//   chunk = 32 output rows x 256 K, 8704 B:
//     scales[256] bf16 at [0   : 512]     index kb*32 + r          (r = 0..31)
//     codes [8192] int8 at [512 : 8704]   index (r/16)*4096 + k*16 + (r%16)
//   value = code * scale
//
// so `load_v<32>(s + kb*32)` is the 32 output rows in order, and two 16-lane
// code loads 4096 apart concatenate to those rows at one k: one 32-lane MAC
// per activation element, no gather.
//
// POOL ORDER (pools.rs build_lmhead_pool, "128-row supertile transpose"):
//   pool chunk k <- file chunk (4*(k/32) + (k%4))*8 + ((k%32)/4), i.e.
//   band = 32 consecutive chunks = 128 output rows x 2048 K;
//   chunk c inside its band: row quarter = c % 4, k-tile = c / 4.
// One band is consumed with a 128-float accumulator; the runtime `group`
// argument walks (quarter, kt) off the chunk index.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "gemv_tab.h"

static constexpr unsigned kRows = 32;        // output rows per chunk
static constexpr unsigned kKBlocks = 8;      // 32-wide K blocks per chunk
static constexpr unsigned kKInBlock = 32;
static constexpr unsigned kTileK = 256;      // K per chunk

// Chunk layout selector (GGUF-direct pools, open_kernels/gguf_pool.py), the
// lm_head twin of gemv_q4.h's GEMV_Q4_SCALES_F32:
//   0 (default, q4nx): 256 bf16 scales at [0:512), codes at [512:8704).
//   1 (GGUF-direct): the scales are the GGUF Q8_0 block scales widened
//     EXACTLY to f32 at [0:1024), codes at [1024:9216). The RNE accum
//     convert back to bf16 keeps the numerics identical to the q4nx chunks.
#ifndef LMHEAD_SCALES_F32
#define LMHEAD_SCALES_F32 0
#endif
#ifndef LMHEAD_Q6
#define LMHEAD_Q6 0
#endif
#if LMHEAD_Q6
// OPEN-LMHEAD-Q6: a GGUF Q6_K head at its own 6.6 bits a value (lm_head_q8.h's chunk geometry, 32 rows x 256 K):
//   d   [32]      f32  at [0    : 128)    the row's Q6_K super-block scale (one per 256 k = one per chunk)
//   sc  [16][32]  int8 at [128  : 640)    sub-block scales, index g*32 + r (g = 16-k group)
//   lo  [2][32][64] B  at [640  : 4736)   per row half rb, per (kb, oc) block of 128 codes in the q8 code order
//                                         (k_local*16 + r): low nibble = code j, high nibble = code 64 + j
//   hi  [2][32][32] B  at [4736 : 6784)   the codes' top two bits: byte b holds codes b, 32+b, 64+b, 96+b
// value = d[r] * sc[g][r] * (code - 32), code = lo | hi << 4 in [0, 63]
static constexpr unsigned kScaleBytes = 0;
static constexpr unsigned kTileBytes = 6784;
#elif LMHEAD_SCALES_F32
static constexpr unsigned kScaleBytes = 1024;
static constexpr unsigned kTileBytes = 9216;
#else
static constexpr unsigned kScaleBytes = 512;
static constexpr unsigned kTileBytes = 8704;
#endif
static constexpr unsigned kRowBlockStride = 4096;  // in codes
static constexpr unsigned kRowSplit = 4;     // 32-row quarters per 128-row band

// Entry-point symbol prefix (see gemv_q4.h): a GGUF-direct build is compiled
// with -DLMHEAD_Q8_PREFIX=lm_head_q8s32 (and LMHEAD_SCALES_F32=1).
#ifndef LMHEAD_Q8_PREFIX
#define LMHEAD_Q8_PREFIX lm_head_q8
#endif

#ifndef LMHEAD_PER_CALL
#define LMHEAD_PER_CALL 2
#endif
static constexpr unsigned kPerCall = LMHEAD_PER_CALL;

// The hidden width. It sizes the activation table this kernel indexes into (gemv_tab.h:
// 2K bytes of int16 x, then the per-block shifts at +2K), so it is a compile-time knob,
// not a runtime one. The default is the Qwen3.6-27B's; Qwen3.5 dense passes 4096.
#ifndef LMHEAD_K
#define LMHEAD_K 2048
#endif
#ifndef LMHEAD_ROWS
#define LMHEAD_ROWS 1
#endif

// Phase 2 item 5 (2026-09-02): the inner product on the integer matrix unit,
// like gemv_q4.h. vegah's form (int8 -> bf16 through the accumulator, bf16 x
// scalar MACs) ran at ~3 GB/s per core against a ~4 GB/s stream; this one is
// DMA-bound. Per 128 B (8 k x 16 rows, 16 B per k): an unzip at step 8 gives
// [8 k][rows 0..7] and [8 k][rows 8..15] -- two mmul<4,8,8,int16,int8> B
// operands -- against the x octet from the int16 table (gemv_tab.h). The
// four 8-row results assemble into the chunk's 32 rows IN ORDER (no
// permutation, unlike the q4 kernel), and per K block:
//   y[r] += scale[kb][r] * 2^-s[kb] * part[r]   (bf16 hi/lo split)
// Runtime kt/first, noinline + inline (COMDAT): one body in program memory.
#if LMHEAD_SCALES_F32
// f32 plane -> bf16 lanes, round-to-nearest-even in INTEGER ops ((u + 0x7FFF +
// ((u>>16) & 1)) >> 16, the same law q4nx_file.hpp's f32_to_bf16 uses). The
// accum-roundtrip float convert mis-schedules inside this kernel's kb loop.
static inline aie::vector<bfloat16, kRows> load_scale(const float *__restrict p) {
  const aie::vector<uint32_t, kRows> w = aie::load_v<kRows>((const uint32_t *)p);
  // The round-to-nearest-even bit is the retained BF16 LSB, i.e. bit 16 of the
  // f32 word, taken as 0/1. Masking with 0x10000 and adding that would bias the
  // sum by 65536 instead of 1, so downshift first and then mask.
  const aie::vector<uint32_t, kRows> lsb = aie::bit_and((uint32_t)1, aie::downshift(w, 16));
  const aie::vector<uint32_t, kRows> r = aie::add(w, aie::add((uint32_t)0x7FFF, lsb));
  return aie::pack(aie::downshift(r, 16)).template cast_to<bfloat16>();
}
#else
static inline aie::vector<bfloat16, kRows> load_scale(const bfloat16 *__restrict p) {
  return aie::load_v<kRows>(p);
}
#endif

static constexpr unsigned kK = LMHEAD_K;
__attribute__((noinline)) inline void gemv_q8_tile(const uint8_t *__restrict tile,
                                                   const uint8_t *__restrict tab,
                                                   unsigned kt, bool first,
                                                   float *__restrict y) {
  event0();
#ifdef LMHEAD_NULL
  // debug: arithmetic compiled out, every DMA kept (the gemv_q4.h GEMV_NULL twin), for
  // timing the head's stream floor; the logits are garbage.
  if (first)
    aie::store_v(y, aie::zeros<float, kRows>());
  event1();
  return;
#endif
  aie::set_rounding(aie::rounding_mode::conv_even);

#if LMHEAD_SCALES_F32
  const float *__restrict s = (const float *)tile;
#else
  const bfloat16 *__restrict s = (const bfloat16 *)tile;
#endif
  const uint8_t *__restrict c0 = tile + kScaleBytes;              // rows 0..15: byte k*16 + r
  const uint8_t *__restrict c1 = c0 + kRowBlockStride;            // rows 16..31
  const int16_t *__restrict xi = (const int16_t *)tab + kt * kTileK;
  const int32_t *__restrict sh = (const int32_t *)(tab + 2 * kK) + kt * kKBlocks;

  aie::accum<accfloat, kRows> acc;
  if (first)
    acc = aie::zeros<accfloat, kRows>();
  else
    acc.from_vector(aie::load_v<kRows>(y));

#pragma clang loop unroll(disable)
  for (unsigned kb = 0; kb < kKBlocks; ++kb) {
    aie::vector<int32_t, 8> v[4];
#pragma clang loop unroll(full)
    for (unsigned rb = 0; rb < 2; ++rb) {
      const uint8_t *__restrict src = (rb == 0 ? c0 : c1) + kb * kKInBlock * 16;
      aie::mmul<4, 8, 8, int16_t, int8_t> Clo, Chi;
#pragma clang loop unroll(full)
      for (unsigned oc = 0; oc < 4; ++oc) {
        const aie::vector<int16_t, 32> A =
            aie::load_v<8>(xi + kb * kKInBlock + oc * 8).template grow_replicate<32>();
        const aie::vector<int8_t, 64> q0 = aie::load_v<64>((const int8_t *)src + oc * 128);
        const aie::vector<int8_t, 64> q1 = aie::load_v<64>((const int8_t *)src + oc * 128 + 64);
        auto [lo, hi] = aie::interleave_unzip(q0, q1, 8);   // rows 0..7 / 8..15 at k 0..7
        if (oc == 0) {
          Clo.mul(A, lo);
          Chi.mul(A, hi);
        } else {
          Clo.mac(A, lo);
          Chi.mac(A, hi);
        }
      }
      v[2 * rb] = Clo.template to_vector<int32_t>().template extract<8>(0);
      v[2 * rb + 1] = Chi.template to_vector<int32_t>().template extract<8>(0);
    }
    const aie::vector<int32_t, kRows> vi = aie::concat(v[0], v[1], v[2], v[3]);   // rows 0..31
    aie::accum<accfloat, kRows> part;
    part.from_vector(aie::to_float<float>(vi, sh[kb]));
    const aie::vector<bfloat16, kRows> hi = part.template to_vector<bfloat16>();
    const aie::vector<bfloat16, kRows> lo = aie::sub(part, hi).template to_vector<bfloat16>();
    const aie::vector<bfloat16, kRows> sv = load_scale(s + kb * kRows);
    acc = aie::mac(acc, hi, sv);
    acc = aie::mac(acc, lo, sv);
  }

  aie::store_v(y, acc.template to_vector<float>());
  event1();
}

#if LMHEAD_Q6
// One 128-code block (8 k x 16 rows) of the Q6 chunk as the two signed 64-code halves the q8 kernel loads.
static inline void q6_block(const uint8_t *__restrict lo, const uint8_t *__restrict hi, aie::vector<int8_t, 64> &q0,
                            aie::vector<int8_t, 64> &q1) {
  const aie::vector<uint8_t, 64> n = aie::load_v<64>(lo);
  const aie::vector<uint8_t, 32> h = aie::load_v<32>(hi);
  const aie::vector<uint8_t, 64> h01 = aie::concat(h, aie::downshift(h, 2));
  const aie::vector<uint8_t, 64> h23 = aie::concat(aie::downshift(h, 4), aie::downshift(h, 6));
  const aie::vector<uint8_t, 64> c0 =
      aie::bit_or(aie::bit_and((uint8_t)15, n), aie::upshift(aie::bit_and((uint8_t)3, h01), 4));
  const aie::vector<uint8_t, 64> c1 = aie::bit_or(aie::downshift(n, 4), aie::upshift(aie::bit_and((uint8_t)3, h23), 4));
  q0 = aie::sub(c0.template cast_to<int8_t>(), (int8_t)32);
  q1 = aie::sub(c1.template cast_to<int8_t>(), (int8_t)32);
}

static inline aie::vector<bfloat16, kRows> q6_scale(const int8_t *__restrict p) {
  const aie::vector<int32_t, kRows> v = aie::unpack(aie::unpack(aie::load_v<kRows>(p)));
  aie::accum<accfloat, kRows> a;
  a.from_vector(aie::to_float<float>(v, 0));
  return a.template to_vector<bfloat16>();     // |sc| <= 128: exact in bf16
}

__attribute__((noinline)) inline void gemv_q6_tile(const uint8_t *__restrict tile, const uint8_t *__restrict tab,
                                                   unsigned kt, bool first, float *__restrict y) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *__restrict d = (const float *)tile;
  const int8_t *__restrict sc = (const int8_t *)(tile + 128);
  const uint8_t *__restrict lo = tile + 640;
  const uint8_t *__restrict hi = tile + 4736;
  const int16_t *__restrict xi = (const int16_t *)tab + kt * kTileK;
  const int32_t *__restrict sh = (const int32_t *)(tab + 2 * kK) + kt * kKBlocks;

  aie::accum<accfloat, kRows> acc = aie::zeros<accfloat, kRows>();   // this chunk, before d
#pragma clang loop unroll(disable)
  for (unsigned kb = 0; kb < kKBlocks; ++kb) {
    aie::vector<int32_t, 8> v[2][4];       // [16-k group][row octet]
#pragma clang loop unroll(full)
    for (unsigned rb = 0; rb < 2; ++rb) {
      aie::mmul<4, 8, 8, int16_t, int8_t> C[2][2];   // [group][rows 0..7 | 8..15]
#pragma clang loop unroll(full)
      for (unsigned oc = 0; oc < 4; ++oc) {
        const aie::vector<int16_t, 32> A =
            aie::load_v<8>(xi + kb * kKInBlock + oc * 8).template grow_replicate<32>();
        aie::vector<int8_t, 64> q0, q1;
        q6_block(lo + rb * 2048 + (kb * 4 + oc) * 64, hi + rb * 1024 + (kb * 4 + oc) * 32, q0, q1);
        auto [l8, h8] = aie::interleave_unzip(q0, q1, 8);
        const unsigned g = oc / 2;
        if (oc % 2 == 0) {
          C[g][0].mul(A, l8);
          C[g][1].mul(A, h8);
        } else {
          C[g][0].mac(A, l8);
          C[g][1].mac(A, h8);
        }
      }
#pragma clang loop unroll(full)
      for (unsigned g = 0; g < 2; ++g) {
        v[g][2 * rb] = C[g][0].template to_vector<int32_t>().template extract<8>(0);
        v[g][2 * rb + 1] = C[g][1].template to_vector<int32_t>().template extract<8>(0);
      }
    }
#pragma clang loop unroll(full)
    for (unsigned g = 0; g < 2; ++g) {
      const aie::vector<int32_t, kRows> vi = aie::concat(v[g][0], v[g][1], v[g][2], v[g][3]);
      aie::accum<accfloat, kRows> part;
      part.from_vector(aie::to_float<float>(vi, sh[kb]));
      const aie::vector<bfloat16, kRows> ph = part.template to_vector<bfloat16>();
      const aie::vector<bfloat16, kRows> pl = aie::sub(part, ph).template to_vector<bfloat16>();
      const aie::vector<bfloat16, kRows> sv = q6_scale(sc + (2 * kb + g) * kRows);
      acc = aie::mac(acc, ph, sv);
      acc = aie::mac(acc, pl, sv);
    }
  }
  // y += d * acc, both split hi / lo (the product to ~16 mantissa bits, as the q8 tile's scale step)
  const aie::vector<float, kRows> dv = aie::load_v<kRows>(d);
  aie::accum<accfloat, kRows> da;
  da.from_vector(dv);
  const aie::vector<bfloat16, kRows> dh = da.template to_vector<bfloat16>();
  const aie::vector<bfloat16, kRows> dl = aie::sub(da, dh).template to_vector<bfloat16>();
  const aie::vector<bfloat16, kRows> ah = acc.template to_vector<bfloat16>();
  const aie::vector<bfloat16, kRows> al = aie::sub(acc, ah).template to_vector<bfloat16>();
  aie::accum<accfloat, kRows> out;
  if (first)
    out = aie::zeros<accfloat, kRows>();
  else
    out.from_vector(aie::load_v<kRows>(y));
  out = aie::mac(out, ah, dh);
  out = aie::mac(out, ah, dl);
  out = aie::mac(out, al, dh);
  aie::store_v(y, out.template to_vector<float>());
  event1();
}
#define gemv_q8_tile gemv_q6_tile    // the entry points (lm_head_q8.cc) call the format's tile
#endif

#if LMHEAD_ROWS == 2
// Two token rows through one chunk: one unzip and one mmul for both (A rows 0 and 1 hold the tokens, 2 and 3 are
// zero), the epilogue per token as gemv_q8_tile's. 2-token table: gptoss/gx_rows.h's layout.
__attribute__((noinline)) inline void gemv_q8_tile2(const uint8_t *__restrict tile, const uint8_t *__restrict tab,
                                                    unsigned kt, bool first, float *__restrict y0,
                                                    float *__restrict y1) {
  event0();
#ifdef LMHEAD_NULL
  if (first) {
    aie::store_v(y0, aie::zeros<float, kRows>());
    aie::store_v(y1, aie::zeros<float, kRows>());
  }
  event1();
  return;
#endif
  aie::set_rounding(aie::rounding_mode::conv_even);
#if LMHEAD_SCALES_F32
  const float *__restrict s = (const float *)tile;
#else
  const bfloat16 *__restrict s = (const bfloat16 *)tile;
#endif
  const uint8_t *__restrict c0 = tile + kScaleBytes;
  const uint8_t *__restrict c1 = c0 + kRowBlockStride;
  const int16_t *__restrict xg = (const int16_t *)tab + kt * kTileK * 2;
  const int32_t *__restrict sh[2];
  sh[0] = (const int32_t *)(tab + 4 * kK) + kt * kKBlocks;
  sh[1] = sh[0] + kK / 32;

  aie::accum<accfloat, kRows> acc[2];
  if (first) {
    acc[0] = aie::zeros<accfloat, kRows>();
    acc[1] = aie::zeros<accfloat, kRows>();
  } else {
    acc[0].from_vector(aie::load_v<kRows>(y0));
    acc[1].from_vector(aie::load_v<kRows>(y1));
  }

#pragma clang loop unroll(disable)
  for (unsigned kb = 0; kb < kKBlocks; ++kb) {
    aie::vector<int32_t, 8> v[2][4];
#pragma clang loop unroll(full)
    for (unsigned rb = 0; rb < 2; ++rb) {
      const uint8_t *__restrict src = (rb == 0 ? c0 : c1) + kb * kKInBlock * 16;
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
#pragma clang loop unroll(full)
      for (unsigned t = 0; t < 2; ++t) {
        v[t][2 * rb] = Clov.template extract<8>(t);
        v[t][2 * rb + 1] = Chiv.template extract<8>(t);
      }
    }
    const aie::vector<bfloat16, kRows> sv = load_scale(s + kb * kRows);
#pragma clang loop unroll(full)
    for (unsigned t = 0; t < 2; ++t) {
      const aie::vector<int32_t, kRows> vi = aie::concat(v[t][0], v[t][1], v[t][2], v[t][3]);
      aie::accum<accfloat, kRows> part;
      part.from_vector(aie::to_float<float>(vi, sh[t][kb]));
      const aie::vector<bfloat16, kRows> hi = part.template to_vector<bfloat16>();
      const aie::vector<bfloat16, kRows> lo = aie::sub(part, hi).template to_vector<bfloat16>();
      acc[t] = aie::mac(acc[t], hi, sv);
      acc[t] = aie::mac(acc[t], lo, sv);
    }
  }
  aie::store_v(y0, acc[0].template to_vector<float>());
  aie::store_v(y1, acc[1].template to_vector<float>());
  event1();
}
#endif

#if LMHEAD_Q6 && LMHEAD_ROWS == 2
// The Q6 tile for two token rows (gemv_q8_tile2's form: A rows 0 and 1 hold the tokens), the verify pass's head.
__attribute__((noinline)) inline void gemv_q6_tile2(const uint8_t *__restrict tile, const uint8_t *__restrict tab,
                                                    unsigned kt, bool first, float *__restrict y0,
                                                    float *__restrict y1) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *__restrict d = (const float *)tile;
  const int8_t *__restrict sc = (const int8_t *)(tile + 128);
  const uint8_t *__restrict lo = tile + 640;
  const uint8_t *__restrict hi = tile + 4736;
  const int16_t *__restrict xg = (const int16_t *)tab + kt * kTileK * 2;
  const int32_t *__restrict sh[2];
  sh[0] = (const int32_t *)(tab + 4 * kK) + kt * kKBlocks;
  sh[1] = sh[0] + kK / 32;

  aie::accum<accfloat, kRows> acc[2] = {aie::zeros<accfloat, kRows>(), aie::zeros<accfloat, kRows>()};
#pragma clang loop unroll(disable)
  for (unsigned kb = 0; kb < kKBlocks; ++kb) {
    aie::vector<int32_t, 8> v[2][2][4];    // [token][16-k group][row octet]
#pragma clang loop unroll(full)
    for (unsigned rb = 0; rb < 2; ++rb) {
      aie::mmul<4, 8, 8, int16_t, int8_t> C[2][2];
#pragma clang loop unroll(full)
      for (unsigned oc = 0; oc < 4; ++oc) {
        const aie::vector<int16_t, 32> A =
            aie::concat(aie::load_v<16>(xg + (kb * 4 + oc) * 16), aie::zeros<int16_t, 16>());
        aie::vector<int8_t, 64> q0, q1;
        q6_block(lo + rb * 2048 + (kb * 4 + oc) * 64, hi + rb * 1024 + (kb * 4 + oc) * 32, q0, q1);
        auto [l8, h8] = aie::interleave_unzip(q0, q1, 8);
        const unsigned g = oc / 2;
        if (oc % 2 == 0) {
          C[g][0].mul(A, l8);
          C[g][1].mul(A, h8);
        } else {
          C[g][0].mac(A, l8);
          C[g][1].mac(A, h8);
        }
      }
#pragma clang loop unroll(full)
      for (unsigned g = 0; g < 2; ++g) {
        const aie::vector<int32_t, 32> cl = C[g][0].template to_vector<int32_t>();
        const aie::vector<int32_t, 32> ch = C[g][1].template to_vector<int32_t>();
#pragma clang loop unroll(full)
        for (unsigned t = 0; t < 2; ++t) {
          v[t][g][2 * rb] = cl.template extract<8>(t);
          v[t][g][2 * rb + 1] = ch.template extract<8>(t);
        }
      }
    }
#pragma clang loop unroll(full)
    for (unsigned g = 0; g < 2; ++g) {
      const aie::vector<bfloat16, kRows> sv = q6_scale(sc + (2 * kb + g) * kRows);
#pragma clang loop unroll(full)
      for (unsigned t = 0; t < 2; ++t) {
        const aie::vector<int32_t, kRows> vi = aie::concat(v[t][g][0], v[t][g][1], v[t][g][2], v[t][g][3]);
        aie::accum<accfloat, kRows> part;
        part.from_vector(aie::to_float<float>(vi, sh[t][kb]));
        const aie::vector<bfloat16, kRows> ph = part.template to_vector<bfloat16>();
        const aie::vector<bfloat16, kRows> pl = aie::sub(part, ph).template to_vector<bfloat16>();
        acc[t] = aie::mac(acc[t], ph, sv);
        acc[t] = aie::mac(acc[t], pl, sv);
      }
    }
  }
  aie::accum<accfloat, kRows> da;
  da.from_vector(aie::load_v<kRows>(d));
  const aie::vector<bfloat16, kRows> dh = da.template to_vector<bfloat16>();
  const aie::vector<bfloat16, kRows> dl = aie::sub(da, dh).template to_vector<bfloat16>();
  float *__restrict ys[2] = {y0, y1};
#pragma clang loop unroll(full)
  for (unsigned t = 0; t < 2; ++t) {
    const aie::vector<bfloat16, kRows> ah = acc[t].template to_vector<bfloat16>();
    const aie::vector<bfloat16, kRows> al = aie::sub(acc[t], ah).template to_vector<bfloat16>();
    aie::accum<accfloat, kRows> out;
    if (first)
      out = aie::zeros<accfloat, kRows>();
    else
      out.from_vector(aie::load_v<kRows>(ys[t]));
    out = aie::mac(out, ah, dh);
    out = aie::mac(out, ah, dl);
    out = aie::mac(out, al, dh);
    aie::store_v(ys[t], out.template to_vector<float>());
  }
  event1();
}
#define gemv_q8_tile2 gemv_q6_tile2
#endif
