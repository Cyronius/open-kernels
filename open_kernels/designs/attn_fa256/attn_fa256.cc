//===- attn_fa256.cc ------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Core kernels of attn_fa256.py: causal flash attention for a 256-token block prefill at head dim 256
// (Qwen3.6-35B-A3B's full-attention layers). One core owns TQ = 16 query rows and walks every key chunk
// of the window, so the online softmax never leaves the core (dit_fa's loop). Differences from dit_fa:
//   - Scores and O are fp32 (bf16 O loses a chunk's contribution once O is 2^8 times larger, which a
//     window of thousands of chunks reaches).
//   - The host folds log2(e)/16 into Q, so S is in log2 units. The running max is an integer, so a
//     rescale is an exact power of two; 2^x is attn_fa's polynomial (aie2p's bf16 exp2 is 6% off).
//   - The causal bound comes from the pass's record (p0, the block's first position), as attn_fa.
//
// Layouts: 8x8 blocks, column-block major ([col block][row block][8][8]) for every mmul operand.
//   Q     [8 d chunks][4 col blocks][2 row blocks][8][8] bf16      K, V pieces [4][8 key blocks][8][8] bf16
//   G     [8 key blocks][2 row blocks][8][8] fp32 (P overwrites it in place as bf16, same block order)
//   O     [32 col blocks][2 row blocks][8][8] fp32
//   st    m [TQ*8] | l [TQ*8] fp32, each row's value in its 8 lanes ("replicated") | p0, status (int32)
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

#ifndef AF_TAU
#define AF_TAU 16
#endif
#ifndef AF_ABL
#define AF_ABL 0   // ablation (timing only): 1 skips S = QK, 2 the softmax, 4 O += PV
#endif

constexpr int TQ = 16;
constexpr int LKP = 64;
constexpr int DC = 32;
constexpr int DFULL = 256;
constexpr int NDCH = DFULL / DC;
constexpr int REP = TQ * 8;                  // floats in a replicated per-row vector set
constexpr int ST_M = 0, ST_L = REP, ST_I = 2 * REP;
constexpr int NV = TQ / 4;                   // 32-lane vectors a replicated set holds
constexpr float NEG = -1.0e30f;

using vf = aie::vector<float, 32>;
using vb = aie::vector<bfloat16, 32>;
using vi = aie::vector<int32_t, 32>;
using af32 = aie::accum<accfloat, 32>;

#define SET_ROUNDING() ::aie::set_rounding(::aie::rounding_mode::conv_even)

static inline vf fsub(vf a, vf b) {
  af32 x, y;
  x.from_vector(a);
  y.from_vector(b);
  return aie::sub(x, y).template to_vector<float>();
}

static inline vf fadd(vf a, vf b) {
  af32 x, y;
  x.from_vector(a);
  y.from_vector(b);
  return aie::add(x, y).template to_vector<float>();
}

static inline void split(vf v, vb &h, vb &l) {
  af32 x;
  x.from_vector(v);
  h = x.template to_vector<bfloat16>();
  l = aie::sub(x, h).template to_vector<bfloat16>();
}

static inline vb to_bf(vf v) {
  af32 a;
  a.from_vector(v);
  return a.template to_vector<bfloat16>();
}

// a (fp32) * b (bf16 hi / lo): the three significant cross terms (~2^-16)
static inline vf fmul_hl(vf a, vb bh, vb bl) {
  vb ah, al;
  split(a, ah, al);
  af32 r = aie::mul(ah, bh);
  r = aie::mac(r, ah, bl);
  r = aie::mac(r, al, bh);
  return r.template to_vector<float>();
}

static inline vf fmul(vf a, vf b) {
  vb bh, bl;
  split(b, bh, bl);
  return fmul_hl(a, bh, bl);
}

// attn_fa's exp2c: 2^x for x <= ~TAU, n = round(x) by the fp32 magic number, 2^f a degree-3 polynomial
// with p(0) = 1 (integers are exact), n added to the exponent field
#define AF_C1 0.69314718f
#define AF_C2 0.24022650f
#define AF_C3 0.05550411f
static inline vf exp2f32(vf x) {
  constexpr float kMagic = 12582912.0f;
  x = aie::max(x, aie::broadcast<float, 32>(-126.0f));
  af32 t;
  t.from_vector(x);
  t = aie::add(t, aie::broadcast<float, 32>(kMagic));
  const vf tv = t.template to_vector<float>();
  const vi n = aie::sub(tv.template cast_to<int32_t>(), aie::broadcast<int32_t, 32>(0x4B400000));
  af32 nf;
  nf.from_vector(tv);
  nf = aie::sub(nf, aie::broadcast<float, 32>(kMagic));
  af32 fa;
  fa.from_vector(x);
  fa = aie::sub(fa, nf.template to_vector<float>());
  const vb f = fa.template to_vector<bfloat16>();
  af32 u;
  u.from_vector(aie::broadcast<float, 32>(AF_C2));
  u = aie::mac(u, f, aie::broadcast<bfloat16, 32>((bfloat16)AF_C3));
  af32 w;
  w.from_vector(aie::broadcast<float, 32>(AF_C1));
  w = aie::mac(w, u.template to_vector<bfloat16>(), f);
  af32 p;
  p.from_vector(aie::broadcast<float, 32>(1.0f));
  p = aie::mac(p, w.template to_vector<bfloat16>(), f);
  const vi bits = aie::add(p.template to_vector<float>().template cast_to<int32_t>(), aie::upshift(n, 23));
  return bits.template cast_to<float>();
}

// max over each group of 8 lanes, left in all 8 of them (a row of an 8x8 block, replicated)
static inline vf row8_max(vf m) {
  for (int s = 1; s < 8; s <<= 1) {
    const auto z = aie::interleave_unzip(m, m, s);
    const vf r = aie::max(z.first, z.second);   // r[0..15] hold the pairwise maxima of step s
    const auto b = aie::interleave_zip(r, r, s);
    m = b.first;
  }
  return m;
}

// C (+)= A B over 2 row blocks (TQ = 16), aie::mmul<8,8,8> (bfp16 emulation), C fp32. transpose_b: B is K
// ([k block][n block] blocks, each [n][k]); else V ([n block][k block], each [k][n]).
template <unsigned colA, unsigned colB, bool transpose_b, bool zero>
static inline void mm16(const bfloat16 *__restrict pA, const bfloat16 *__restrict pB, float *__restrict pC) {
  using MMUL = aie::mmul<8, 8, 8, bfloat16, bfloat16, accauto>;
  constexpr unsigned rowA = 2;
  for (unsigned j = 0; j < colB; j += 2) {
    float *__restrict c00 = pC + j * rowA * MMUL::size_C;
    float *__restrict c01 = c00 + rowA * MMUL::size_C;
    MMUL C00, C01, C10, C11;
    if constexpr (!zero) {
      C00 = MMUL(aie::load_v<MMUL::size_C>(c00));
      C10 = MMUL(aie::load_v<MMUL::size_C>(c00 + MMUL::size_C));
      C01 = MMUL(aie::load_v<MMUL::size_C>(c01));
      C11 = MMUL(aie::load_v<MMUL::size_C>(c01 + MMUL::size_C));
    }
    for (unsigned i = 0; i < colA; ++i) {
      const aie::vector<bfloat16, 64> A0 = aie::load_v<64>(pA + (i * rowA) * MMUL::size_A);
      const aie::vector<bfloat16, 64> A1 = aie::load_v<64>(pA + (i * rowA + 1) * MMUL::size_A);
      aie::vector<bfloat16, 64> B0, B1;
      if constexpr (transpose_b) {
        B0 = aie::transpose(aie::load_v<64>(pB + (i * colB + j) * MMUL::size_B), 8, 8);
        B1 = aie::transpose(aie::load_v<64>(pB + (i * colB + j + 1) * MMUL::size_B), 8, 8);
      } else {
        B0 = aie::load_v<64>(pB + (j * colA + i) * MMUL::size_B);
        B1 = aie::load_v<64>(pB + ((j + 1) * colA + i) * MMUL::size_B);
      }
      if (zero && i == 0) {
        C00.mul(A0, B0);
        C01.mul(A0, B1);
        C10.mul(A1, B0);
        C11.mul(A1, B1);
      } else {
        C00.mac(A0, B0);
        C01.mac(A0, B1);
        C10.mac(A1, B0);
        C11.mac(A1, B1);
      }
    }
    aie::store_v(c00, C00.template to_vector<float>());
    aie::store_v(c00 + MMUL::size_C, C10.template to_vector<float>());
    aie::store_v(c01, C01.template to_vector<float>());
    aie::store_v(c01 + MMUL::size_C, C11.template to_vector<float>());
  }
}

// lane l of a 32-lane slice of an 8x8 block: key (l % 8) minus row (l / 8)
alignas(64) static const int32_t k_lane[32] = {0,  1,  2,  3,  4,  5,  6,  7,  -1, 0,  1,  2,  3,  4,  5,  6,
                                               -2, -1, 0,  1,  2,  3,  4,  5,  -3, -2, -1, 0,  1,  2,  3,  4};

extern "C" {

// the pass's record, the first words of the first Q element: p0 = the block's first position
void af_meta(bfloat16 *__restrict e, float *__restrict st) {
  int32_t *si = reinterpret_cast<int32_t *>(st + ST_I);
  si[0] = reinterpret_cast<const int32_t *>(e)[0];
}

// A Q element is one d chunk of 4 cores' rows, [4 col blocks][8 row blocks][8][8]; keep this core's 2 row blocks
void af_capture(bfloat16 *__restrict src, bfloat16 *__restrict q, int32_t dch, int32_t sub) {
  for (int cb = 0; cb < DC / 8; cb++) {
    const bfloat16 *__restrict ps = src + (cb * 8 + 2 * sub) * 64;
    bfloat16 *__restrict pd = q + dch * (TQ * DC) + cb * 128;
    for (int j = 0; j < 128; j += 32)
      aie::store_v(pd + j, aie::load_v<32>(ps + j));
  }
}

void af_init(float *__restrict o, float *__restrict st) {
  const vf z = aie::zeros<float, 32>();
  for (int i = 0; i < TQ * DFULL; i += 32)
    aie::store_v(o + i, z);
  for (int v = 0; v < NV; v++) {
    aie::store_v(st + ST_M + v * 32, aie::broadcast<float, 32>(NEG));
    aie::store_v(st + ST_L + v * 32, z);
  }
}

// G (+)= Q[:, d chunk] K_piece^T. At d = 0 it decides the chunk's status: 0 = every key past every row's
// position (skip), 1 = all visible, 2 = ragged.
void af_qk(bfloat16 *__restrict q, bfloat16 *__restrict k, float *__restrict g, float *__restrict st, int32_t d,
           int32_t tok0, int32_t chunk) {
  SET_ROUNDING();
  int32_t *si = reinterpret_cast<int32_t *>(st + ST_I);
  if (d == 0) {
    const int32_t first = si[0] + tok0;   // the last key row 0 sees
    const int32_t k0 = chunk * LKP;
    si[1] = k0 > first + TQ - 1 ? 0 : (k0 + LKP - 1 <= first ? 1 : 2);
  }
  if (si[1] == 0 || (AF_ABL & 1))
    return;
  if (d == 0)
    mm16<DC / 8, LKP / 8, true, true>(q, k, g);
  else
    mm16<DC / 8, LKP / 8, true, false>(q + d * (TQ * DC), k, g);
}

// mask, lazy rescale, P = 2^(G - m) in place as bf16, l += rowsum(P) on the MAC array
void af_sm(float *g, float *__restrict st, float *__restrict o, int32_t tok0, int32_t chunk) {
  SET_ROUNDING();
  int32_t *si = reinterpret_cast<int32_t *>(st + ST_I);
  const int32_t status = si[1];
  if (status == 0 || (AF_ABL & 2))
    return;
  if (status == 2) {
    // key k0 + j is hidden from row r when j - r > p0 + tok0 - k0
    const vi lane = aie::load_v<32>(k_lane);
    const int32_t lim = si[0] + tok0 - chunk * LKP;
    for (int kb = 0; kb < LKP / 8; kb++)
      for (int v = 0; v < NV; v++) {
        float *pg = g + kb * REP + v * 32;
        const vi jr = aie::add(lane, aie::broadcast<int32_t, 32>(kb * 8 - v * 4));
        aie::store_v(pg, aie::select(aie::load_v<32>(pg), aie::broadcast<float, 32>(NEG),
                                     aie::gt(jr, aie::broadcast<int32_t, 32>(lim))));
      }
  }
  vf mx[NV];
  bool up_any = false;
  for (int v = 0; v < NV; v++) {
    vf m0 = aie::load_v<32>(g + v * 32), m1 = aie::load_v<32>(g + REP + v * 32);
    for (int kb = 2; kb < LKP / 8; kb += 2) {
      m0 = aie::max(m0, aie::load_v<32>(g + kb * REP + v * 32));
      m1 = aie::max(m1, aie::load_v<32>(g + (kb + 1) * REP + v * 32));
    }
    mx[v] = row8_max(aie::max(m0, m1));
    const vf mt = fadd(aie::load_v<32>(st + ST_M + v * 32), aie::broadcast<float, 32>(float(AF_TAU)));
    up_any = up_any || !aie::gt(mx[v], mt).empty();
  }
  if (up_any) {
    // m moves to round(row max) where a row passed m + TAU; O and l scale by the exact 2^(m - m')
    const vb zero = aie::zeros<bfloat16, 32>();
    for (int v = 0; v < NV; v++) {
      const vf m = aie::load_v<32>(st + ST_M + v * 32);
      const auto up = aie::gt(mx[v], fadd(m, aie::broadcast<float, 32>(float(AF_TAU))));
      const vf mn = aie::select(m, aie::to_float<float>(aie::to_fixed<int32_t>(mx[v], 0), 0), up);
      const vb fac = to_bf(exp2f32(fsub(m, mn)));
      for (int cb = 0; cb < DFULL / 8; cb++) {
        float *po = o + cb * REP + v * 32;
        aie::store_v(po, fmul_hl(aie::load_v<32>(po), fac, zero));
      }
      aie::store_v(st + ST_L + v * 32, fmul_hl(aie::load_v<32>(st + ST_L + v * 32), fac, zero));
      aie::store_v(st + ST_M + v * 32, mn);
    }
  }
  // in place: P's bf16 index i is G's float index i, so a write never passes the read
  bfloat16 *p = reinterpret_cast<bfloat16 *>(g);
  for (int kb = 0; kb < LKP / 8; kb++)
    for (int v = 0; v < NV; v++) {
      const int i = kb * REP + v * 32;
      aie::store_v(p + i, to_bf(exp2f32(fsub(aie::load_v<32>(g + i), aie::load_v<32>(st + ST_M + v * 32)))));
    }
  // l += P @ ones: the mmul sees P in bfp16 exactly as the PV product does
  using MMUL = aie::mmul<8, 8, 8, bfloat16, bfloat16, accauto>;
  const aie::vector<bfloat16, 64> ones = aie::broadcast<bfloat16, 64>((bfloat16)1.0f);
  for (int z = 0; z < TQ / 8; z++) {
    MMUL C;
    C.mul(aie::load_v<64>(p + z * 64), ones);
    for (int i = 1; i < LKP / 8; i++)
      C.mac(aie::load_v<64>(p + (z + i * (TQ / 8)) * 64), ones);
    const aie::vector<float, 64> sv = C.template to_vector<float>();
    float *pl = st + ST_L + z * 64;
    aie::store_v(pl, fadd(aie::load_v<32>(pl), sv.extract<32>(0)));
    aie::store_v(pl + 32, fadd(aie::load_v<32>(pl + 32), sv.extract<32>(1)));
  }
}

// O[:, d chunk] += P V_piece
void af_pv(float *__restrict g, bfloat16 *__restrict v, float *__restrict o, float *__restrict st, int32_t d) {
  SET_ROUNDING();
  const int32_t *si = reinterpret_cast<const int32_t *>(st + ST_I);
  if (si[1] == 0 || (AF_ABL & 4))
    return;
  mm16<LKP / 8, DC / 8, false, false>(reinterpret_cast<const bfloat16 *>(g), v, o + d * (TQ * DC));
}

// O /= l
void af_fin(float *__restrict o, float *__restrict st) {
  SET_ROUNDING();
  const vf two = aie::broadcast<float, 32>(2.0f);
  for (int v = 0; v < NV; v++) {
    const vf l = aie::load_v<32>(st + ST_L + v * 32);
    // hardware seed, two Newton steps
    vf r = aie::inv(l);
    r = fmul(r, fsub(two, fmul(l, r)));
    r = fmul(r, fsub(two, fmul(l, r)));
    vb rh, rl;
    split(r, rh, rl);
    for (int cb = 0; cb < DFULL / 8; cb++) {
      float *po = o + cb * REP + v * 32;
      aie::store_v(po, fmul_hl(aie::load_v<32>(po), rh, rl));
    }
  }
}

} // extern "C"
