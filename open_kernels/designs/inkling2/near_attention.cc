// Inkling 2 near-range block attention: 32 query rows per core, 16-key chunks, relative bias from per-core strips.
// The output tile is O^T [128 dims][32 rows], then the row maxima and masses.
// Scores are in log2 units (the host folds log2(e) into Q and the relative query), as in attn_fa.

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

constexpr int TQ = 32, LKP = 16, HD = 128, RD = 16, STRIP = 48, EXTENT = 1024;
constexpr int KEY_BYTES = LKP * HD * 2, STRIP_BYTES = RD * STRIP * 2;
constexpr int QWORDS = (HD + RD) * TQ;
constexpr int ST_M = 0, ST_L = 32, ST_I = 64;
constexpr int TAU = 16;
constexpr float NEG = -1.0e30f;
#ifndef NA_EXP_PRECISE
#define NA_EXP_PRECISE 1
#endif
#ifndef NA_ABL
#define NA_ABL 0  // timing ablation only: 1 QK, 2 bias, 4 softmax, 8 PV
#endif

using vf = aie::vector<float, 32>;
using vb = aie::vector<bfloat16, 32>;
using vi = aie::vector<int32_t, 32>;
using af32 = aie::accum<accfloat, 32>;

namespace {

inline vf fsub(vf a, vf b) {
  af32 x, y;
  x.from_vector(a);
  y.from_vector(b);
  return aie::sub(x, y).template to_vector<float>();
}

inline void split(vf v, vb &h, vb &l) {
  af32 x;
  x.from_vector(v);
  h = x.template to_vector<bfloat16>();
  l = aie::sub(x, h).template to_vector<bfloat16>();
}

inline vb to_bf(vf v) {
  af32 a;
  a.from_vector(v);
  return a.template to_vector<bfloat16>();
}

inline vf fmul_hl(vf a, vb bh, vb bl) {
  vb ah, al;
  split(a, ah, al);
  af32 r = aie::mul(ah, bh);
  r = aie::mac(r, ah, bl);
  r = aie::mac(r, al, bh);
  return r.template to_vector<float>();
}

// 2^x for x <= ~TAU, as attn_fa's exp2v.
inline vf exp2v(vf x) {
  x = aie::max(x, aie::broadcast<float, 32>(-127.0f));
  x = aie::min(x, aie::broadcast<float, 32>(64.0f));
  const vi n = aie::to_fixed<int32_t>(x, 0);
  const vf f = fsub(x, aie::to_float<float>(n, 0));
  vb fh, fl;
  split(f, fh, fl);
  af32 t;
  t.from_vector(aie::broadcast<float, 32>(0.0555041087f));
  t = aie::mac(t, fh, aie::broadcast<bfloat16, 32>((bfloat16)0.0096181291f));
  const vb t3 = t.template to_vector<bfloat16>();
  t.from_vector(aie::broadcast<float, 32>(0.2402265070f));
  t = aie::mac(t, t3, fh);
#if NA_EXP_PRECISE
  // Keep the quadratic term in two BF16 limbs; a single limb costs ~1e-4 relative error.
  t = aie::mac(t, t3, fl);
  const vb t2 = t.template to_vector<bfloat16>();
  const vb t2l = aie::sub(t, t2).template to_vector<bfloat16>();
  t.from_vector(aie::broadcast<float, 32>(0.6931471806f));
  t = aie::mac(t, t2, fh);
  t = aie::mac(t, t2, fl);
  t = aie::mac(t, t2l, fh);
#else
  const vb t2 = t.template to_vector<bfloat16>();
  t.from_vector(aie::broadcast<float, 32>(0.6931471806f));
  t = aie::mac(t, t2, fh);
#endif
  const vb t1h = t.template to_vector<bfloat16>();
  const vb t1l = aie::sub(t, t1h).template to_vector<bfloat16>();
  af32 p;
  p.from_vector(aie::broadcast<float, 32>(1.0f));
  p = aie::mac(p, t1h, fh);
  p = aie::mac(p, t1h, fl);
  p = aie::mac(p, t1l, fh);
  vi bits = p.template to_vector<float>().template cast_to<int32_t>();
  bits = aie::add(bits, aie::upshift(n, 23));
  bits = aie::select(bits, aie::zeros<int32_t, 32>(), aie::lt(n, aie::broadcast<int32_t, 32>(-125)));
  return bits.template cast_to<float>();
}

alignas(64) const int32_t iota[32] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                      16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

// Meta words: first query position, key position of chunk 0, window (0 = full), valid rows, lowest visible key.
struct Meta {
  int32_t pos, kbase, window, rows, low;
};

#ifndef NA_QK_KEYS
#define NA_QK_KEYS 2
#endif
#ifndef NA_PV_DIMS
#define NA_PV_DIMS 4
#endif

// S^T[j] = K[j] . Q^T with Q as hi + lo. Each key value is a scalar broadcast against a row vector.
void qk_t(const bfloat16 *__restrict qh, const bfloat16 *__restrict ql, const bfloat16 *__restrict k,
          float *__restrict s) {
  constexpr int N = NA_QK_KEYS;
  for (int j = 0; j < LKP; j += N) {
    af32 a[N], c[N];
#pragma clang loop unroll(full)
    for (int u = 0; u < N; u++) {
      a[u] = aie::zeros<accfloat, 32>();
      c[u] = aie::zeros<accfloat, 32>();
    }
#pragma clang loop unroll_count(2)
    for (int d = 0; d < HD; d++) {
      const vb q = aie::load_v<32>(qh + d * TQ), qlo = aie::load_v<32>(ql + d * TQ);
#pragma clang loop unroll(full)
      for (int u = 0; u < N; u++) {
        const vb b = aie::broadcast<bfloat16, 32>(k[(j + u) * HD + d]);
        a[u] = aie::mac(a[u], b, q);
        c[u] = aie::mac(c[u], b, qlo);
      }
    }
#pragma clang loop unroll(full)
    for (int u = 0; u < N; u++)
      aie::store_v(s + (j + u) * TQ, aie::add(a[u], c[u]).template to_vector<float>());
  }
}

// Lane r of key j uses strip column 15 - j + r, which is distance (q0 + r) - (k0 + j).
void bias_t(const bfloat16 *__restrict qh, const bfloat16 *__restrict ql, const bfloat16 *__restrict strip,
            float *__restrict s) {
  for (int j = 0; j < LKP; j += 2) {
    af32 a0, a1, c0 = aie::zeros<accfloat, 32>(), c1 = aie::zeros<accfloat, 32>();
    a0.from_vector(aie::load_v<32>(s + j * TQ));
    a1.from_vector(aie::load_v<32>(s + (j + 1) * TQ));
#pragma clang loop unroll_count(2)
    for (int d = 0; d < RD; d++) {
      const bfloat16 *row = strip + d * STRIP;
      const vb lo = aie::concat(aie::load_v<16>(row), aie::load_v<16>(row + 16));
      const auto tail = aie::load_v<16>(row + 32);
      const vb hi = aie::concat(tail, tail);
      const vb t0 = vb(::shift_bytes((v32bfloat16)lo, (v32bfloat16)hi, (LKP - 1 - j) * 2));
      const vb t1 = vb(::shift_bytes((v32bfloat16)lo, (v32bfloat16)hi, (LKP - 2 - j) * 2));
      const vb rh = aie::load_v<32>(qh + (HD + d) * TQ), rl = aie::load_v<32>(ql + (HD + d) * TQ);
      a0 = aie::mac(a0, t0, rh);
      c0 = aie::mac(c0, t0, rl);
      a1 = aie::mac(a1, t1, rh);
      c1 = aie::mac(c1, t1, rl);
    }
    aie::store_v(s + j * TQ, aie::add(a0, c0).template to_vector<float>());
    aie::store_v(s + (j + 1) * TQ, aie::add(a1, c1).template to_vector<float>());
  }
}

// O^T[d] += sum_j V[j][d] (P hi[j] + P lo[j]); each key slot holds P hi then P lo as row vectors.
void pv_t(const bfloat16 *__restrict p, const bfloat16 *__restrict v, float *__restrict o) {
  constexpr int N = NA_PV_DIMS;
  for (int d = 0; d < HD; d += N) {
    af32 a[N];
#pragma clang loop unroll(full)
    for (int u = 0; u < N; u++)
      a[u].from_vector(aie::load_v<32>(o + (d + u) * TQ));
#pragma clang loop unroll_count(2)
    for (int j = 0; j < LKP; j++) {
      const vb ph = aie::load_v<32>(p + j * 2 * TQ), pl = aie::load_v<32>(p + j * 2 * TQ + TQ);
#pragma clang loop unroll(full)
      for (int u = 0; u < N; u++) {
        const vb b = aie::broadcast<bfloat16, 32>(v[j * HD + d + u]);
        a[u] = aie::mac(a[u], ph, b);
        a[u] = aie::mac(a[u], pl, b);
      }
    }
#pragma clang loop unroll(full)
    for (int u = 0; u < N; u++)
      aie::store_v(o + (d + u) * TQ, a[u].template to_vector<float>());
  }
}

}  // namespace

extern "C" {

void na_capture(const uint8_t *__restrict obj, bfloat16 *__restrict dst, float *__restrict st, int32_t meta) {
  const bfloat16 *src = reinterpret_cast<const bfloat16 *>(obj);
  for (int i = 0; i < QWORDS; i += 32)
    aie::store_v(dst + i, aie::load_v<32>(src + i));
  if (meta) {
    const int32_t *m = reinterpret_cast<const int32_t *>(obj + QWORDS * 2);
    int32_t *si = reinterpret_cast<int32_t *>(st + ST_I);
    aie::store_v(si, aie::load_v<16>(m));
  }
}

void na_init(float *__restrict o, float *__restrict st) {
  const vf z = aie::zeros<float, 32>();
  for (int i = 0; i < TQ * HD; i += 32)
    aie::store_v(o + i, z);
  aie::store_v(st + ST_M, aie::broadcast<float, 32>(NEG));
  aie::store_v(st + ST_L, z);
}

// Status in si[8]: 0 skip, 1 every key visible to every row, 2 per-element mask.
void na_qk(bfloat16 *__restrict qh, bfloat16 *__restrict ql, uint8_t *__restrict a, uint8_t *__restrict b,
           float *__restrict s, float *__restrict st, int32_t core, int32_t chunk) {
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  int32_t *si = reinterpret_cast<int32_t *>(st + ST_I);
  const Meta m{si[0], si[1], si[2], si[3], si[4]};
  const int32_t q0 = m.pos + core * TQ, ql_ = q0 + TQ - 1;
  const int32_t k0 = m.kbase + chunk * LKP, kl = k0 + LKP - 1;
  int32_t status;
  if (core * TQ >= m.rows || k0 > ql_ || kl < m.low || (m.window && kl <= q0 - m.window))
    status = 0;
  else if (k0 >= m.low && kl <= q0 && (!m.window || k0 > ql_ - m.window))
    status = 1;
  else
    status = 2;
  si[8] = status;
  if (!status)
    return;
  if (!(NA_ABL & 1)) qk_t(qh, ql, reinterpret_cast<const bfloat16 *>(a), s);
  if (!(NA_ABL & 2) && q0 - kl < EXTENT) {
    const uint8_t *frame = core < 4 ? a : b;
    bias_t(qh, ql, reinterpret_cast<const bfloat16 *>(frame + KEY_BYTES + (core & 3) * STRIP_BYTES), s);
  }
}

void na_sm(float *s, float *__restrict st, float *__restrict o, int32_t core, int32_t chunk) {
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  int32_t *si = reinterpret_cast<int32_t *>(st + ST_I);
  const int32_t status = si[8];
  if (!status || (NA_ABL & 4))
    return;
  const vf neg = aie::broadcast<float, 32>(NEG);
  if (status == 2) {
    const int32_t window = si[2];
    const vi q = aie::add(aie::load_v<32>(iota), aie::broadcast<int32_t, 32>(si[0] + core * TQ));
    const int32_t k0 = si[1] + chunk * LKP;
    for (int j = 0; j < LKP; j++) {
      const int32_t key = k0 + j;
      float *ps = s + j * TQ;
      auto hidden = aie::lt(q, aie::broadcast<int32_t, 32>(key));
      if (window)
        hidden = hidden | aie::ge(q, aie::broadcast<int32_t, 32>(key + window));
      vf row = aie::select(aie::load_v<32>(ps), neg, hidden);
      if (key < si[4])
        row = neg;
      aie::store_v(ps, row);
    }
  }
  vf m0 = aie::load_v<32>(s), m1 = aie::load_v<32>(s + TQ), m2 = aie::load_v<32>(s + 2 * TQ),
     m3 = aie::load_v<32>(s + 3 * TQ);
  for (int j = 4; j < LKP; j += 4) {
    m0 = aie::max(m0, aie::load_v<32>(s + j * TQ));
    m1 = aie::max(m1, aie::load_v<32>(s + (j + 1) * TQ));
    m2 = aie::max(m2, aie::load_v<32>(s + (j + 2) * TQ));
    m3 = aie::max(m3, aie::load_v<32>(s + (j + 3) * TQ));
  }
  const vf mx = aie::max(aie::max(m0, m1), aie::max(m2, m3));
  const vf m = aie::load_v<32>(st + ST_M);
  af32 mt;
  mt.from_vector(m);
  const auto up = aie::gt(mx, aie::add(mt, aie::broadcast<float, 32>(float(TAU))).template to_vector<float>());
  if (!up.empty()) {
    const vf mn = aie::select(m, aie::to_float<float>(aie::to_fixed<int32_t>(mx, 0), 0), up);
    const vb fac = to_bf(exp2v(fsub(m, mn)));
    const vb zero = aie::zeros<bfloat16, 32>();
    for (int d = 0; d < HD; d++)
      aie::store_v(o + d * TQ, fmul_hl(aie::load_v<32>(o + d * TQ), fac, zero));
    aie::store_v(st + ST_L, fmul_hl(aie::load_v<32>(st + ST_L), fac, zero));
    aie::store_v(st + ST_M, mn);
  }
  const vf mm = aie::load_v<32>(st + ST_M);
  const vf live = aie::broadcast<float, 32>(NEG * 0.5f);
  af32 l, l2 = aie::zeros<accfloat, 32>(), l3 = aie::zeros<accfloat, 32>(), l4 = aie::zeros<accfloat, 32>();
  l.from_vector(aie::load_v<32>(st + ST_L));
  const vb one = aie::broadcast<bfloat16, 32>((bfloat16)1.0f);
  const vf zero = aie::zeros<float, 32>();
  bfloat16 *p = reinterpret_cast<bfloat16 *>(s);
  // Two keys per iteration so their exponential chains overlap.
  for (int j = 0; j < LKP; j += 2) {
    const vf x0 = aie::load_v<32>(s + j * TQ), x1 = aie::load_v<32>(s + (j + 1) * TQ);
    const vf p0 = aie::select(zero, exp2v(fsub(x0, mm)), aie::gt(x0, live));
    const vf p1 = aie::select(zero, exp2v(fsub(x1, mm)), aie::gt(x1, live));
    vb h0, o0, h1, o1;
    split(p0, h0, o0);
    split(p1, h1, o1);
    aie::store_v(p + j * 2 * TQ, h0);
    aie::store_v(p + j * 2 * TQ + TQ, o0);
    aie::store_v(p + (j + 1) * 2 * TQ, h1);
    aie::store_v(p + (j + 1) * 2 * TQ + TQ, o1);
    l = aie::mac(l, h0, one);
    l2 = aie::mac(l2, o0, one);
    l3 = aie::mac(l3, h1, one);
    l4 = aie::mac(l4, o1, one);
  }
  aie::store_v(st + ST_L, aie::add(aie::add(l, l2), aie::add(l3, l4)).template to_vector<float>());
}

void na_pv(float *__restrict s, uint8_t *__restrict b, float *__restrict o, float *__restrict st) {
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  const int32_t *si = reinterpret_cast<const int32_t *>(st + ST_I);
  if (!si[8] || (NA_ABL & 8))
    return;
  pv_t(reinterpret_cast<const bfloat16 *>(s), reinterpret_cast<const bfloat16 *>(b), o);
}

// The tile ends with the row maxima (log2 units) and the masses.
void na_store(float *__restrict o, float *__restrict st) {
  aie::store_v(o + TQ * HD, aie::load_v<32>(st + ST_M));
  aie::store_v(o + TQ * HD + TQ, aie::load_v<32>(st + ST_L));
}

}  // extern "C"
