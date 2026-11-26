#include "vecmath.h"

#include <stdint.h>

namespace {
constexpr unsigned tq = 16, tk = 32, hd = 128, rd = 16, strip = 64, group = 3;
constexpr unsigned query_words = (hd + rd) * tq;
constexpr int extent = 1024;
constexpr float masked = -1.0e30f;

using vi16 = aie::vector<int32_t, tq>;

alignas(64) const int32_t lanes[tq] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

// The singleton image runs the 16-lane exponentials on the 32-lane copy: same lanewise bits, one fewer function.
#ifdef INKLING2_EXP32
v16f exp16(const v16f& x) { return vexpN<32>(aie::concat(x, x)).extract<tq>(0); }
#else
v16f exp16(const v16f& x) { return vexpN<tq>(x); }
#endif

// 32-bit BF16 broadcast replicates pairs. Zip a vector with itself first.
v32b duplicate(const v16b& a) {
    const auto v = a.grow_replicate<32>();
    return aie::interleave_zip(v, v, 1).first;
}

v32b broadcast_lane(const v32b& doubled, unsigned lane) {
    return v32b(::broadcast_elem((v32bfloat16)doubled, lane));
}

v16f score(const bfloat16* __restrict qh, const bfloat16* __restrict ql,
           const bfloat16* __restrict k, const bfloat16* __restrict bh,
           const bfloat16* __restrict bl, unsigned key, bool relative) {
    accf16 a = aie::zeros<accfloat, tq>();
    for (unsigned base = 0; base < hd; base += tq) {
        const auto values = duplicate(aie::load_v<tq>(k + base));
#pragma clang loop unroll_count(4)
        for (unsigned lane = 0; lane < tq; ++lane) {
            const auto x = broadcast_lane(values, lane).extract<tq>(0);
            a = aie::mac(a, x, aie::load_v<tq>(qh + (base + lane) * tq));
            a = aie::mac(a, x, aie::load_v<tq>(ql + (base + lane) * tq));
        }
    }
    if (relative) {
        const unsigned off = tk - 1 - key;
        for (unsigned d = 0; d < rd; ++d) {
            const auto high = aie::shuffle_down(aie::load_v<strip>(bh + d * strip), off).extract<tq>(0);
            const auto low = aie::shuffle_down(aie::load_v<strip>(bl + d * strip), off).extract<tq>(0);
            const auto rh = aie::load_v<tq>(qh + (hd + d) * tq);
            const auto rl = aie::load_v<tq>(ql + (hd + d) * tq);
            a = aie::mac(a, rh, high);
            a = aie::mac(a, rl, high);
            a = aie::mac(a, rh, low);
        }
    }
    return a.to_vector<float>();
}
} // namespace

extern "C" void inkling_attn_init(const uint8_t* __restrict query, float* __restrict out) {
    aie::set_rounding(aie::rounding_mode::conv_even);
    const auto* info = reinterpret_cast<const int32_t*>(query);
    const bool sink = info[4] != 0;
    const auto* bias = reinterpret_cast<const float*>(query + 64);
    const auto zeros = aie::zeros<float, 32>();
    for (unsigned i = 0; i < tq * hd; i += 32) aie::store_v(out + i, zeros);
    aie::store_v(out + tq * hd, sink ? aie::load_v<tq>(bias) : aie::broadcast<float, tq>(masked));
    aie::store_v(out + tq * (hd + 1), aie::broadcast<float, tq>(sink ? 1.0f : 0.0f));
}

#ifdef INKLING2_ALIGNED_CHUNKS
namespace {
using v32f = aie::vector<float, 32>;
using acc32 = aie::accum<accfloat, 32>;

// Decode and verify rows (header word 6, at most 8; word 5 is the head within its group).
// K arrives transposed per chunk and the strip area holds FP32 relative bias per row and
// head with -1e30 on hidden keys, so keys are the vector lanes and no tile row is wasted.
__attribute__((noinline)) void decode_rows(const uint8_t* __restrict query, const uint8_t* __restrict kv,
                                           float* __restrict out, float* __restrict scratch,
                                           bfloat16* __restrict probs) {
    const auto* info = reinterpret_cast<const int32_t*>(query);
    const int32_t rows = info[6], head = info[5];
    const auto* qh = reinterpret_cast<const bfloat16*>(query + 128);
    const auto* ql = qh + query_words;
    const auto* kt = reinterpret_cast<const bfloat16*>(kv);
    const auto* v = kt + tk * hd;
    const auto* bias = reinterpret_cast<const float*>(v + tk * hd);
    const v32f hidden = aie::broadcast<float, 32>(-1.0e29f);
    for (int32_t r = 0; r < rows; ++r) {
        acc32 a = aie::zeros<accfloat, 32>(), c = aie::zeros<accfloat, 32>();
#pragma clang loop unroll_count(4)
        for (unsigned d = 0; d < hd; ++d) {
            const auto key = aie::load_v<32>(kt + d * tk);
            a = aie::mac(a, key, aie::broadcast<bfloat16, 32>(qh[d * tq + r]));
            c = aie::mac(c, key, aie::broadcast<bfloat16, 32>(ql[d * tq + r]));
        }
        acc32 sum = aie::add(a, c);
        acc32 b;
        b.from_vector(aie::load_v<32>(bias + (r * group + head) * tk));
        const v32f s = aie::add(sum, b).to_vector<float>();
        const v32f peak_old = aie::broadcast<float, 32>(out[tq * hd + r]);
        const v32f top = aie::broadcast<float, 32>(aie::reduce_max(s));
        // A chunk with no visible key leaves the row's state bit for bit.
        if (aie::le(top, hidden).test(0)) continue;
        const v32f peak = aie::max(peak_old, top);
        const v32f p = aie::select(aie::zeros<float, 32>(), vexpN<32>(fsubN<32>(s, peak)), aie::gt(s, hidden));
        const v32f keep = vexpN<32>(fsubN<32>(peak_old, peak));
        v16b ph0, pl0, ph1, pl1;
        split16(p.extract<16>(0), ph0, pl0);
        split16(p.extract<16>(1), ph1, pl1);
        aie::store_v(probs, ph0);
        aie::store_v(probs + 16, ph1);
        aie::store_v(probs + 32, pl0);
        aie::store_v(probs + 48, pl1);
        const auto ones = aie::broadcast<bfloat16, 32>((bfloat16)1.0f);
        acc32 m = aie::zeros<accfloat, 32>();
        m = aie::mac(m, aie::concat(ph0, ph1), ones);
        m = aie::mac(m, aie::concat(pl0, pl1), ones);
        acc32 total;
        total.from_vector(fmulN<32>(aie::broadcast<float, 32>(out[tq * (hd + 1) + r]), keep));
        total = aie::add(total, aie::broadcast<float, 32>(aie::reduce_add(m.to_vector<float>())));
        out[tq * hd + r] = peak[0];
        out[tq * (hd + 1) + r] = total.to_vector<float>()[0];
        v16b kh, kl;
        split16(keep.extract<16>(0), kh, kl);
        const auto keep_h = aie::concat(kh, kh), keep_l = aie::concat(kl, kl);
        for (unsigned d = 0; d < hd; d += 32) {
            float* dest = out + r * hd + d;
            v32b oh, ol;
            split32(aie::load_v<32>(dest), oh, ol);
            acc32 o = aie::mul(oh, keep_h);
            o = aie::mac(o, oh, keep_l);
            o = aie::mac(o, ol, keep_h);
#pragma clang loop unroll_count(4)
            for (unsigned j = 0; j < tk; ++j) {
                const auto value = aie::load_v<32>(v + j * hd + d);
                o = aie::mac(o, value, aie::broadcast<bfloat16, 32>(probs[j]));
                o = aie::mac(o, value, aie::broadcast<bfloat16, 32>(probs[32 + j]));
            }
            aie::store_v(dest, o.to_vector<float>());
        }
        (void)scratch;
    }
}
}  // namespace
#define INKLING2_ATTENTION_STEP inkling_attn_chunk_aligned
#else
#define INKLING2_ATTENTION_STEP inkling_attn_chunk
#endif
extern "C" void INKLING2_ATTENTION_STEP(const uint8_t* __restrict query, const uint8_t* __restrict kv,
                                    float* __restrict out, float* __restrict scores,
                                    bfloat16* __restrict probs, int32_t chunk) {
#ifdef INKLING2_ATTENTION_NULL
    return;
#endif
    aie::set_rounding(aie::rounding_mode::conv_even);
    const auto* info = reinterpret_cast<const int32_t*>(query);
    const int32_t query_start = info[0], key_start = info[1] + chunk * tk;
    const int32_t key_end = info[2], window = info[3];
    if (key_start >= key_end || key_start > query_start + static_cast<int32_t>(tq) - 1 ||
        (window && key_start + static_cast<int32_t>(tk) - 1 <= query_start - window)) return;

#ifdef INKLING2_ALIGNED_CHUNKS
    if (info[6] > 0) {
        decode_rows(query, kv, out, scores, probs);
        return;
    }
#endif
    const auto* qh = reinterpret_cast<const bfloat16*>(query + 128);
    const auto* ql = qh + query_words;
    const auto* k = reinterpret_cast<const bfloat16*>(kv);
    const auto* v = k + tk * hd;
    const auto* bh = v + tk * hd;
    const auto* bl = bh + rd * strip;
    const auto query_pos = aie::add(aie::load_v<tq>(lanes), aie::broadcast<int32_t, tq>(query_start));
    const auto min_pos = aie::sub(query_pos, aie::broadcast<int32_t, tq>(window));
#ifdef INKLING2_ALIGNED_CHUNKS
    // The host sets this flag only when its chunk boundaries are absolute.
    const bool aligned = info[7] == 1;
    const int32_t chunk_end = key_start + static_cast<int32_t>(tk);
    const int32_t last = (key_end < chunk_end ? key_end : chunk_end) - 1;
    const auto visible_chunk = aie::le(aie::broadcast<int32_t, tq>(key_start), query_pos) &
        (window ? aie::gt(aie::broadcast<int32_t, tq>(last), min_pos) : aie::mask<tq>(true));
#endif
    const bool relative = query_start - key_start - static_cast<int32_t>(tk) + 1 < extent;
    auto* peak_ptr = out + tq * hd;
    auto* mass_ptr = peak_ptr + tq;
    const v16f old_peak = aie::load_v<tq>(peak_ptr);
    v16f peak = old_peak;
    const auto invalid = aie::broadcast<float, tq>(masked);
    for (unsigned key = 0; key < tk; ++key) {
        const int32_t position = key_start + key;
        const auto p = aie::broadcast<int32_t, tq>(position);
        const auto visible = aie::le(p, query_pos) & (window ? aie::gt(p, min_pos) : aie::mask<tq>(true));
        const v16f s = position < key_end ? aie::select(invalid, score(qh, ql, k + key * hd, bh, bl, key, relative), visible)
                                          : invalid;
        peak = aie::max(peak, s);
        aie::store_v(scores + key * tq, s);
    }

    const auto keep = exp16(fsubN<tq>(old_peak, peak));
    const auto old_mass = aie::load_v<tq>(mass_ptr);
    auto mass = fmulN<tq>(old_mass, keep);
    const auto zero = aie::zeros<float, tq>();
    for (unsigned key = 0; key < tk; ++key) {
        const auto s = aie::load_v<tq>(scores + key * tq);
        const auto p = aie::select(zero, exp16(fsubN<tq>(s, peak)), aie::gt(s, invalid));
        v16b hi, lo;
        split16(p, hi, lo);
        aie::store_v(probs + key * 64, duplicate(hi));
        aie::store_v(probs + key * 64 + 32, duplicate(lo));
        accf16 a;
        a.from_vector(mass);
        const auto ones = aie::broadcast<bfloat16, tq>((bfloat16)1.0f);
        a = aie::mac(a, hi, ones);
        a = aie::mac(a, lo, ones);
        mass = a.to_vector<float>();
    }
#ifdef INKLING2_ALIGNED_CHUNKS
    if (aligned) {
        peak = aie::select(old_peak, peak, visible_chunk);
        mass = aie::select(old_mass, mass, visible_chunk);
    }
#endif
    aie::store_v(peak_ptr, peak);
    aie::store_v(mass_ptr, mass);

    v16b keep_hi, keep_lo;
    split16(keep, keep_hi, keep_lo);
    const auto dh = duplicate(keep_hi), dl = duplicate(keep_lo);
    for (unsigned row = 0; row < tq; ++row) {
#ifdef INKLING2_ALIGNED_CHUNKS
        // An empty chunk must preserve the numerator bit for bit.
        const int32_t pos = query_start + static_cast<int32_t>(row);
        if (aligned && (pos < key_start || (window && last <= pos - window))) continue;
#endif
        const auto rh = broadcast_lane(dh, row), rl = broadcast_lane(dl, row);
        for (unsigned d = 0; d < hd; d += 32) {
            auto* dest = out + row * hd + d;
            accf32 a = aie::zeros<accfloat, 32>();
            v32b hi, lo;
            split32(aie::load_v<32>(dest), hi, lo);
            a = aie::mac(a, hi, rh);
            a = aie::mac(a, hi, rl);
            a = aie::mac(a, lo, rh);
            for (unsigned key = 0; key < tk; ++key) {
                const auto val = aie::load_v<32>(v + key * hd + d);
                const auto p = probs + key * 64;
                a = aie::mac(a, val, broadcast_lane(aie::load_v<32>(p), row));
                a = aie::mac(a, val, broadcast_lane(aie::load_v<32>(p + 32), row));
            }
            aie::store_v(dest, a.to_vector<float>());
        }
    }
}
