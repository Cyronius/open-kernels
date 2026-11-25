// One extra output row carries the base-2 center and FP32 softmax mass.
void af_store_state(float *__restrict o, const float *__restrict st) {
  float *meta = o + TQ * HD;
  aie::store_v(meta, aie::load_v<32>(st + ST_M));
  aie::store_v(meta + TQ, aie::load_v<32>(st + ST_L));
  for (int d = 2 * TQ; d < HD; d += 32)
    aie::store_v(meta + d, aie::zeros<float, 32>());
}
// Three-product BFP PV. Pack probabilities once per key chunk and each
// transposed V strip once, then reuse those packed blocks across output tiles.
using PVBlock = aie::block_vector<bfp16ebs8, 64>;
using PVAccum = aie::accum<accfloat, 64>;

static inline __attribute__((always_inline)) void pv_split(
    const PVAccum& input, aie::block_vector_output_buffer_stream<bfp16ebs8, 64>& stream) {
  const PVBlock hi = input.to_vector<bfp16ebs8>();
  const auto decoded = PVAccum(hi).to_vector<bfloat16>();
  const auto residual = aie::sub(input, PVAccum(decoded)).to_vector<bfloat16>();
  stream.push(hi);
  stream.push(PVAccum(residual).to_vector<bfp16ebs8>());
}

void af_pv_select(float *__restrict scratch, bfloat16 *__restrict value, float *__restrict out,
                  float *__restrict state, bfloat16 *__restrict low, int32_t matrix) {
  if (!matrix) {
    af_pv(scratch, value, out, state, low);
    return;
  }
  SET_ROUNDING();
  if (reinterpret_cast<const int32_t*>(state + ST_I)[1] == 0)
    return;
  constexpr int side = 8, words = side * side, panel_bytes = side * 9;
  constexpr int panels = LKP / side;
  alignas(64) uint8_t probability_bytes[2 * (TQ / side) * panels * panel_bytes];
  alignas(64) uint8_t value_bytes[2 * panels * panel_bytes];
  auto* packed_p = reinterpret_cast<bfp16ebs8*>(probability_bytes);
  auto* packed_v = reinterpret_cast<bfp16ebs8*>(value_bytes);
  const auto* high = reinterpret_cast<const bfloat16*>(scratch);
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> p_output(packed_p);
  for (int row = 0; row < TQ; row += side) {
    for (int key = 0; key < LKP; key += side) {
      aie::vector<bfloat16, words> hi, lo;
#pragma clang loop unroll(full)
      for (int k = 0; k < side; ++k) {
        const auto dup_hi = aie::load_v<2 * side>(high + (key + k) * (2 * TQ) + 2 * row);
        hi.insert(k, aie::filter_even(dup_hi, 1));
        lo.insert(k, aie::load_v<side>(low + (key + k) * TQ + row));
      }
      const PVAccum top(aie::transpose(hi, side, side));
      const PVAccum tail(aie::transpose(lo, side, side));
      pv_split(aie::add(top, tail), p_output);
    }
  }

#pragma clang loop unroll(disable)
  for (int d = 0; d < HD; d += side) {
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> v_output(packed_v);
    for (int key = 0; key < LKP; key += side) {
      aie::vector<bfloat16, words> v;
#pragma clang loop unroll(full)
      for (int k = 0; k < side; ++k)
        v.insert(k, aie::load_v<side>(value + (key + k) * HD + d));
      pv_split(PVAccum(aie::transpose(v, side, side)), v_output);
    }
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> p_input(packed_p);
    for (int row = 0; row < TQ; row += side) {
      aie::vector<float, words> current;
#pragma clang loop unroll(full)
      for (int r = 0; r < side; ++r)
        current.insert(r, aie::load_v<side>(out + (row + r) * HD + d));
      PVAccum acc(current);
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> v_input(packed_v);
      for (int k = 0; k < panels; ++k) {
        const auto ph = p_input.pop();
        const auto pl = p_input.pop();
        const auto vh = v_input.pop();
        const auto vl = v_input.pop();
        acc = ::mac_8x8_8x8T_conf(ph, vh, acc, 0, 0, 0);
        acc = ::mac_8x8_8x8T_conf(ph, vl, acc, 0, 0, 0);
        acc = ::mac_8x8_8x8T_conf(pl, vh, acc, 0, 0, 0);
      }
      const auto result = acc.to_vector<float>();
#pragma clang loop unroll(full)
      for (int r = 0; r < side; ++r)
        aie::store_v(out + (row + r) * HD + d, result.extract<side>(r));
    }
  }
}

// All keys in a dispatched older-key segment are visible. The scores are
// still key-major for the unchanged split-probability softmax and PV code.
void af_qk_select(bfloat16 *__restrict qh, bfloat16 *__restrict ql, bfloat16 *__restrict key,
                  float *__restrict scores, float *__restrict state, int32_t token, int32_t chunk,
                  int32_t bfp) {
  if (!bfp) {
    af_qk(qh, ql, key, scores, state, token, chunk);
    return;
  }
  SET_ROUNDING();
  auto* info = reinterpret_cast<int32_t*>(state + ST_I);
  info[1] = chunk * LKP + LKP - 1 <= info[0] ? 1 : 0;
  if (!info[1])
    return;
  constexpr int side = 8, columns = HD / side;
  const auto* k = reinterpret_cast<const bfp16ebs8*>(key);
  const auto* h = reinterpret_cast<const bfp16ebs8*>(qh);
  const auto* l = reinterpret_cast<const bfp16ebs8*>(ql);
  for (int at = 0; at < LKP; at += side) {
#pragma clang loop unroll(disable)
    for (int row = 0; row < TQ; row += side) {
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> keys(k), high(h), low(l);
      keys.seek(2 * (at / side) * columns);
      high.seek((row / side) * columns);
      low.seek((row / side) * columns);
      PVAccum acc = aie::zeros<accfloat, 64>();
#pragma clang loop unroll(disable)
      for (int d = 0; d < columns; ++d) {
        const auto kh = keys.pop();
        const auto kl = keys.pop();
        const auto q_hi = high.pop();
        const auto q_lo = low.pop();
        acc = ::mac_8x8_8x8T_conf(kh, q_hi, acc, 0, 0, 0);
        acc = ::mac_8x8_8x8T_conf(kh, q_lo, acc, 0, 0, 0);
        acc = ::mac_8x8_8x8T_conf(kl, q_hi, acc, 0, 0, 0);
      }
      const auto result = acc.to_vector<float>();
#pragma clang loop unroll(full)
      for (int r = 0; r < side; ++r)
        aie::store_v(scores + (at + r) * TQ + row, result.extract<side>(r));
    }
  }
}
