#include "open_qwen36/inkling2_dspark.hpp"

#include "open_qwen36/inkling2_cpu.hpp"
#include "open_qwen36/inkling2_host.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#ifdef __AVX2__
#include <immintrin.h>
#endif
#if defined(__AVX2__) && (defined(__F16C__) || (defined(_MSC_VER) && !defined(__clang__)))
#define INKLING2_DSPARK_F16C 1
#if !defined(__F16C__)
#include <intrin.h>
#endif
#endif

namespace open_qwen36 {
namespace {

using inkling2_cpu::Projection;
using inkling2_cpu::Rows;
using inkling2_cpu::checked_tensor;
using inkling2_cpu::dot;
using inkling2_cpu::load_small;
using inkling2_cpu::norm;
using inkling2_cpu::rms;
using inkling2_cpu::sigmoid;
using inkling2_cpu::softcap;

constexpr size_t hidden = Inkling2Host::embedding_width, vocab = Inkling2Host::vocabulary_size;
constexpr size_t feature_width = Inkling2Host::draft_feature_width, padded_vocab = 201024;
constexpr size_t layers = 3, rank = 256, ff = 9216, window = 1024;
constexpr size_t heads = 24, kv_heads = 8, dim = 128, kv_width = kv_heads * dim;
constexpr size_t taps = feature_width / hidden;
constexpr float score_scale = 0.08838834764831845f;  // 1 / sqrt(128)
static_assert(taps == layers + 1 && Inkling2Dspark::block_size == Inkling2Host::verify_limit);

[[noreturn]] void fail(const std::string& reason) {
    throw std::runtime_error("inkling2 DSpark: " + reason);
}

void check_geometry(const GgufFile& file) {
    if (file.kv_str("general.architecture") != "dflash") fail("expected the Inkling 2 Mini draft GGUF");
    const std::pair<const char*, uint64_t> expected[] = {
        {"block_count", layers}, {"block_size", Inkling2Dspark::block_size}, {"embedding_length", hidden},
        {"feed_forward_length", ff}, {"attention.head_count", heads}, {"attention.head_count_kv", kv_heads},
        {"attention.key_length", dim}, {"attention.value_length", dim}, {"attention.sliding_window", window},
        {"rope.dimension_count", dim}, {"dspark.markov_rank", rank}, {"vocab_size", padded_vocab},
        {"vocab_size_unpadded", vocab}, {"context_length", Inkling2Host::max_context},
        {"attention.causal", 0}, {"sample_from_anchor", 0}, {"has_confidence_head", 0},
    };
    for (const auto& [key, value] : expected)
        if (file.kv_u64(std::string("dflash.") + key) != value) fail(std::string("invalid ") + key);
    if (file.kv_u32_array("dflash.target_layers") != std::vector<uint32_t>{5, 23, 41, 48} ||
        file.kv_u32_array("dflash.attention.sliding_window_pattern") != std::vector<uint32_t>{1, 0, 1})
        fail("unexpected feature taps or attention pattern");
}

float positive(const GgufFile& file, const std::string& key) {
    const float value = static_cast<float>(file.kv_float(key));
    if (!std::isfinite(value) || value <= 0) fail("invalid " + key);
    return value;
}

struct DraftLayer {
    std::vector<float> context_norm, context_k_norm, attention_norm, q_norm, k_norm, ffn_norm;
    Projection context_k, context_v, q, k, v, output, gate, up, down;

    DraftLayer(const GgufFile& file, const std::string& prefix)
        : context_norm(load_small(file, prefix + "dflare_norm.weight", {hidden})),
          context_k_norm(load_small(file, prefix + "attn_k_ctx_norm.weight", {dim})),
          attention_norm(load_small(file, prefix + "attn_norm.weight", {hidden})),
          q_norm(load_small(file, prefix + "attn_q_norm.weight", {dim})),
          k_norm(load_small(file, prefix + "attn_k_norm.weight", {dim})),
          ffn_norm(load_small(file, prefix + "ffn_norm.weight", {hidden})),
          context_k(file, prefix + "attn_k_ctx.weight", hidden, kv_width),
          context_v(file, prefix + "attn_v_ctx.weight", hidden, kv_width),
          q(file, prefix + "attn_q.weight", hidden, heads * dim),
          k(file, prefix + "attn_k.weight", hidden, kv_width),
          v(file, prefix + "attn_v.weight", hidden, kv_width),
          output(file, prefix + "attn_output.weight", heads * dim, hidden),
          gate(file, prefix + "ffn_gate.weight", hidden, ff),
          up(file, prefix + "ffn_up.weight", hidden, ff),
          down(file, prefix + "ffn_down.weight", ff, hidden) {}
};

struct Cache {
    std::vector<uint16_t> key, value;
    size_t ring = 0;

    size_t offset(size_t pos) const { return (ring ? pos % ring : pos) * kv_width; }
    size_t first_visible(size_t pos) const { return ring && pos > ring ? pos - ring : 0; }
    void reserve(size_t pos, size_t limit) {
        const size_t need = (ring ? std::min(ring, pos) : pos) * kv_width;
        const size_t max_size = (ring ? ring : limit) * kv_width;
        if (need > key.capacity()) key.reserve(std::min(max_size, std::max(need, key.capacity() * 2)));
        if (need > value.capacity()) value.reserve(std::min(max_size, std::max(need, value.capacity() * 2)));
    }
    void append(size_t pos, const Rows& k, const Rows& v) {
        const size_t end = pos + k.count();
        const size_t need = (ring ? std::min(ring, end) : end) * kv_width;
        key.resize(need);
        value.resize(need);
        const size_t first = std::max(pos, first_visible(end));
        for (size_t p = first; p < end; ++p) {
            const size_t at = offset(p);
            std::transform(k[p - pos], k[p - pos] + kv_width, key.begin() + at, f32_to_fp16);
            std::transform(v[p - pos], v[p - pos] + kv_width, value.begin() + at, f32_to_fp16);
        }
    }
};

void add(Rows& x, const Rows& y) {
    for (size_t i = 0; i < x.data.size(); ++i) x.data[i] += y.data[i];
}

void round_kv(Rows& rows) {
    for (float& v : rows.data) v = fp16_to_f32(f32_to_fp16(v));
}

#ifdef INKLING2_DSPARK_F16C
bool f16c_available() {
#ifdef __F16C__
    return true;
#else
    static const bool supported = [] {
        int registers[4];
        __cpuid(registers, 1);
        return (registers[2] & (1 << 29)) != 0;
    }();
    return supported;
#endif
}
#endif

float attention_dot(const float* query, const float* key) {
#if defined(__AVX2__) && !defined(__FMA__)
    // Preserve the four accumulator chains and reduction order of dot().
    __m128 sum = _mm_setzero_ps();
    for (size_t d = 0; d < dim; d += 4)
        sum = _mm_add_ps(sum, _mm_mul_ps(_mm_loadu_ps(query + d), _mm_loadu_ps(key + d)));
    float lanes[4];
    _mm_storeu_ps(lanes, sum);
    return (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
#else
    return dot(query, key, dim);
#endif
}

void update_value(float* output, const float* value, float keep, float take) {
#if defined(__AVX2__) && !defined(__FMA__)
    const auto keep_lanes = _mm256_set1_ps(keep), take_lanes = _mm256_set1_ps(take);
    for (size_t d = 0; d < dim; d += 8) {
        const auto previous = _mm256_mul_ps(_mm256_loadu_ps(output + d), keep_lanes);
        const auto incoming = _mm256_mul_ps(_mm256_loadu_ps(value + d), take_lanes);
        _mm256_storeu_ps(output + d, _mm256_add_ps(previous, incoming));
    }
#else
    for (size_t d = 0; d < dim; ++d) output[d] = output[d] * keep + value[d] * take;
#endif
}

#ifdef __AVX2__
// The Markov state makes the seven output projections sequential. Read the
// mapped Q8_0 head directly and apply each block's scale in registers.
template <bool HardwareHalf>
std::vector<float> project_q8_head(const uint8_t* weights, const float* input, float cap) {
    constexpr size_t values = 32, bytes = 34, row_bytes = hidden / values * bytes;
    static_assert(hidden % values == 0);
    std::vector<float> output(vocab);
#pragma omp parallel for schedule(static)
    for (int r = 0; r < static_cast<int>(vocab); ++r) {
        const uint8_t* block = weights + static_cast<size_t>(r) * row_bytes;
        const float* x = input;
        __m256 sum = _mm256_setzero_ps();
        for (size_t col = 0; col < hidden; col += values, block += bytes, x += values) {
            uint16_t scale;
            std::memcpy(&scale, block, sizeof(scale));
            float converted;
#ifdef INKLING2_DSPARK_F16C
            if constexpr (HardwareHalf)
                converted = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(scale)));
            else
#endif
                converted = fp16_to_f32(scale);
            const __m256 d = _mm256_set1_ps(converted);
            __m256 products[4];
            for (size_t i = 0; i < 4; ++i) {
                const size_t j = i * 8;
                const auto packed = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(block + 2 + j));
                const auto q = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(packed));
                products[i] = _mm256_mul_ps(q, _mm256_loadu_ps(x + j));
            }
            const auto dot = _mm256_add_ps(_mm256_add_ps(products[0], products[1]),
                                          _mm256_add_ps(products[2], products[3]));
            sum = _mm256_add_ps(sum, _mm256_mul_ps(d, dot));
        }
        float lanes[4];
        _mm_storeu_ps(lanes, _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1)));
        output[r] = softcap((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]), cap);
    }
    return output;
}
#endif

}  // namespace

struct Inkling2Dspark::Impl {
    const GgufFile& target;
    GgufFile file;
    size_t capacity, pos = 0;
    float eps, embedding_eps, logit_cap;
    Projection joint, correction, head;
#ifdef __AVX2__
    const uint8_t* q8_head = nullptr;
#endif
    std::vector<float> fusion, context_norm, mask, output_norm, embedding_norm, joint_bias, correction_bias, last_context;
    std::array<float, dim / 2> frequencies;
    std::vector<DraftLayer> weights;
    std::array<Cache, layers> cache;

    Impl(const std::string& path, const GgufFile& target_weights, size_t limit)
        : target(target_weights), file(path), capacity(limit) {
        if (!limit || limit > Inkling2Host::max_context) fail("invalid context capacity");
        check_geometry(file);
        const auto arch = target.kv_str("general.architecture");
        if (arch != "inkling2_loop" && arch != "inkling2") fail("expected an Inkling 2 Mini target");
        eps = positive(file, "dflash.attention.layer_norm_rms_epsilon");
        embedding_eps = positive(target, arch + ".attention.layer_norm_rms_epsilon");
        logit_cap = positive(file, "dflash.final_logit_softcapping");
        const float base = positive(file, "dflash.rope.freq_base");
        for (size_t d = 0; d < frequencies.size(); ++d)
            frequencies[d] = std::pow(base, -static_cast<float>(d) / frequencies.size());
        checked_tensor(target, "token_embd.weight", {hidden, padded_vocab});
        checked_tensor(file, "markov_w1.weight", {rank, padded_vocab});
        joint = Projection(file, "joint_proj.weight", 2 * rank + 2 * hidden, 3 * rank);
        correction = Projection(file, "dspark_hidden_out.weight", rank, hidden);
        head = Projection(target, "output.weight", hidden, padded_vocab);
#ifdef __AVX2__
        if (target.tensor("output.weight").type == GgufFile::Type::Q8_0)
            q8_head = target.raw("output.weight");
#endif
        fusion = load_small(file, "dspark_fuse.weight", {taps, taps});
        for (size_t l = 0; l < taps; ++l) {
            auto* first = fusion.data() + l * taps;
            const float peak = *std::max_element(first, first + taps);
            float mass = 0.0f;
            for (size_t t = 0; t < taps; ++t) mass += (first[t] = std::exp(first[t] - peak));
            for (size_t t = 0; t < taps; ++t) first[t] /= mass;
        }
        context_norm = load_small(file, "dspark_norm.weight", {hidden});
        mask = load_small(file, "mask_embd.weight", {hidden});
        output_norm = load_small(file, "output_norm.weight", {hidden});
        embedding_norm = load_small(target, "token_embd_norm.weight", {hidden});
        joint_bias = load_small(file, "joint_proj.bias", {3 * rank});
        correction_bias = load_small(file, "dspark_hidden_out.bias", {hidden});
        for (size_t l = 0; l < layers; ++l) {
            weights.emplace_back(file, "blk." + std::to_string(l) + ".");
            cache[l].ring = l == 1 ? 0 : std::min(window, limit);
        }
    }

    Rows mixture(const float* features, size_t count, size_t tap_set, const std::vector<float>& weight) const {
        Rows rows(count, hidden);
        const float* coefficients = fusion.data() + tap_set * taps;
        for (size_t n = 0; n < count; ++n) {
            const float* input = features + n * feature_width;
            for (size_t t = 0; t < taps; ++t)
                for (size_t j = 0; j < hidden; ++j) rows[n][j] += coefficients[t] * input[t * hidden + j];
        }
        norm(rows, eps, weight);
        return rows;
    }

    void normalize_rotate(Rows& rows, const std::vector<float>& weight) const {
        for (size_t n = 0; n < rows.count(); ++n) {
            std::array<float, dim / 2> c, s;
            for (size_t d = 0; d < c.size(); ++d) {
                const float angle = static_cast<float>(pos + n) * frequencies[d];
                c[d] = std::cos(angle);
                s[d] = std::sin(angle);
            }
            for (size_t h = 0; h < rows.width; h += dim) {
                float* x = rows[n] + h;
                rms(x, dim, eps, weight.data());
                for (size_t d = 0; d < c.size(); ++d) {
                    const float a = x[d], b = x[d + dim / 2];
                    x[d] = a * c[d] - b * s[d];
                    x[d + dim / 2] = a * s[d] + b * c[d];
                }
            }
        }
    }

    void append(const std::vector<float>& features, size_t count) {
        if (features.size() % feature_width || count > features.size() / feature_width || count > capacity - pos)
            fail("invalid accepted feature block");
        if (!count) return;
        std::array<Rows, layers> k, v;
        for (size_t l = 0; l < layers; ++l) {
            const auto& layer = weights[l];
            const auto mixed = mixture(features.data(), count, l, layer.context_norm);
            k[l] = layer.context_k.apply(mixed);
            v[l] = layer.context_v.apply(mixed);
            normalize_rotate(k[l], layer.context_k_norm);
        }
        auto context = mixture(features.data() + (count - 1) * feature_width, 1, layers, context_norm);
        for (auto& kv : cache) kv.reserve(pos + count, capacity);
        for (size_t l = 0; l < layers; ++l) cache[l].append(pos, k[l], v[l]);
        pos += count;
        last_context = std::move(context.data);
    }

    Rows attended(const Rows& q, const Rows& k, const Rows& v, const Cache& kv) const {
        const size_t count = q.count(), first = kv.first_visible(pos);
        Rows result(count, heads * dim);
#ifdef INKLING2_DSPARK_F16C
        const bool hardware_half = f16c_available();
#endif
#pragma omp parallel for schedule(static)
        for (int group = 0; group < static_cast<int>(kv_heads); ++group) {
            constexpr size_t sharing = heads / kv_heads;
            float peak[block_size][sharing], mass[block_size][sharing]{};
            for (auto& row : peak) std::fill_n(row, sharing, -std::numeric_limits<float>::infinity());
            auto accumulate = [&](const float* key, const float* value) {
                for (size_t n = 0; n < count; ++n)
                    for (size_t share = 0; share < sharing; ++share) {
                        const size_t base = (group * sharing + share) * dim;
                        const float score = attention_dot(q[n] + base, key) * score_scale;
                        const float top = std::max(peak[n][share], score);
                        const float shift = peak[n][share] - top, weight = score - top;
                        const float keep = shift == 0.0f ? 1.0f : std::exp(shift);
                        const float take = weight == 0.0f ? 1.0f : std::exp(weight);
                        update_value(result[n] + base, value, keep, take);
                        mass[n][share] = mass[n][share] * keep + take;
                        peak[n][share] = top;
                    }
            };
            for (size_t p = first; p < pos; ++p) {
                const size_t base = kv.offset(p) + group * dim;
                std::array<float, dim> key, value;
#ifdef INKLING2_DSPARK_F16C
                if (hardware_half) {
                    for (size_t d = 0; d < dim; d += 8) {
                        const auto raw_k = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kv.key.data() + base + d));
                        const auto raw_v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kv.value.data() + base + d));
                        _mm256_storeu_ps(key.data() + d, _mm256_cvtph_ps(raw_k));
                        _mm256_storeu_ps(value.data() + d, _mm256_cvtph_ps(raw_v));
                    }
                } else
#endif
                {
                    for (size_t d = 0; d < dim; ++d) {
                        key[d] = fp16_to_f32(kv.key[base + d]);
                        value[d] = fp16_to_f32(kv.value[base + d]);
                    }
                }
                accumulate(key.data(), value.data());
            }
            // The noisy block is bidirectional and shares the anchor's context window.
            for (size_t n = 0; n < count; ++n) accumulate(k[n] + group * dim, v[n] + group * dim);
            for (size_t n = 0; n < count; ++n)
                for (size_t share = 0; share < sharing; ++share) {
                    float* out = result[n] + (group * sharing + share) * dim;
                    for (size_t d = 0; d < dim; ++d) out[d] /= mass[n][share];
                }
        }
        return result;
    }

    Rows backbone(int anchor) const {
        Rows x(block_size, hidden);
        target.embed_row("token_embd.weight", anchor, hidden, x[0]);
        rms(x[0], hidden, embedding_eps, embedding_norm.data());
        for (size_t n = 1; n < block_size; ++n) std::copy(mask.begin(), mask.end(), x[n]);
        for (size_t l = 0; l < layers; ++l) {
            const auto& layer = weights[l];
            auto input = x;
            norm(input, eps, layer.attention_norm);
            auto q = layer.q.apply(input), k = layer.k.apply(input), v = layer.v.apply(input);
            normalize_rotate(q, layer.q_norm);
            normalize_rotate(k, layer.k_norm);
            round_kv(k);
            round_kv(v);
            add(x, layer.output.apply(attended(q, k, v, cache[l])));
            input = x;
            norm(input, eps, layer.ffn_norm);
            auto ffn = layer.gate.apply(input);
            const auto up = layer.up.apply(input);
            for (size_t i = 0; i < ffn.data.size(); ++i) ffn.data[i] *= sigmoid(ffn.data[i]) * up.data[i];
            add(x, layer.down.apply(ffn));
        }
        norm(x, eps, output_norm);
        return x;
    }

    std::vector<float> project_head(const Rows& input) const {
#ifdef __AVX2__
        if (q8_head && target.fold("output.weight") == 1.0f) {
#ifdef INKLING2_DSPARK_F16C
            if (f16c_available()) return project_q8_head<true>(q8_head, input[0], logit_cap);
#endif
            return project_q8_head<false>(q8_head, input[0], logit_cap);
        }
#endif
        auto logits = head.apply(input, 0, vocab).data;
        for (float& value : logits) value = softcap(value, logit_cap);
        return logits;
    }

    std::vector<int> propose(int anchor, size_t count, std::vector<float>* captured) const {
        if (anchor < 0 || static_cast<size_t>(anchor) >= vocab || count > max_proposal ||
            !pos || count >= capacity - pos)
            fail("invalid anchor, proposal count or context capacity");
        const auto x = backbone(anchor);
        std::array<float, rank> state{};
        Rows joint_input(1, 2 * rank + 2 * hidden), latent(1, rank);
        std::vector<int> ids;
        std::vector<float> all_logits;
        int previous = anchor;
        std::copy(last_context.begin(), last_context.end(), joint_input[0] + 2 * rank + hidden);
        for (size_t n = 0; n < count; ++n) {
            std::copy(state.begin(), state.end(), joint_input[0]);
            file.embed_row("markov_w1.weight", previous, rank, joint_input[0] + rank);
            std::copy_n(x[n + 1], hidden, joint_input[0] + 2 * rank);
            auto projected = joint.apply(joint_input);
            for (size_t d = 0; d < rank; ++d) {
                const float gate = sigmoid(projected[0][d] + joint_bias[d]);
                const float candidate = std::tanh(projected[0][rank + d] + joint_bias[rank + d]);
                state[d] = candidate + gate * (state[d] - candidate);
                latent[0][d] = std::tanh(projected[0][2 * rank + d] + joint_bias[2 * rank + d]);
            }
            // The base and residual projections use the same target head.
            auto input = correction.apply(latent);
            for (size_t d = 0; d < hidden; ++d) input[0][d] += x[n + 1][d] + correction_bias[d];
            auto logits = project_head(input);
            previous = static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
            ids.push_back(previous);
            if (captured) all_logits.insert(all_logits.end(), logits.begin(), logits.end());
        }
        if (captured) *captured = std::move(all_logits);
        return ids;
    }
};

Inkling2Dspark::Inkling2Dspark(const std::string& path, const GgufFile& target, size_t context)
    : impl_(std::make_unique<Impl>(path, target, context)) {}
Inkling2Dspark::~Inkling2Dspark() = default;

void Inkling2Dspark::append(const std::vector<float>& features, size_t accepted) { impl_->append(features, accepted); }

std::vector<int> Inkling2Dspark::propose(int anchor, size_t count, std::vector<float>* logits) const {
    if (!count) {
        if (logits) logits->clear();
        return {};
    }
    return impl_->propose(anchor, count, logits);
}

void Inkling2Dspark::reset() {
    impl_->pos = 0;
    impl_->last_context.clear();
    for (auto& cache : impl_->cache) {
        for (auto* data : {&cache.key, &cache.value}) {
            if (data->capacity() <= window * kv_width) data->clear();
            else std::vector<uint16_t>().swap(*data);
        }
    }
}

size_t Inkling2Dspark::position() const { return impl_->pos; }

}  // namespace open_qwen36
