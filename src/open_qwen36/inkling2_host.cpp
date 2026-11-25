#include "open_qwen36/inkling2_host.hpp"

#include "open_qwen36/inkling2_cpu.hpp"
#include "open_qwen36/q4nx_file.hpp"  // bf16 conversions

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <new>

#ifdef _WIN64
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#include <stdexcept>
#include <utility>

namespace open_qwen36 {
namespace {

// The fixed-loop Mini geometry. The constructor checks the matching GGUF metadata
// and tensor shapes before it creates the per-visit state.
constexpr size_t hidden = Inkling2Host::embedding_width, ff = 1152, layers = Inkling2Host::layer_count;
constexpr size_t experts = 128, top = 8;
constexpr size_t heads = 24, kv_heads = 8, dim = 128, rel_dim = 16;
constexpr size_t q_width = heads * dim, kv_width = kv_heads * dim;
constexpr size_t taps = 4, rank = 64, window = 1024, extent = 1024;
constexpr size_t core_start = 6, core_end = 18, passes = 3, block_size = 6;
constexpr size_t visits = layers + (passes - 1) * (core_end - core_start);
constexpr size_t padded_vocab = 201024, vocab = Inkling2Host::vocabulary_size;
using inkling2_cpu::Type;
using inkling2_cpu::Rows;
using inkling2_cpu::Projection;
using inkling2_cpu::checked_tensor;
using inkling2_cpu::dot;
using inkling2_cpu::fail;
using inkling2_cpu::load_small;
using inkling2_cpu::norm;
using inkling2_cpu::rms;
using inkling2_cpu::sigmoid;
using inkling2_cpu::softcap;

void situ_row(const float* both, float gate_cap, float up_cap, float* inner) {
    for (size_t j = 0; j < ff; ++j) {
        const float gate = both[j], up = both[ff + j];
        inner[j] = softcap(gate, gate_cap) * sigmoid(gate) * softcap(up, up_cap);
    }
}

Rows situ(const Rows& both, float gate_cap, float up_cap) {
    Rows inner(both.count(), ff);
    for (size_t i = 0; i < both.count(); ++i)
        situ_row(both[i], gate_cap, up_cap, inner[i]);
    return inner;
}

void clip(float* x, size_t n, float limit, const float* weight = nullptr) {
    if (weight)
        for (size_t j = 0; j < n; ++j) x[j] *= weight[j];
    const float scale = std::min(1.0f, limit / (std::sqrt(dot(x, x, n)) + 1e-6f));
    for (size_t j = 0; j < n; ++j) x[j] *= scale;
}

std::vector<Rows> mix(const std::vector<Rows>& bank, const Rows& current,
                      const std::vector<float>& query, float eps) {
    const size_t n = bank.size() + 1, nq = query.size() / hidden;
    std::vector<Rows> out(nq, Rows(current.count(), hidden));
    const float scale = 1.0f / std::sqrt(static_cast<float>(hidden));
    const long long mix_rows = static_cast<long long>(current.count());
    // Token rows are independent, so splitting them keeps every output bit.
#pragma omp parallel if (mix_rows >= 64)
    {
        Rows normalized(n, hidden);
        std::vector<float> scores(n);
#pragma omp for schedule(static)
        for (long long mix_row = 0; mix_row < mix_rows; ++mix_row) {
            const size_t t = static_cast<size_t>(mix_row);
            for (size_t b = 0; b < n; ++b) {
                const float* row = b < bank.size() ? bank[b][t] : current[t];
                std::copy_n(row, hidden, normalized[b]);
                rms(normalized[b], hidden, eps);
            }
            for (size_t q = 0; q < nq; ++q) {
                for (size_t b = 0; b < n; ++b)
                    scores[b] = dot(query.data() + q * hidden, normalized[b], hidden) * scale;
                const float peak = *std::max_element(scores.begin(), scores.end());
                float mass = 0;
                for (float& score : scores) {
                    score = std::exp(score - peak);
                    mass += score;
                }
                for (size_t b = 0; b < n; ++b) {
                    const float p = scores[b] / mass;
                    for (size_t j = 0; j < hidden; ++j) out[q][t][j] += p * normalized[b][j];
                }
            }
        }
    }
    return out;
}

struct Conv {
    std::vector<float> weights;
    Projection dynamic;

    void set_weights(const std::vector<float>& tensor, size_t width) {
        // GGUF groups the oldest-first taps by channel. Read one contiguous lag at a time.
        weights.resize(width * taps);
        for (size_t lag = 0; lag < taps; ++lag)
            for (size_t j = 0; j < width; ++j)
                weights[lag * width + j] = tensor[j * taps + taps - lag - 1];
    }

    Rows run(const Rows& x, const Rows* input, std::vector<float>& history, std::vector<float>* tape) const {
        const size_t width = x.width, count = x.count();
        if (tape) *tape = x.data;
        Rows projected;
        if (input) projected = dynamic.apply(*input);
        if (history.empty()) history.resize((taps - 1) * width);
        // The original (pre-convolution) inputs are the state for future tokens.
        auto history_row = [&](long long t) {
            return t < 0 ? history.data() + static_cast<size_t>(t + static_cast<long long>(taps - 1)) * width
                         : x[static_cast<size_t>(t)];
        };
        Rows out = x;
        const float scale = 1.0f / std::sqrt(static_cast<float>(1 + taps * (input ? 2 : 1)));
        const long long rows = static_cast<long long>(count);
#pragma omp parallel for schedule(static) if (rows >= 64)
        for (long long row = 0; row < rows; ++row) {
            const size_t t = static_cast<size_t>(row);
            float* dst = out[t];
            for (size_t lag = 0; lag < taps; ++lag) {
                const float* src = history_row(static_cast<long long>(t) - static_cast<long long>(lag));
                const float* weight = weights.data() + lag * width;
                if (input) {
                    for (size_t head = 0; head < width / dim; ++head) {
                        const float delta = projected[t][head * taps + lag];
                        const size_t first = head * dim;
                        for (size_t j = first; j < first + dim; ++j)
                            dst[j] += src[j] * (weight[j] + delta);
                    }
                } else {
                    for (size_t j = 0; j < width; ++j) dst[j] += src[j] * weight[j];
                }
            }
            for (size_t j = 0; j < width; ++j) dst[j] *= scale;
        }
        for (size_t r = 0; r < taps - 1; ++r) {
            const auto t = static_cast<long long>(count + r) - static_cast<long long>(taps - 1);
            std::copy_n(history_row(t), width, history.data() + r * width);
        }
        return out;
    }
};

struct Controller {
    Projection down, gain, forget;

    void apply(Rows& x, const Rows& message, float eps) const {
        Rows joined(x.count(), 2 * hidden);
        for (size_t t = 0; t < x.count(); ++t) {
            std::copy_n(x[t], hidden, joined[t]);
            std::copy_n(message[t], hidden, joined[t] + hidden);
            rms(joined[t], hidden, eps);
            rms(joined[t] + hidden, hidden, eps);
        }
        const Rows z = down.apply(joined);
        const Rows g = gain.apply(z), f = forget.apply(z);
        const long long update_rows = static_cast<long long>(x.count());
#pragma omp parallel for schedule(static) if (update_rows >= 64)
        for (long long update_row = 0; update_row < update_rows; ++update_row) {
            const size_t t = static_cast<size_t>(update_row);
            for (size_t j = 0; j < hidden; ++j)
                x[t][j] += std::tanh(g[t][j]) * message[t][j] - std::tanh(f[t][j]) * x[t][j];
        }
    }
};

#ifdef _WIN64
class LongKvAllocation {
    struct Release {
        void operator()(uint16_t* storage) const noexcept { VirtualFree(storage, 0, MEM_RELEASE); }
    };
    std::unique_ptr<uint16_t, Release> memory_;
    size_t committed_ = 0;

    static size_t rounded_bytes(size_t entries) {
        static const size_t granularity = [] {
            SYSTEM_INFO info{};
            GetSystemInfo(&info);
            return static_cast<size_t>(info.dwAllocationGranularity);
        }();
        const size_t bytes = entries * sizeof(uint16_t);
        return (bytes + granularity - 1) / granularity * granularity;
    }

public:
    void prepare(size_t required, size_t limit) {
        if (!memory_) {
            auto* storage = static_cast<uint16_t*>(VirtualAlloc(nullptr, rounded_bytes(limit), MEM_RESERVE, PAGE_READWRITE));
            if (!storage) throw std::bad_alloc();
            memory_.reset(storage);
        }
        const size_t wanted = rounded_bytes(required);
        if (committed_ < wanted) {
            auto* begin = memory_.get() + committed_ / sizeof(uint16_t);
            const size_t additional = wanted - committed_;
            if (!VirtualAlloc(begin, additional, MEM_COMMIT, PAGE_READWRITE)) throw std::bad_alloc();
            std::uninitialized_default_construct_n(begin, additional / sizeof(uint16_t));
            committed_ = wanted;
        }
    }
    bool retire(size_t first, size_t count) const noexcept {
        const size_t offset = first * sizeof(uint16_t), bytes = count * sizeof(uint16_t);
        return memory_ && bytes && rounded_bytes(first) == offset && rounded_bytes(count) == bytes &&
            offset <= committed_ && bytes <= committed_ - offset &&
            VirtualFree(memory_.get() + first, bytes, MEM_DECOMMIT) != 0;
    }
    uint16_t* data() noexcept { return memory_.get(); }
    const uint16_t* data() const noexcept { return memory_.get(); }
};
#endif

class CachedKvRows {
    std::vector<uint16_t> values_;
    mutable bool keys_retired_ = false;
#ifdef _WIN64
    LongKvAllocation long_;
#endif

public:
    bool keys_retired() const noexcept { return keys_retired_; }
    bool can_retire_keys() const noexcept {
#ifdef _WIN64
        return long_.data() != nullptr;
#else
        return false;
#endif
    }
    static bool retire_key_rows(const void* storage, size_t first, size_t count) noexcept {
#ifdef _WIN64
        const auto& cache = *static_cast<const CachedKvRows*>(storage);
        if (cache.long_.retire(first * kv_width, count * kv_width)) {
            cache.keys_retired_ = true;
            return true;
        }
#else
        (void)storage;
        (void)first;
        (void)count;
#endif
        return false;
    }

    void prepare(size_t required, size_t limit) {
#ifdef _WIN64
        constexpr size_t long_context = 65536;
        if (limit >= long_context * kv_width) {
            long_.prepare(required, limit);
            return;
        }
#endif
        if (values_.capacity() < required)
            values_.reserve(std::min(limit, std::max(required, 2 * values_.capacity())));
    }
    void resize(size_t size) {
#ifdef _WIN64
        if (long_.data()) return;
#endif
        values_.resize(size);
    }
    uint16_t* data() noexcept {
#ifdef _WIN64
        if (auto* storage = long_.data()) return storage;
#endif
        return values_.data();
    }
    const uint16_t* data() const noexcept {
#ifdef _WIN64
        if (const auto* storage = long_.data()) return storage;
#endif
        return values_.data();
    }
};

using Histories = std::array<std::vector<float>, 4>;
struct VisitCache {
    size_t ring = 0;
    CachedKvRows key, value;
    Histories conv;

    size_t offset(size_t p) const { return (ring ? p % ring : p) * kv_width; }
};

struct Pending {
    std::vector<uint16_t> key, value;
    Histories conv, conv_inputs;
};

struct PendingBlock {
    std::array<Pending, visits> states;
    std::array<size_t, visits> rings;
    size_t rows;
};

struct Layer {
    bool sliding;
    std::array<Projection, 3> qkv;
    Projection relative, sink, gate, output, router, gate_up, down;
    std::array<Conv, 4> conv;
    std::array<Controller, 4> controller;
    std::array<std::vector<float>, 3> attn_norm;
    std::vector<float> q_clip, rel_clip, rel_bias, attn_query, ffn_query, msg_query, ffn_norm, router_bias;

    Layer(const GgufFile& file, size_t l) : sliding(l % block_size != block_size - 1) {
        const std::string prefix = "blk." + std::to_string(l) + ".";
        auto mat = [&](const std::string& name, size_t k, size_t n, size_t planes = 0) {
            return Projection(file, prefix + name + ".weight", k, n, planes);
        };
        auto vals = [&](const std::string& name, const std::vector<uint64_t>& shape) {
            return load_small(file, prefix + name + ".weight", shape);
        };
        const std::array<std::string, 3> kinds{"q", "k", "v"};
        for (size_t i = 0; i < kinds.size(); ++i) {
            const std::string base = "attn_" + kinds[i];
            qkv[i] = mat(base, hidden, i == 0 ? q_width : kv_width);
            attn_norm[i] = vals(sliding ? "attn_norm" : base + "_inp_norm", {hidden});
            if (i) {
                conv[i - 1].set_weights(vals(base + "_conv", {taps, kv_width}), kv_width);
                if (!sliding) conv[i - 1].dynamic = mat(base + "_conv_proj", hidden, kv_heads * taps);
            }
        }
        relative = mat("attn_rel", hidden, heads * rel_dim);
        sink = mat("attn_sink_proj", hidden, heads);
        gate = mat("attn_gate", hidden, q_width);
        output = mat("attn_output", q_width, hidden);
        conv[2].set_weights(vals("attn_post_conv", {taps, hidden}), hidden);
        conv[3].set_weights(vals("ffn_post_conv", {taps, hidden}), hidden);
        q_clip = vals("attn_q_clip", {dim, heads});
        rel_clip = vals("attn_rel_clip", {rel_dim, heads});
        rel_bias = vals("attn_rel_bias", {rel_dim, extent});
        if (l) attn_query = vals("attn_res_query", sliding ? std::vector<uint64_t>{hidden}
                                                         : std::vector<uint64_t>{hidden, 3});
        ffn_query = vals("ffn_res_query", {hidden});
        ffn_norm = vals("ffn_norm", {hidden});
        router_bias = load_small(file, prefix + "exp_probs_b.bias", {experts});
        router = mat("ffn_gate_inp", hidden, experts);
        gate_up = mat("ffn_gate_up_exps", hidden, 2 * ff, experts);
        down = mat("ffn_down_exps", ff, hidden, experts);
        if (l >= core_start && l < core_end) {
            msg_query = vals("loop_msg_query", {hidden, controller.size()});
            for (size_t i = 0; i < controller.size(); ++i) {
                const std::string name = "loop_" + (i < kinds.size() ? kinds[i] : "ffn") + "_ctrl_";
                controller[i] = {mat(name + "down", 2 * hidden, rank),
                                 mat(name + "gain", rank, hidden), mat(name + "forget", rank, hidden)};
            }
        }
    }

    bool project_host_attention_inputs(const std::array<const float*, 3>& input, size_t count,
                                       const std::array<Rows*, 6>& output) const {
#ifdef __AVX2__
        const std::array<const Projection*, 6> legs{&qkv[0], &qkv[1], &qkv[2], &relative, &sink, &gate};
        for (const auto* projection : legs)
            if (!projection->q8 || projection->file->fold(projection->name) != 1.0f) return false;

        // Balance the large and small projections across one set of workers.
        constexpr size_t rows_per_chunk = 32;
        constexpr size_t stride = hidden / inkling2_cpu::q8_block_values * inkling2_cpu::q8_block_bytes;
        struct Chunk { const float* input; const uint8_t* weights; float* output; size_t rows; };
        std::vector<Chunk> chunks;
        chunks.reserve(count * ((2 * q_width + 2 * kv_width + heads * (rel_dim + 1)) / rows_per_chunk + legs.size()));
        for (size_t leg = 0; leg < legs.size(); ++leg) {
            const auto& projection = *legs[leg];
            *output[leg] = Rows(count, projection.outputs);
            for (size_t token = 0; token < count; ++token)
                for (size_t row = 0; row < projection.outputs; row += rows_per_chunk)
                    chunks.push_back({input[leg < input.size() ? leg : 0] + token * hidden,
                                      projection.q8 + row * stride, (*output[leg])[token] + row,
                                      std::min(rows_per_chunk, projection.outputs - row)});
        }
#pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(chunks.size()); ++i) {
            const auto& chunk = chunks[i];
            for (size_t row = 0; row < chunk.rows; ++row)
                chunk.output[row] = inkling2_cpu::dot_q8_rows<1>(chunk.input, chunk.weights + row * stride, hidden)[0];
        }
        return true;
#else
        return false;
#endif
    }

    Rows moe(const Rows& x, float route_scale, float gate_cap, float up_cap,
             size_t layer, Inkling2ExpertProjector* projector) const {
        struct Routed { size_t token; float weight; };
        std::array<std::vector<Routed>, experts> by_expert;
        Rows scores = router.apply(x);
        for (size_t t = 0; t < x.count(); ++t) {
            std::array<size_t, experts> ids;
            std::iota(ids.begin(), ids.end(), 0);
            for (size_t e = 0; e < experts; ++e) scores[t][e] = sigmoid(scores[t][e]);
            std::partial_sort(ids.begin(), ids.begin() + top, ids.end(), [&](size_t a, size_t b) {
                const float sa = scores[t][a] + router_bias[a], sb = scores[t][b] + router_bias[b];
                return sa != sb ? sa > sb : a < b;
            });
            float total = 0;
            for (size_t i = 0; i < top; ++i) total += scores[t][ids[i]];
            const float scale = route_scale / std::max(total, 1e-20f);
            for (size_t i = 0; i < top; ++i)
                by_expert[ids[i]].push_back({t, scale * scores[t][ids[i]]});
        }
        Rows out(x.count(), hidden);
        if (projector && x.count() == 1) {
            Inkling2ExpertProjector::ExpertIds ids{};
            Rows input(top, hidden), both(top, 2 * ff), values(top, hidden);
            size_t slot = 0;
            for (size_t e = 0; e < experts; ++e)
                if (!by_expert[e].empty()) {
                    ids[slot] = e;
                    std::copy_n(x[0], hidden, input[slot]);
                    ++slot;
                }
            projector->project(*gate_up.file, gate_up.name, layer, false, ids, input.data.data(), both.data.data());
            const Rows inner = situ(both, gate_cap, up_cap);
            projector->project(*down.file, down.name, layer, true, ids, inner.data.data(), values.data.data());
            for (size_t i = 0; i < top; ++i)
                for (size_t j = 0; j < hidden; ++j)
                    out[0][j] += by_expert[ids[i]].front().weight * values[i][j];
            return out;
        }
        const size_t block_rows = projector ? projector->grouped_block_rows() : 0;
        if (block_rows && projector->grouped_overlap()) {
            struct Group {
                Inkling2ExpertProjector::ExpertIds ids{};
                Inkling2ExpertProjector::ExpertRows counts{};
                std::array<const Routed*, top> routes{};
                size_t slots = 0;
            };
            // The same slices, in the same expert order, as the synchronous loop below.
            std::vector<Group> groups;
            for (size_t expert = 0, at = 0; expert < experts;) {
                Group group;
                while (group.slots < top && expert < experts) {
                    const auto& routed = by_expert[expert];
                    if (at == routed.size()) {
                        ++expert;
                        at = 0;
                        continue;
                    }
                    const size_t count = std::min(block_rows, routed.size() - at);
                    group.ids[group.slots] = expert;
                    group.counts[group.slots] = count;
                    group.routes[group.slots] = routed.data() + at;
                    at += count;
                    ++group.slots;
                }
                if (!group.slots) break;
                groups.push_back(group);
            }
            const size_t group_count = groups.size();
            if (!group_count) return out;
            Rows input(top * block_rows, hidden), both(top * block_rows, 2 * ff), inner(top * block_rows, ff);
            Rows values(top * block_rows, hidden);
            std::vector<size_t> live(top * block_rows);
            struct Release {
                Inkling2ExpertProjector* projector;
                ~Release() { projector->finish_grouped(); }
            } release{projector};
            const auto stage_up = [&](size_t g) {
                const Group& group = groups[g];
                for (size_t slot = 0; slot < group.slots; ++slot)
                    for (size_t row = 0; row < group.counts[slot]; ++row)
                        std::copy_n(x[group.routes[slot][row].token], hidden, input[slot * block_rows + row]);
                projector->stage_grouped(*gate_up.file, gate_up.name, layer, false, g & 1, group.ids, group.counts,
                                         input.data.data());
            };
            const auto finish_up = [&](size_t g) {
                const Group& group = groups[g];
                projector->read_grouped(false, g & 1, group.counts, both.data.data());
                size_t active = 0;
                for (size_t slot = 0; slot < group.slots; ++slot)
                    for (size_t row = 0; row < group.counts[slot]; ++row) live[active++] = slot * block_rows + row;
#pragma omp parallel for schedule(static)
                for (long long i = 0; i < static_cast<long long>(active); ++i)
                    situ_row(both[live[i]], gate_cap, up_cap, inner[live[i]]);
                projector->stage_grouped(*down.file, down.name, layer, true, g & 1, group.ids, group.counts,
                                         inner.data.data());
            };
            // Reduce in group order to keep each token's FP32 sum in expert order.
            const auto finish_down = [&](size_t g) {
                const Group& group = groups[g];
                projector->read_grouped(true, g & 1, group.counts, values.data.data());
                size_t active = 0;
                for (size_t slot = 0; slot < group.slots; ++slot)
                    for (size_t row = 0; row < group.counts[slot]; ++row) live[active++] = slot * block_rows + row;
                // Split by column, not row: one token can appear in two slots of a group.
                constexpr long long span = 128;
#pragma omp parallel for schedule(static)
                for (long long first = 0; first < static_cast<long long>(hidden); first += span) {
                    const size_t last = std::min(hidden, static_cast<size_t>(first + span));
                    for (size_t i = 0; i < active; ++i) {
                        const auto& route = group.routes[live[i] / block_rows][live[i] % block_rows];
                        const float* value = values[live[i]];
                        float* sum = out[route.token];
                        for (size_t j = static_cast<size_t>(first); j < last; ++j) sum[j] += route.weight * value[j];
                    }
                }
            };
            stage_up(0);
            projector->start_grouped(false, 0);
            if (group_count > 1) stage_up(1);
            projector->wait_grouped();
            if (group_count > 1) projector->start_grouped(false, 1);
            for (size_t g = 0; g < group_count; ++g) {
                finish_up(g);
                projector->wait_grouped();  // gate/up(g + 1), when there is one
                projector->start_grouped(true, g & 1);
                if (g) finish_down(g - 1);
                if (g + 2 < group_count) stage_up(g + 2);
                projector->wait_grouped();
                if (g + 2 < group_count) projector->start_grouped(false, g & 1);
            }
            finish_down(group_count - 1);
            return out;
        }
        if (block_rows) {
            Rows input(top * block_rows, hidden), both(top * block_rows, 2 * ff), inner(top * block_rows, ff);
            size_t expert = 0, at = 0;
            while (expert < experts) {
                Inkling2ExpertProjector::ExpertIds ids{};
                Inkling2ExpertProjector::ExpertRows counts{};
                std::array<const Routed*, top> routes{};
                size_t slots = 0;
                // Visit slices in expert order to keep each token's FP32 sum in the same order.
                while (slots < top && expert < experts) {
                    const auto& routed = by_expert[expert];
                    if (at == routed.size()) {
                        ++expert;
                        at = 0;
                        continue;
                    }
                    ids[slots] = expert;
                    const size_t count = std::min(block_rows, routed.size() - at);
                    counts[slots] = count;
                    routes[slots] = routed.data() + at;
                    for (size_t row = 0; row < count; ++row)
                        std::copy_n(x[routed[at + row].token], hidden, input[slots * block_rows + row]);
                    at += count;
                    ++slots;
                }
                if (!slots) break;
                if (!projector->project_grouped_blocks(*gate_up.file, gate_up.name, layer, false, ids, counts,
                                                       input.data.data(), both.data.data()))
                    fail("grouped expert gate/up projection is unavailable");
                for (size_t slot = 0; slot < slots; ++slot)
                    for (size_t row = 0; row < counts[slot]; ++row) {
                        const size_t index = slot * block_rows + row;
                        situ_row(both[index], gate_cap, up_cap, inner[index]);
                    }
                if (!projector->project_grouped_blocks(*down.file, down.name, layer, true, ids, counts,
                                                       inner.data.data(), input.data.data()))
                    fail("grouped expert down projection is unavailable");
                for (size_t slot = 0; slot < slots; ++slot)
                    for (size_t row = 0; row < counts[slot]; ++row) {
                        const auto& route = routes[slot][row];
                        const float* value = input[slot * block_rows + row];
                        for (size_t j = 0; j < hidden; ++j) out[route.token][j] += route.weight * value[j];
                    }
            }
            return out;
        }
        for (size_t e = 0; e < experts; ++e) {
            const auto& routed = by_expert[e];
            if (routed.empty()) continue;
            Rows input(routed.size(), hidden);
            for (size_t i = 0; i < routed.size(); ++i)
                std::copy_n(x[routed[i].token], hidden, input[i]);
            const auto apply = [&](const Projection& p, const Rows& a, bool is_down) {
                Rows y;
                if (projector) y.data = projector->project_block(*p.file, p.name, layer, is_down, e,
                                                                a.data.data(), a.count());
                if (y.data.empty()) return p.apply(a, e);
                y.width = p.outputs;
                if (y.data.size() != y.width * a.count()) fail("unexpected expert block output shape");
                return y;
            };
            const Rows values = apply(down, situ(apply(gate_up, input, false), gate_cap, up_cap), true);
            for (size_t i = 0; i < routed.size(); ++i)
                for (size_t j = 0; j < hidden; ++j)
                    out[routed[i].token][j] += routed[i].weight * values[i][j];
        }
        return out;
    }
};

Rows attend(const Layer& l, const Rows& q, const Rows& r, const Rows& sinks, size_t pos,
            const VisitCache& cache, const Pending& next) {
    Rows out(q.count(), q_width);
    const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
#pragma omp parallel for
    for (long long h = 0; h < static_cast<long long>(heads); ++h) {
        const size_t kh = static_cast<size_t>(h) / (heads / kv_heads);
        for (size_t t = 0; t < q.count(); ++t) {
            const size_t row = pos + t;
            const size_t start = l.sliding && row >= window ? row + 1 - window : 0;
            float* result = out[t] + h * dim;
            float peak = sinks[t][h], mass = 1.0f;
            const float* query = q[t] + h * dim;
            for (size_t j = start; j <= row; ++j) {
                const uint16_t* key = j < pos ? cache.key.data() + cache.offset(j)
                                             : next.key.data() + (j - pos) * kv_width;
                const uint16_t* val = j < pos ? cache.value.data() + cache.offset(j)
                                             : next.value.data() + (j - pos) * kv_width;
                key += kh * dim;
                val += kh * dim;
                float score = 0;
                for (size_t k = 0; k < dim; ++k) score += query[k] * bf16_to_f32(key[k]);
                score *= scale;
                const size_t distance = row - j;
                if (distance < extent)
                    score += dot(r[t] + h * rel_dim, l.rel_bias.data() + distance * rel_dim, rel_dim);
                const float new_peak = std::max(peak, score);
                const float keep = std::exp(peak - new_peak), add = std::exp(score - new_peak);
                mass = keep * mass + add;
                for (size_t k = 0; k < dim; ++k)
                    result[k] = keep * result[k] + add * bf16_to_f32(val[k]);
                peak = new_peak;
            }
            for (size_t k = 0; k < dim; ++k) result[k] /= mass;
        }
    }
    return out;
}

void encode_kv(const Rows& rows, std::vector<uint16_t>& out) {
    out.resize(rows.data.size());
    std::transform(rows.data.begin(), rows.data.end(), out.begin(), f32_to_bf16);
}

}  // namespace

struct Inkling2Host::Impl {
    GgufFile file;
    std::unique_ptr<Inkling2ExpertProjector> expert_projector;
    size_t capacity, pos = 0;
    uint64_t sequence = 0;
    float eps, qk_clip, rel_clip, log_floor, alpha, route_scale, gate_cap, up_cap, logit_cap;
    std::vector<Layer> weights;
    std::array<VisitCache, visits> cache;
    std::unique_ptr<PendingBlock> tentative;
    Projection lm_head;
    std::vector<float> embedding_norm, out_norm, out_query;

    explicit Impl(const std::string& path, size_t context, std::unique_ptr<Inkling2ExpertProjector> projector)
        : file(path), expert_projector(std::move(projector)), capacity(context) {
        const std::string arch = "inkling2_loop";
        if (file.kv_str("general.architecture") != arch) fail("expected a fixed-loop Inkling 2 Mini GGUF");
        const auto key = [&](const char* name) { return arch + "." + name; };
        const std::pair<const char*, uint64_t> fixed[] = {
            {"block_count", layers}, {"embedding_length", hidden}, {"expert_feed_forward_length", ff},
            {"expert_count", experts}, {"expert_used_count", top}, {"attention.key_length", dim},
            {"attention.value_length", dim}, {"attention.relative_length", rel_dim},
            {"attention.relative_extent", extent}, {"attention.sliding_window", window},
            {"shortconv.l_cache", taps}, {"attn_res.block_size", block_size}, {"loop.start", core_start},
            {"loop.end", core_end}, {"loop.count", passes}, {"loop.gate_rank", rank},
            {"loop.varying", 0}, {"vocab_size", padded_vocab}, {"vocab_size_unpadded", vocab},
            {"context_length", max_context}, {"rope.dimension_count", 0},
        };
        for (const auto& [name, expected] : fixed)
            if (file.kv_u64(key(name)) != expected) fail("unsupported " + key(name));
        const auto head_counts = [&](const char* name, uint32_t count) {
            if (file.kv_u32_array(key(name)) != std::vector<uint32_t>(layers, count))
                fail("unsupported " + key(name));
        };
        head_counts("attention.head_count", heads);
        head_counts("attention.head_count_kv", kv_heads);
        const auto pattern = file.kv_u32_array(key("attention.sliding_window_pattern"));
        if (pattern.size() != layers) fail("invalid sliding-window pattern length");
        for (size_t l = 0; l < layers; ++l)
            if (pattern[l] != static_cast<uint32_t>(l % block_size != block_size - 1))
                fail("unsupported sliding-window pattern");
        if (!capacity || capacity > max_context) fail("invalid context capacity");
        const auto coefficient = [&](const char* name) {
            const float value = static_cast<float>(file.kv_float(key(name)));
            if (!std::isfinite(value) || value <= 0) fail("invalid " + key(name));
            return value;
        };
        eps = coefficient("attention.layer_norm_rms_epsilon");
        qk_clip = coefficient("attention.qk_clip");
        rel_clip = coefficient("attention.relative_clip");
        log_floor = static_cast<float>(file.kv_u64(key("attention.log_scale_floor")));
        if (log_floor <= 0) fail("invalid log scale floor");
        alpha = coefficient("attention.log_scale_alpha");
        route_scale = coefficient("expert_weights_scale");
        gate_cap = coefficient("activation.situ_beta");
        up_cap = coefficient("activation.situ_linear_beta");
        logit_cap = coefficient("final_logit_softcapping");
        checked_tensor(file, "token_embd.weight", {hidden, padded_vocab});
        embedding_norm = load_small(file, "token_embd_norm.weight", {hidden});
        out_norm = load_small(file, "output_norm.weight", {hidden});
        out_query = load_small(file, "output_res_query.weight", {hidden});
        lm_head = Projection(file, "output.weight", hidden, padded_vocab);
        weights.reserve(layers);
        for (size_t l = 0; l < layers; ++l) {
            const auto& layer = weights.emplace_back(file, l);
            if (expert_projector && (!layer.gate_up.expert_scales.empty() || !layer.down.expert_scales.empty()))
                fail("NVFP4 experts currently require the host route");
        }
    }

    Rows layer(size_t i, size_t visit, const Rows& x, std::vector<Rows>& blocks,
               const std::vector<Rows>& previous, const Rows& previous_end,
               const VisitCache& state, Pending& next, bool verify) const {
        const Layer& l = weights[i];
        auto inputs = i ? mix(blocks, x, l.attn_query, eps) : std::vector<Rows>{x};
        std::vector<Rows> messages;
        if (previous_end.width) messages = mix(previous, previous_end, l.msg_query, eps);
        // Match singleton target projections during batched verification on the aligned image.
        const bool verify_dense = verify && x.count() > 1 && expert_projector && expert_projector->aligned_attention_chunks();
        const bool singleton = expert_projector && expert_projector->singleton_attention_available();
        const bool match_singleton = singleton && verify_dense;
        const bool dense = (x.count() > 1 || singleton) && (!verify_dense || match_singleton) &&
                           expert_projector && expert_projector->dense_attention_available();
        const auto host_projection = [verify_dense](const Projection& projection, const Rows& rows) {
            if (!verify_dense) return projection.apply(rows);
            Rows result(rows.count(), projection.outputs);
            Rows singleton(1, rows.width);
            for (size_t token = 0; token < rows.count(); ++token) {
                std::copy_n(rows[token], rows.width, singleton[0]);
                const auto projected = projection.apply(singleton);
                std::copy_n(projected[0], result.width, result[token]);
            }
            return result;
        };
        const bool shared_input = l.sliding && messages.empty();
        const auto normalized_input = [&](size_t n) {
            Rows u = inputs[l.sliding ? 0 : n];
            if (!messages.empty()) l.controller[n].apply(u, messages[n], eps);
            norm(u, eps, l.attn_norm[n]);
            return u;
        };
        next.conv = state.conv;
        const auto tape = [&](size_t leg) { return verify ? &next.conv_inputs[leg] : nullptr; };
        const Rows q_input = normalized_input(0);
        Rows q, k, v, r, sinks, gate;
        bool projected_inputs = dense;
        if (dense) {
            const size_t count = x.count();
            const Rows k_input = shared_input ? Rows{} : normalized_input(1);
            const Rows v_input = shared_input ? Rows{} : normalized_input(2);
            q = Rows(count, q_width);
            k = Rows(count, kv_width);
            v = Rows(count, kv_width);
            r = Rows(count, heads * rel_dim);
            sinks = Rows(count, heads);
            gate = Rows(count, q_width);
            const float* q_data = q_input.data.data();
            const std::array<const float*, 3> normalized{q_data,
                shared_input ? q_data : k_input.data.data(), shared_input ? q_data : v_input.data.data()};
            if (match_singleton) {
                for (size_t t = 0; t < count; ++t) {
                    const std::array<const float*, 3> row{normalized[0] + t * hidden,
                        normalized[1] + t * hidden, normalized[2] + t * hidden};
                    if (!expert_projector->project_attention_inputs(file, i, 1, row,
                            {q[t], k[t], v[t], r[t], sinks[t], gate[t]}))
                        fail("singleton verification input projection is unavailable");
                }
            } else if (!expert_projector->project_attention_inputs(file, i, count, normalized,
                           {q.data.data(), k.data.data(), v.data.data(), r.data.data(), sinks.data.data(), gate.data.data()})) {
                fail("dense attention input projection is unavailable");
            }
            k = l.conv[0].run(k, l.sliding ? nullptr : &k_input, next.conv[0], tape(0));
            v = l.conv[1].run(v, l.sliding ? nullptr : &v_input, next.conv[1], tape(1));
        } else {
            const Rows k_input = shared_input ? Rows{} : normalized_input(1);
            const Rows v_input = shared_input ? Rows{} : normalized_input(2);
            const Rows& k_source = shared_input ? q_input : k_input;
            const Rows& v_source = shared_input ? q_input : v_input;
            projected_inputs = (x.count() == 1 || verify_dense) && l.project_host_attention_inputs(
                {q_input[0], k_source[0], v_source[0]}, x.count(), {&q, &k, &v, &r, &sinks, &gate});
            if (!projected_inputs) {
                q = host_projection(l.qkv[0], q_input);
                k = host_projection(l.qkv[1], k_source);
                v = host_projection(l.qkv[2], v_source);
                r = host_projection(l.relative, q_input);
            }
            k = l.conv[0].run(k, l.sliding ? nullptr : &k_source, next.conv[0], tape(0));
            v = l.conv[1].run(v, l.sliding ? nullptr : &v_source, next.conv[1], tape(1));
        }
        for (size_t t = 0; t < x.count(); ++t) {
            const float tau = l.sliding ? 1.0f :
                1.0f + alpha * std::log(std::max(1.0f, static_cast<float>(pos + t + 1) / log_floor));
            for (size_t h = 0; h < heads; ++h) {
                clip(q[t] + h * dim, dim, qk_clip, l.q_clip.data() + h * dim);
                clip(r[t] + h * rel_dim, rel_dim, rel_clip, l.rel_clip.data() + h * rel_dim);
                for (size_t j = 0; j < dim; ++j) q[t][h * dim + j] *= tau;
                for (size_t j = 0; j < rel_dim; ++j) r[t][h * rel_dim + j] *= tau;
            }
            for (size_t h = 0; h < kv_heads; ++h) {
                clip(k[t] + h * dim, dim, qk_clip);
                rms(v[t] + h * dim, dim, eps);
            }
        }
        encode_kv(k, next.key);
        encode_kv(v, next.value);
        if (!projected_inputs) sinks = host_projection(l.sink, q_input);
        Rows a;
        if (expert_projector) {
            a.data = expert_projector->project_attention({i, pos, x.count(), state.ring, l.sliding ? window : 0,
                q.data.data(), r.data.data(), sinks.data.data(), l.rel_bias.data(),
                state.key.data(), state.value.data(), next.key.data(), next.value.data(),
                l.gate_up.file->tensor(l.gate_up.name).type == GgufFile::Type::IQ4_NL, visit, sequence,
                &state.key, state.key.can_retire_keys() ? CachedKvRows::retire_key_rows : nullptr});
        }
        if (a.data.empty()) {
            if (state.key.keys_retired()) fail("the NPU owns archived attention keys");
            a = attend(l, q, r, sinks, pos, state, next);
        } else {
            a.width = q_width;
            if (a.data.size() != x.count() * q_width) fail("unexpected attention output shape");
        }
        if (!projected_inputs) gate = host_projection(l.gate, q_input);
        const long long gated = static_cast<long long>(a.data.size());
#pragma omp parallel for schedule(static) if (gated >= 65536)
        for (long long j = 0; j < gated; ++j) a.data[j] *= sigmoid(gate.data[j]);
        Rows attention_output;
        if (dense) {
            attention_output = Rows(x.count(), hidden);
            if (match_singleton) {
                for (size_t t = 0; t < x.count(); ++t)
                    if (!expert_projector->project_attention_output(file, i, 1, a[t], attention_output[t]))
                        fail("singleton verification output projection is unavailable");
            } else if (!expert_projector->project_attention_output(file, i, x.count(), a.data.data(),
                                                                   attention_output.data.data())) {
                fail("dense attention output projection is unavailable");
            }
        } else {
            attention_output = host_projection(l.output, a);
        }
        a = l.conv[2].run(attention_output, nullptr, next.conv[2], tape(2));
        if (i % block_size == 0) blocks.push_back(x);
        else
            for (size_t j = 0; j < a.data.size(); ++j) a.data[j] += x.data[j];
        Rows f = std::move(mix(blocks, a, l.ffn_query, eps).front());
        if (!messages.empty()) l.controller[3].apply(f, messages[3], eps);
        norm(f, eps, l.ffn_norm);
        if (match_singleton && l.gate_up.file->tensor(l.gate_up.name).type == GgufFile::Type::Q8_0) {
            Rows projected(f.count(), hidden), row(1, hidden);
            for (size_t t = 0; t < f.count(); ++t) {
                std::copy_n(f[t], hidden, row[0]);
                const auto result = l.moe(row, route_scale, gate_cap, up_cap, i, expert_projector.get());
                std::copy_n(result[0], hidden, projected[t]);
            }
            f = std::move(projected);
        } else {
            f = l.moe(f, route_scale, gate_cap, up_cap, i, expert_projector.get());
        }
        f = l.conv[3].run(f, nullptr, next.conv[3], tape(3));
        for (size_t j = 0; j < a.data.size(); ++j) a.data[j] += f.data[j];
        return a;
    }

    Rows embed(const std::vector<int>& ids) const {
        if (ids.empty() || ids.size() > capacity - pos) fail("token block exceeds context capacity");
        for (int id : ids)
            if (id < 0 || static_cast<size_t>(id) >= vocab) fail("token ID outside the model vocabulary");
        Rows x(ids.size(), hidden);
        for (size_t t = 0; t < ids.size(); ++t)
            file.embed_row("token_embd.weight", static_cast<size_t>(ids[t]), hidden, x[t]);
        norm(x, eps, embedding_norm);
        return x;
    }

    void require_idle() const {
        if (tentative) fail("a verify pass is pending");
    }

    std::vector<float> append(const std::vector<int>& ids, bool want_logits, const VisitObserver& observe,
                              std::vector<float>* draft_features) {
        require_idle();
        return append_rows(embed(ids), want_logits, observe, false, draft_features);
    }

    std::vector<float> append_embeddings(std::vector<float> embeddings, bool want_logits,
                                         const VisitObserver& observe, std::vector<float>* draft_features) {
        require_idle();
        if (embeddings.empty() || embeddings.size() % hidden) fail("invalid image embedding shape");
        if (embeddings.size() / hidden > capacity - pos) fail("image embeddings exceed context capacity");
        for (float value : embeddings)
            if (!std::isfinite(value)) fail("nonfinite image embedding");
        Rows x;
        x.width = hidden;
        x.data = std::move(embeddings);
        return append_rows(std::move(x), want_logits, observe, false, draft_features);
    }

    std::vector<float> verify(const std::vector<int>& ids, std::vector<float>* draft_features) {
        require_idle();
        if (ids.empty() || ids.size() > verify_limit) fail("invalid verify block width");
        return append_rows(embed(ids), true, {}, true, draft_features);
    }

    std::vector<float> logits_for(const Rows& normalized) const {
        std::vector<float> logits;
        if (expert_projector) {
            for (size_t row = 0; row < normalized.count();) {
                std::vector<float> part;
                if (row + 1 < normalized.count())
                    part = expert_projector->project_logits_pair(file, normalized[row]);
                size_t count = 2;
                if (part.empty()) {
                    part = expert_projector->project_logits(file, normalized[row]);
                    count = 1;
                }
                if (part.empty() && row == 0) break;  // this projector leaves the head on the host
                if (part.size() != count * vocab) fail("unexpected output vocabulary size");
                if (normalized.count() == 1) logits = std::move(part);
                else logits.insert(logits.end(), part.begin(), part.end());
                row += count;
            }
        }
        if (logits.empty()) logits = lm_head.apply(normalized, 0, vocab).data;
        for (float& score : logits) score = softcap(score, logit_cap);
        return logits;
    }

    std::vector<float> append_rows(Rows x, bool want_logits, const VisitObserver& observe, bool verify,
                                   std::vector<float>* draft_features) {
        const size_t count = x.count();
        constexpr std::array<size_t, 3> tap_visits{5, 23, 41};
        static_assert(visits == 48);
        static_assert(draft_feature_width == (tap_visits.size() + 1) * hidden);
        std::vector<float> features;
        if (draft_features) features.resize(count * draft_feature_width);
        const auto capture = [&](size_t tap, const Rows& rows) {
            for (size_t row = 0; row < count; ++row)
                std::copy_n(rows[row], hidden, features.data() + row * draft_feature_width + tap * hidden);
        };
        std::vector<Rows> blocks, previous;
        Rows previous_end;
        PendingBlock pending;
        pending.rows = count;
        size_t visit = 0, tap = 0;
        const auto advance = [&](size_t i) {
            if (draft_features && tap < tap_visits.size() && visit == tap_visits[tap]) capture(tap++, x);
            pending.rings[visit] = weights[i].sliding ? std::min(window, capacity) : 0;
            x = layer(i, visit, x, blocks, previous, previous_end, cache[visit], pending.states[visit], verify);
            if (observe) observe(visit, i, x.data);
            ++visit;
        };
        for (size_t i = 0; i < core_start; ++i) advance(i);
        const Rows prelude = x;
        const auto prelude_blocks = blocks;
        for (size_t pass = 0; pass < passes; ++pass) {
            x = prelude;
            blocks = prelude_blocks;
            for (size_t i = core_start; i < core_end; ++i) advance(i);
            previous.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
            previous_end = x;
        }
        previous.clear();
        previous_end = {};
        for (size_t i = core_end; i < layers; ++i) advance(i);

        std::vector<float> logits;
        if (verify || draft_features) {
            Rows all = std::move(mix(blocks, x, out_query, eps).front());
            norm(all, eps, out_norm);
            if (draft_features) capture(tap_visits.size(), all);
            if (verify || (want_logits && count == 1)) logits = logits_for(all);
            else if (want_logits) {
                Rows last(1, hidden);
                std::copy_n(all[count - 1], hidden, last[0]);
                logits = logits_for(last);
            }
        } else if (want_logits) {
            Rows last(1, hidden);
            std::copy_n(x[x.count() - 1], hidden, last[0]);
            std::vector<Rows> bank;
            for (const auto& b : blocks) {
                bank.emplace_back(1, hidden);
                std::copy_n(b[b.count() - 1], hidden, bank.back()[0]);
            }
            last = std::move(mix(bank, last, out_query, eps).front());
            norm(last, eps, out_norm);
            logits = logits_for(last);
        }
        if (verify) tentative = std::make_unique<PendingBlock>(std::move(pending));
        else commit(pending, count);
        if (draft_features) *draft_features = std::move(features);
        return logits;
    }

    void commit(PendingBlock& pending, size_t count) {
        // Reserve every visit before changing any sequence data. A failed block
        // leaves the previously committed position, KV and convolution state usable.
        for (size_t i = 0; i < visits; ++i) {
            auto& c = cache[i];
            const size_t limit = pending.rings[i] ? pending.rings[i] : capacity;
            const size_t required = std::min(limit, pos + count) * kv_width;
            for (auto* lanes : {&c.key, &c.value})
                lanes->prepare(required, limit * kv_width);
        }
        for (size_t i = 0; i < visits; ++i) {
            auto& c = cache[i];
            auto& next = pending.states[i];
            c.ring = pending.rings[i];
            const size_t rows = c.ring ? std::min(c.ring, pos + count) : pos + count;
            c.key.resize(rows * kv_width);
            c.value.resize(rows * kv_width);
            const size_t first = c.ring && count > c.ring ? count - c.ring : 0;
            for (size_t t = first; t < count; ++t) {
                const size_t offset = c.offset(pos + t);
                std::copy_n(next.key.data() + t * kv_width, kv_width, c.key.data() + offset);
                std::copy_n(next.value.data() + t * kv_width, kv_width, c.value.data() + offset);
            }
            if (count < pending.rows) {
                const size_t take = std::min(count, taps - 1), keep = taps - 1 - take;
                for (size_t leg = 0; leg < next.conv.size(); ++leg) {
                    auto& dst = next.conv[leg];
                    const auto& old = c.conv[leg];
                    const size_t width = dst.size() / (taps - 1);
                    if (keep && !old.empty()) std::copy_n(old.end() - keep * width, keep * width, dst.begin());
                    else std::fill_n(dst.begin(), keep * width, 0.0f);
                    std::copy_n(next.conv_inputs[leg].begin() + (count - take) * width, take * width,
                                dst.begin() + keep * width);
                }
            }
            c.conv.swap(next.conv);
        }
        pos += count;
    }

    void accept(size_t count) {
        if (!tentative || count > tentative->rows) fail("invalid accepted verify prefix");
        if (count) commit(*tentative, count);
        tentative.reset();
    }
};

Inkling2Host::Inkling2Host(const std::string& gguf, size_t context,
                           std::unique_ptr<Inkling2ExpertProjector> expert_projector)
    : impl_(std::make_unique<Impl>(gguf, context, std::move(expert_projector))) {}
Inkling2Host::~Inkling2Host() = default;

std::vector<float> Inkling2Host::append(const std::vector<int>& ids, bool logits, const VisitObserver& observe,
                                      std::vector<float>* draft_features) {
    return impl_->append(ids, logits, observe, draft_features);
}

std::vector<float> Inkling2Host::append_embeddings(std::vector<float> embeddings, bool logits,
                                                   const VisitObserver& observe, std::vector<float>* draft_features) {
    return impl_->append_embeddings(std::move(embeddings), logits, observe, draft_features);
}

std::vector<float> Inkling2Host::verify(const std::vector<int>& ids, std::vector<float>* draft_features) {
    return impl_->verify(ids, draft_features);
}
void Inkling2Host::accept(size_t count) { impl_->accept(count); }

void Inkling2Host::reset() {
    impl_->tentative.reset();
    impl_->pos = 0;
    impl_->cache = {};
    ++impl_->sequence;
    if (impl_->expert_projector) impl_->expert_projector->reset_attention_cache(impl_->sequence);
}

size_t Inkling2Host::position() const { return impl_->pos; }
const GgufFile& Inkling2Host::weights() const { return impl_->file; }
bool Inkling2Host::block_experts_available() const {
    return impl_->expert_projector && impl_->expert_projector->block_experts_available();
}

}  // namespace open_qwen36
