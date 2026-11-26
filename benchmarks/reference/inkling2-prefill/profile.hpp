#pragma once

#include <array>
#include <chrono>
#include <cstdio>

namespace inkling_prefill_profile {
using Clock = std::chrono::steady_clock;
enum Part {
    block, residual_mix, controller, controller_update, conv, attention, attention_gate,
    attention_host, layer_norm, qkv_post, kv_commit, embed, head, head_host,
    dense_input, dense_output, moe_total, route, gather, situ, reduce,
    group_total, weight_prepare, activation_pack, activation_sync, instruction_sync,
    group_unpack, configure, wait, readback_sync, finite_scan,
    attention_kv, attention_query, attention_sliding_dispatch, attention_full_dispatch, attention_merge,
    attention_old_prepare, attention_old_upload, attention_old_dispatch_4k, attention_old_dispatch_16k,
    attention_old_merge, attention_near_prepare, attention_near_dispatch, attention_near_merge,
    dense_host_input, dense_host_output, attention_kv_sync, dispatch_configure, dispatch_wait, dispatch_readback,
    count
};
inline constexpr const char* names[] = {
    "block", "residual_mix", "controller", "controller_update", "conv", "attention", "attention_gate",
    "attention_host", "layer_norm", "qkv_post", "kv_commit", "embed", "head", "head_host",
    "dense_input", "dense_output",
    "moe_total", "route", "gather", "situ", "reduce",
    "group_total", "weight_prepare", "activation_pack", "activation_sync", "instruction_sync",
    "group_unpack", "configure", "wait", "readback_sync", "finite_scan",
    "attention_kv", "attention_query", "attention_sliding_dispatch", "attention_full_dispatch", "attention_merge",
    "attention_old_prepare", "attention_old_upload", "attention_old_dispatch_4k", "attention_old_dispatch_16k",
    "attention_old_merge", "attention_near_prepare", "attention_near_dispatch", "attention_near_merge",
    "dense_host_input", "dense_host_output", "attention_kv_sync", "dispatch_configure", "dispatch_wait",
    "dispatch_readback"
};
static_assert(sizeof(names) / sizeof(names[0]) == count);
inline thread_local std::array<double, count> ms{};
inline thread_local std::array<size_t, count> calls{};
inline thread_local bool grouped = false;

inline void add(Part part, Clock::time_point start) {
    ms[part] += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    ++calls[part];
}

struct Measure {
    Part part;
    Clock::time_point start = Clock::now();
    ~Measure() { add(part, start); }
};

struct Group {
    Measure total{group_total};
    Group() { grouped = true; }
    ~Group() { grouped = false; }
};

struct GroupInterface {
    Clock::time_point start = Clock::now();
    ~GroupInterface() { ms[group_total] += std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
};

inline void reset() { ms = {}; calls = {}; }

inline void print(size_t position, size_t rows, bool verify = false) {
    std::fprintf(stderr, "inkling_prefill_profile position=%zu rows=%zu", position, rows);
    for (size_t p = 0; p < count; ++p)
        std::fprintf(stderr, " %s_ms=%.6f %s_calls=%zu", names[p], ms[p], names[p], calls[p]);
    std::fprintf(stderr, " verify=%d\n", static_cast<int>(verify));
}
}
