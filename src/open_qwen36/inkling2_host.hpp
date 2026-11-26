#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace open_qwen36 {

class GgufFile;

struct Inkling2AttentionProjections {
    // Token-major rows for Q, K, V, relative query, attention sinks and output gate.
    float *query, *key, *value, *relative, *sinks, *gate;
};

struct Inkling2AttentionView {
    size_t layer, position, rows, ring, window;
    // Token-major Q [24*128], relative Q [24*16], sinks [24] and table [1024,16].
    const float *query, *relative, *sinks, *table;
    // BF16 rows of eight 128-wide heads. Pending rows start at position.
    const uint16_t *cached_key, *cached_value, *pending_key, *pending_value;
    bool iq4_gate_up = false;
    // The visit identifies its own cache even when a weight layer repeats.
    size_t visit = 0;
    uint64_t sequence = 0;
    // Optional ownership hook for immutable, page-backed full-attention keys.
    const void* key_storage = nullptr;
    bool (*retire_key_rows)(const void*, size_t, size_t) noexcept = nullptr;
};

class Inkling2ExpertProjector {
public:
    using ExpertIds = std::array<size_t, 8>;
    using ExpertRows = std::array<size_t, 8>;
    virtual ~Inkling2ExpertProjector() = default;
    // Each expert gets one contiguous activation row (2,304 gate/up or 1,152 down)
    // and writes one 2,304-value output row, in the order of ids.
    virtual void project(const GgufFile& file, const std::string& tensor, size_t layer, bool down,
                         const ExpertIds& ids, const float* input, float* output) = 0;
    // One row through gate/up, SiTU (gate and up caps) and down in one launch, writing one 2,304-value
    // row per expert in the order of ids. False selects project() for each leg.
    virtual bool project_routed(const GgufFile&, const std::string&, const GgufFile&, const std::string&, size_t,
                                const ExpertIds&, float, float, const float*, float*) { return false; }
    // Rows assigned to one expert, in token order. An empty return selects the host matrix multiply.
    virtual std::vector<float> project_block(const GgufFile&, const std::string&, size_t, bool, size_t,
                                             const float*, size_t) { return {}; }
    virtual size_t grouped_block_rows() const { return 0; }
    // Each slot has grouped_block_rows() contiguous input and output rows.
    // Only the first counts[slot] rows are read or written. A count of zero leaves the slot unused.
    virtual bool project_grouped_blocks(const GgufFile&, const std::string&, size_t, bool,
                                        const ExpertIds&, const ExpertRows&, const float*, float*) { return false; }
    // Overlapped grouped prefill. One launch is in flight at a time. Each projection leg has
    // two lanes of buffers, so the host can stage one group and finish another during a launch.
    virtual bool grouped_overlap() const { return false; }
    // Pack one group into a lane (0 or 1). The lane must be idle and its last result read.
    virtual void stage_grouped(const GgufFile&, const std::string&, size_t, bool, size_t,
                               const ExpertIds&, const ExpertRows&, const float*) {}
    virtual void start_grouped(bool, size_t) {}
    // Wait for the launch in flight, if there is one.
    virtual void wait_grouped() {}
    // Validate a finished lane and unpack grouped_block_rows() rows per slot.
    virtual void read_grouped(bool, size_t, const ExpertRows&, float*) {}
    // Leave nothing in flight and release every lane, including after an error.
    virtual void finish_grouped() noexcept {}
    virtual bool block_experts_available() const { return false; }
    virtual bool dense_attention_available() const { return false; }
    virtual bool singleton_attention_available() const { return false; }
    virtual bool aligned_attention_chunks() const { return false; }
    // Normalized Q, K and V inputs share a pointer when the same rows feed all three.
    virtual bool project_attention_inputs(const GgufFile&, size_t, size_t, const std::array<const float*, 3>&,
                                          const Inkling2AttentionProjections&) { return false; }
    virtual bool project_attention_output(const GgufFile&, size_t, size_t, const float*, float*) { return false; }
    // Loop controllers first .. first + count - 1 (q, k, v, ffn) of a core layer over token rows:
    // out[n] = x[n] updated from message[n]. out[n] may equal x[n]. False selects the host route.
    virtual bool apply_loop_controllers(const GgufFile&, size_t, size_t, size_t, size_t, float,
                                        const float* const*, const float* const*, float* const*) { return false; }
    // Returns token-major attention rows, or an empty vector for the host route.
    virtual void reset_attention_cache(uint64_t) noexcept {}
    virtual std::vector<float> project_attention(const Inkling2AttentionView&) { return {}; }
    // Project the final normalized row. An empty return selects the host head.
    virtual std::vector<float> project_logits(const GgufFile&, const float*) { return {}; }
    // Two consecutive normalized rows. An empty return selects the single-row path.
    virtual std::vector<float> project_logits_pair(const GgufFile&, const float*) { return {}; }
};

/// Native host execution of the fixed-loop Inkling 2 Mini text graph.
/// Weight views are shared by physical layer. A sequence owns one cache per visit.
class Inkling2Host {
public:
    static constexpr size_t layer_count = 24;
    static constexpr size_t embedding_width = 2304;
    static constexpr size_t vocabulary_size = 200058;
    static constexpr size_t max_context = 1048576;
    static constexpr size_t verify_limit = 8;
    // Verify passes wider than verify_limit use the block route: close to decode, not bit-equal.
    static constexpr size_t wide_verify_limit = 64;
    static constexpr size_t draft_feature_width = 4 * embedding_width;
    using VisitObserver = std::function<void(size_t visit, size_t layer, const std::vector<float>& residual)>;

    explicit Inkling2Host(const std::string& gguf, size_t context,
                          std::unique_ptr<Inkling2ExpertProjector> expert_projector = {});
    ~Inkling2Host();
    Inkling2Host(const Inkling2Host&) = delete;
    Inkling2Host& operator=(const Inkling2Host&) = delete;

    /// Append a block of token IDs. Returns the last row's logits if requested.
    /// Optional DSpark features are token-major, with 2,304 values for each of visits 5, 23, 41 and 48.
    std::vector<float> append(const std::vector<int>& ids, bool logits = true,
                              const VisitObserver& observe = {}, std::vector<float>* draft_features = nullptr);
    /// Append token-major rows from the image projector directly to the residual stream.
    std::vector<float> append_embeddings(std::vector<float> embeddings, bool logits = true,
                                         const VisitObserver& observe = {}, std::vector<float>* draft_features = nullptr);
    /// Evaluate up to wide_verify_limit tokens and return every row's logits. State stays tentative until accept().
    /// Only pass features from the accepted prefix to the drafter.
    std::vector<float> verify(const std::vector<int>& ids, std::vector<float>* draft_features = nullptr);
    void accept(size_t count);
    void reset();
    size_t position() const;
    bool block_experts_available() const;
    const GgufFile& weights() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace open_qwen36
