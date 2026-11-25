#include "open_qwen36/inkling2_npu.hpp"

#include "open_qwen36/gguf_file.hpp"
#include "open_qwen36/pools.hpp"
#include "open_qwen36/q4nx_file.hpp"
#include "open_qwen36/inkling2_wide_attention.hpp"
#include "nlohmann/json.hpp"
#include "stream_patch.hpp"
#include "open_qwen36/inkling2_singleton_io.hpp"

#include "xrt/experimental/xrt_ext.h"
#include "xrt/experimental/xrt_xclbin.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

#include <algorithm>
#include <array>
#include <bitset>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <new>
#include <stdexcept>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX2__) || defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace open_qwen36 {
namespace {

constexpr size_t width = 2304, expert_width = 1152, experts = 128, slots = 8, layers = Inkling2Host::layer_count;
constexpr size_t values = slots * width;
constexpr size_t vocab = Inkling2Host::vocabulary_size, file_vocab = 201024, head_band = 128;
constexpr size_t head_rows = (vocab + head_band - 1) / head_band * head_band;
constexpr size_t head_bytes = (head_rows / 32) * (width / 256) * 9216;
constexpr size_t align_bo = 1u << 20;  // XDNA buffer alignment
namespace block_layout = pools::inkling2_block;
constexpr size_t compact_m = slots * block_layout::decode_tile_m;

size_t padded_bo(size_t n) { return (n + align_bo - 1) / align_bo * align_bo; }

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error("Inkling 2 NPU: " + message); }

std::vector<uint32_t> read_stream(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    const auto bytes = in.tellg();
    if (bytes < 16 || bytes > 1024 * 1024 || bytes % 4) fail("invalid stream size for " + path.string());
    std::vector<uint32_t> words(static_cast<size_t>(bytes) / 4);
    in.seekg(0);
    if (!in.read(reinterpret_cast<char*>(words.data()), bytes)) fail("could not read " + path.string());
    return words;
}

struct Stream {
    xrt::bo instruction;
    size_t count = 0, expert_bytes = 0;
    std::vector<stream_patch::MoePatch> fills;

    Stream(const xrt::device& device, uint32_t group, const std::filesystem::path& path,
           size_t cores, size_t padded_k, bool iq4) {
        const auto words = read_stream(path);
        expert_bytes = (width / 64) * (padded_k / 64) * (iq4 ? 2560 : 5120);
        const size_t per_core = expert_bytes / cores;
        std::vector<bool> seen(slots * cores);
        size_t bd_write = 0;
        for (size_t i = 4; i < words.size(); i += stream_patch::op_len(words[i])) {
            if (words[i] == 1) bd_write = i;
            if (words[i] != 0x81) continue;
            if (i + 11 >= words.size()) fail("truncated DMA patch in " + path.string());
            if (words[i + 8] != 0) continue;
            const size_t offset = words[i + 10], slot = offset / expert_bytes, within = offset % expert_bytes;
            const size_t stripe = slot * cores + within / per_core;
            if (slot >= slots || within % per_core || words[i + 11] || seen[stripe] || !bd_write ||
                words[bd_write + 2] + 4 != words[i + 6] || words[bd_write + 4] != per_core / 4)
                fail("invalid weight fill in " + path.string());
            seen[stripe] = true;
            fills.push_back({i + 10, slot, 0, 0, within});
        }
        if (fills.size() != seen.size()) fail("missing weight fills in " + path.string());
        count = words.size();
        instruction = xrt::bo(device, count * sizeof(uint32_t), xrt::bo::flags::cacheable, group);
        std::memcpy(instruction.map<void*>(), words.data(), count * sizeof(uint32_t));
    }
};

struct Pool {
    xrt::bo bo;
    std::bitset<experts> packed;
};

struct CompactStream {
    xrt::bo instruction;
    size_t count = 0, bytes = 0, stride = 0;
    std::vector<std::pair<size_t, uint32_t>> fills;

    CompactStream(const xrt::device& device, uint32_t group, const std::filesystem::path& path,
                  size_t rows, size_t cols, bool iq4, bool grouped, size_t outputs = width, bool split_scale = false) {
        const auto words = read_stream(path);
        bytes = block_layout::weight_bytes(iq4, outputs, cols, split_scale);
        stride = (bytes + 4095) / 4096 * 4096;
        const std::array<size_t, 3> spans{bytes * (grouped ? slots : 1), block_layout::input_bytes(rows, cols),
                                         rows * outputs * sizeof(float)};
        const std::array<size_t, 3> slices = grouped
            ? std::array<size_t, 3>{slots, slots, slots}
            : std::array<size_t, 3>{block_layout::array_cols, rows / block_layout::tile_m, block_layout::array_cols};
        std::array<size_t, 3> covered{};
        size_t bd_write = 0;
        for (size_t i = 4; i < words.size(); i += stream_patch::op_len(words[i])) {
            if (words[i] == 1) bd_write = i;
            if (words[i] != 0x81) continue;
            if (i + 11 >= words.size() || !bd_write || words[bd_write + 2] + 4 != words[i + 6])
                fail("invalid projection DMA patch in " + path.string());
            const size_t arg = words[i + 8], offset = words[i + 10];
            const size_t length = static_cast<size_t>(words[bd_write + 4]) * 4;
            if (arg >= covered.size() || words[i + 11] || offset != covered[arg] ||
                length != spans[arg] / slices[arg] || length > spans[arg] - covered[arg])
                fail("projection DMA does not match the model dimensions in " + path.string());
            covered[arg] += length;
            if (arg == 0) fills.emplace_back(i + 10, static_cast<uint32_t>(offset));
        }
        if (covered != spans) fail("incomplete projection DMA coverage in " + path.string());
        count = words.size();
        instruction = xrt::bo(device, count * sizeof(uint32_t), xrt::bo::flags::cacheable, group);
        std::memcpy(instruction.map<void*>(), words.data(), count * sizeof(uint32_t));
    }
};

struct CompactFormat {
    xrt::hw_context context;
    xrt::kernel kernel;
    std::vector<CompactStream> streams;

    CompactFormat(xrt::device& device, const std::filesystem::path& root, const char* prefix, size_t rows,
                  bool iq4, bool grouped, const xrt::kernel* singleton = nullptr, bool shared_q8 = false) {
        const std::string name = std::string(prefix) + (iq4 ? "-iq4" : "-q8");
        if (singleton) {
            kernel = *singleton;
        } else {
            const xrt::xclbin image((root / (name + ".xclbin")).string());
            context = xrt::hw_context(device, device.register_xclbin(image));
            kernel = xrt::kernel(context, "MLIR_AIE");
        }
        const auto group = kernel.group_id(1);
        const bool split_scale = (singleton || shared_q8) && !iq4;
        streams.reserve(2);
        streams.emplace_back(device, group, root / (name + "-up.bin"), rows, width, iq4, grouped, width, split_scale);
        streams.emplace_back(device, group, root / (name + "-down.bin"), rows, expert_width, iq4, grouped, width, split_scale);
    }
};

struct SingletonStream {
    shared_dense::Shape shape;
    xrt::bo instruction;
    size_t count;

    SingletonStream(const xrt::device& device, uint32_t group, const std::filesystem::path& path, shared_dense::Shape role)
        : shape(role) {
        const auto words = read_stream(path);
        shared_dense::check_stream(words, {shape.weight_bytes(), shape.input_bytes(), shape.output_bytes()});
        count = words.size();
        instruction = xrt::bo(device, count * sizeof(uint32_t), xrt::bo::flags::cacheable, group);
        std::memcpy(instruction.map<void*>(), words.data(), count * sizeof(uint32_t));
        instruction.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
};

struct SingletonDecode {
    xrt::hw_context context;
    xrt::kernel kernel;
    std::vector<SingletonStream> dense;
    std::array<std::array<xrt::bo, 2>, layers> weights;
    std::vector<float> activation;

    SingletonDecode(xrt::device& device, const std::filesystem::path& root, const nlohmann::json& manifest,
                    const std::vector<size_t>& tiers) {
        nlohmann::json contract = {
            {"format", "inkling2-shared-split-q8"},
            {"image", "singleton-decode.xclbin"},
            {"experts", nlohmann::json::object()},
            {"attention", nlohmann::json::object()},
            {"dense", nlohmann::json::object()},
        };
        for (const std::string name : {"iq4-up", "iq4-down", "q8-up", "q8-down"})
            contract["experts"][name] = "singleton-" + name + ".bin";
        for (const auto tier : tiers)
            contract["attention"][std::to_string(tier)] = "singleton-attention-" + std::to_string(tier) + ".bin";
        for (const std::string name : {"fused", "output"})
            contract["dense"][name] = "singleton-" + name + ".bin";
        if (contract != manifest) fail("invalid Inkling 2 singleton decode manifest");

        const xrt::xclbin image((root / "singleton-decode.xclbin").string());
        context = xrt::hw_context(device, device.register_xclbin(image));
        kernel = xrt::kernel(context, "MLIR_AIE");
        const auto group = kernel.group_id(1);
        dense.reserve(2);
        dense.emplace_back(device, group, root / "singleton-fused.bin", shared_dense::fused);
        dense.emplace_back(device, group, root / "singleton-output.bin", shared_dense::output);
    }
};

struct Head {
    xrt::hw_context context;
    xrt::kernel kernel;
    xrt::bo instruction;
    size_t count = 0;

    Head(xrt::device& device, const std::filesystem::path& root, const std::string& name, size_t rows) {
        const auto words = read_stream(root / (name + ".bin"));
        const std::array<size_t, 3> bytes{head_bytes, rows * width * sizeof(uint16_t), rows * head_rows * sizeof(float)};
        std::array<size_t, 3> covered{};
        size_t bd_write = 0;
        for (size_t i = 4; i < words.size(); i += stream_patch::op_len(words[i])) {
            if (words[i] == 1) bd_write = i;
            if (words[i] != 0x81) continue;
            if (i + 11 >= words.size() || !bd_write || words[bd_write + 2] + 4 != words[i + 6])
                fail("invalid Q8 head DMA patch");
            const size_t arg = words[i + 8], offset = words[i + 10];
            const size_t length = static_cast<size_t>(words[bd_write + 4]) * 4;
            if (arg >= covered.size() || words[i + 11] || offset != covered[arg] ||
                length == 0 || length > bytes[arg] - covered[arg])
                fail("Q8 head DMA does not match the model dimensions");
            covered[arg] += length;
        }
        if (covered != bytes) fail("incomplete Q8 head DMA coverage");

        const xrt::xclbin image((root / (name + ".xclbin")).string());
        context = xrt::hw_context(device, device.register_xclbin(image));
        kernel = xrt::kernel(context, "MLIR_AIE");
        count = words.size();
        instruction = xrt::bo(device, count * sizeof(uint32_t), xrt::bo::flags::cacheable, kernel.group_id(1));
        std::memcpy(instruction.map<void*>(), words.data(), count * sizeof(uint32_t));
        instruction.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
};

namespace attention_layout {
constexpr size_t query_rows = 16, key_rows = 32, dim = 128, relative_dim = 16, strip = 64, extent = 1024;
constexpr size_t kv_heads = 8, group = 3, heads = kv_heads * group;
constexpr size_t q_width = heads * dim, kv_width = kv_heads * dim;
constexpr size_t query_words = (dim + relative_dim) * query_rows;
constexpr size_t query_bytes = 128 + 2 * query_words * sizeof(uint16_t);
constexpr size_t matrix_words = key_rows * dim, strip_words = relative_dim * strip;
constexpr size_t key_words = 2 * (matrix_words + strip_words);
constexpr size_t result_values = query_rows * (dim + 2);
constexpr size_t shared_rows = 4, shared_frame = 4 * block_layout::tile_n;
constexpr size_t shared_frames = (result_values + shared_frame - 1) / shared_frame;
constexpr size_t shared_output_column = shared_rows * shared_frame * shared_frames;

size_t shared_query_stride(bool iq4) {
    const size_t frame = block_layout::tile_n * (iq4 ? 18 : 34);
    return (query_bytes + frame - 1) / frame * frame;
}

using QueryHeader = std::array<int32_t, 8>;

uint8_t* query_record(uint8_t* dst, size_t head, size_t stride, size_t rows_per_column) {
    return dst + (head / group * rows_per_column + head % group) * stride;
}

void pack_query(const Inkling2AttentionView& view, size_t token, size_t rows, const QueryHeader& header,
                uint8_t* dst, size_t stride, size_t rows_per_column) {
    constexpr float scale = 0.08838834764831844f; // 1/sqrt(128)
    std::memset(dst, 0, kv_heads * rows_per_column * stride);
    for (size_t head = 0; head < heads; ++head) {
        auto* record = query_record(dst, head, stride, rows_per_column);
        std::memcpy(record, header.data(), sizeof(header));
        auto* bias = reinterpret_cast<float*>(record + 64);
        auto* hi = reinterpret_cast<uint16_t*>(record + 128);
        auto* lo = hi + query_words;
        for (size_t row = 0; row < rows; ++row) {
            const size_t at = (token + row) * heads + head;
            bias[row] = view.sinks[at];
            for (size_t d = 0; d < dim + relative_dim; ++d) {
                const float x = d < dim ? view.query[at * dim + d] * scale
                                        : view.relative[at * relative_dim + d - dim];
                const size_t lane = d * query_rows + row;
                hi[lane] = f32_to_bf16(x);
                lo[lane] = f32_to_bf16(x - bf16_to_f32(hi[lane]));
            }
        }
    }
}

template <bool Shared>
void merge_attention_tile(const float* partial, size_t token, size_t rows, float* out, float* peaks, float* masses) {
    static_assert(shared_frame % dim == 0);
    constexpr size_t state_offset = query_rows * dim;
    static_assert(state_offset % shared_frame + query_rows * 2 <= shared_frame);
    for (size_t head = 0; head < heads; ++head) {
        const auto* head_data = Shared
            ? partial + head / group * shared_output_column + head % group * shared_frame
            : partial + head * result_values;
        // Shared frames interleave the four worker rows in a column.
        const auto result_at = [head_data](size_t offset) {
            if constexpr (Shared)
                offset += (offset / shared_frame) * shared_frame * (shared_rows - 1);
            return head_data + offset;
        };
        const auto* state = result_at(state_offset);
        for (size_t row = 0; row < rows; ++row) {
            const float peak = state[row], mass = state[query_rows + row];
            if (mass <= 0) continue;
            const size_t lane = (token + row) * heads + head;
            const float merged = masses[lane] ? std::max(peaks[lane], peak) : peak;
            const float old_scale = masses[lane] ? std::exp(peaks[lane] - merged) : 0.0f;
            const float new_scale = std::exp(peak - merged);
            auto* dest = out + lane * dim;
            const auto* numerator = result_at(row * dim);
            for (size_t d = 0; d < dim; ++d)
                dest[d] = old_scale * dest[d] + new_scale * numerator[d];
            peaks[lane] = merged;
            masses[lane] = old_scale * masses[lane] + new_scale * mass;
        }
    }
}

#if defined(__AVX2__) && defined(_OPENMP)
bool stream_old_kv(const Inkling2AttentionView& view, size_t query, size_t first,
                   size_t end, size_t chunks, uint16_t* dst, const inkling_wide_bfp::KeyCache* archive) {
    constexpr size_t min_chunks = 128;
    if (chunks < min_chunks || view.window || view.ring || end <= first || end > view.position ||
        end - first != chunks * key_rows || query < end - 1 || query - (end - 1) < extent ||
        reinterpret_cast<std::uintptr_t>(dst) % 64 != 0)
        return false;

    // Complete, cached old ranges leave the relative-position stripe untouched.
#pragma omp parallel
    {
#pragma omp for schedule(static) nowait
        for (long long chunk = 0; chunk < static_cast<long long>(chunks); ++chunk) {
            const size_t at = first + static_cast<size_t>(chunk) * key_rows;
            for (size_t head = 0; head < kv_heads; ++head) {
                auto* out = dst + (head * chunks + chunk) * key_words;
                const size_t source = at * kv_width + head * dim;
                const bool restored = archive && archive->restore_head_frame(view.visit, head, at, out);
                const auto* keys = view.cached_key + source;
                const auto* values = view.cached_value + source;
                for (size_t row = 0; row < key_rows; ++row) {
                    for (size_t d = 0; d < dim; d += 16) {
                        auto* to = out + row * dim + d;
                        const size_t offset = row * kv_width + d;
                        if (!restored) _mm256_stream_si256(reinterpret_cast<__m256i*>(to),
                            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(keys + offset)));
                        _mm256_stream_si256(reinterpret_cast<__m256i*>(to + matrix_words),
                            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values + offset)));
                    }
                }
            }
        }
        // Every writer must finish its stores before the caller syncs the BO.
        _mm_sfence();
    }
    return true;
}
#endif

void pack_kv(const Inkling2AttentionView& view, const std::vector<uint16_t>& table, size_t query, size_t first,
             size_t end, size_t chunks, uint16_t* dst, const inkling_wide_bfp::KeyCache* archive) {
#if defined(__AVX2__) && defined(_OPENMP)
    if (stream_old_kv(view, query, first, end, chunks, dst, archive)) return;
#endif
    for (size_t at = first; at < end; at += key_rows) {
        const size_t chunk = (at - first) / key_rows;
        const size_t real = std::min(end - at, key_rows);
        const int distance = static_cast<int>(query) - static_cast<int>(at + key_rows - 1);
        const bool relative = distance < static_cast<int>(extent);
        std::array<uint16_t, 2 * strip_words> bias{};
        if (relative) {
            constexpr size_t table_words = extent * relative_dim;
            const int lo = std::max(distance, 0);
            const int hi = std::min(static_cast<int>(extent), distance + static_cast<int>(key_rows + query_rows - 1));
            for (int i = lo; i < hi; ++i) {
                for (size_t d = 0; d < relative_dim; ++d) {
                    const size_t to = d * strip + static_cast<size_t>(i - distance);
                    bias[to] = table[i * relative_dim + d];
                    bias[strip_words + to] = table[table_words + i * relative_dim + d];
                }
            }
        }
        for (size_t head = 0; head < kv_heads; ++head) {
            auto* block = dst + (head * chunks + chunk) * key_words;
            const bool restored = archive && archive->restore_head_frame(view.visit, head, at, block);
            for (size_t key = at; key < at + real; ++key) {
                const auto* k = view.pending_key;
                const auto* v = view.pending_value;
                size_t index;
                if (key < view.position) {
                    k = view.cached_key;
                    v = view.cached_value;
                    index = view.ring ? key % view.ring : key;
                } else {
                    index = key - view.position;
                }
                index = index * kv_width + head * dim;
                const size_t local = (key - at) * dim;
                if (!restored) std::copy_n(k + index, dim, block + local);
                std::copy_n(v + index, dim, block + matrix_words + local);
            }
            std::fill(block + real * dim, block + matrix_words, uint16_t{0});
            std::fill(block + matrix_words + real * dim, block + 2 * matrix_words, uint16_t{0});
            if (relative) std::copy(bias.begin(), bias.end(), block + 2 * matrix_words);
        }
    }
}

// Decode and verify records (at most 8 rows): K transposed, V, then FP32 bias [row][head in group][key]
// with masked keys at -1e30. The kernel needs no relative strip in this mode.
void pack_kv_rows(const Inkling2AttentionView& view, size_t query, size_t rows, size_t first, size_t end,
                  size_t chunks, uint16_t* dst) {
    constexpr float hidden = -1.0e30f;
    const size_t row0 = query - view.position;
#pragma omp parallel for schedule(static)
    for (long long item = 0; item < static_cast<long long>(chunks * kv_heads); ++item) {
        const size_t chunk = static_cast<size_t>(item) / kv_heads, head = static_cast<size_t>(item) % kv_heads;
        const size_t at = first + chunk * key_rows;
        auto* block = dst + (head * chunks + chunk) * key_words;
        auto* keys_t = block;
        auto* values = block + matrix_words;
        auto* bias = reinterpret_cast<float*>(block + 2 * matrix_words);
        std::array<const uint16_t*, key_rows> key_rows_src{};
        for (size_t j = 0; j < key_rows; ++j) {
            const size_t key = at + j;
            if (key >= end) {
                std::fill_n(values + j * dim, dim, uint16_t{0});
                continue;
            }
            const uint16_t* k = view.pending_key;
            const uint16_t* v = view.pending_value;
            size_t index;
            if (key < view.position) {
                k = view.cached_key;
                v = view.cached_value;
                index = view.ring ? key % view.ring : key;
            } else {
                index = key - view.position;
            }
            index = index * kv_width + head * dim;
            key_rows_src[j] = k + index;
            std::copy_n(v + index, dim, values + j * dim);
        }
        static const std::array<uint16_t, dim> zero_row{};
        for (auto& row : key_rows_src)
            if (!row) row = zero_row.data();
#if defined(__AVX2__) || defined(_M_X64) || defined(__x86_64__)
        // 8x8 blocks of 16-bit words through three unpack rounds.
        for (size_t j0 = 0; j0 < key_rows; j0 += 8)
            for (size_t d0 = 0; d0 < dim; d0 += 8) {
                __m128i r[8];
                for (size_t i = 0; i < 8; ++i)
                    r[i] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(key_rows_src[j0 + i] + d0));
                const __m128i a0 = _mm_unpacklo_epi16(r[0], r[1]), a1 = _mm_unpackhi_epi16(r[0], r[1]);
                const __m128i a2 = _mm_unpacklo_epi16(r[2], r[3]), a3 = _mm_unpackhi_epi16(r[2], r[3]);
                const __m128i a4 = _mm_unpacklo_epi16(r[4], r[5]), a5 = _mm_unpackhi_epi16(r[4], r[5]);
                const __m128i a6 = _mm_unpacklo_epi16(r[6], r[7]), a7 = _mm_unpackhi_epi16(r[6], r[7]);
                const __m128i b0 = _mm_unpacklo_epi32(a0, a2), b1 = _mm_unpackhi_epi32(a0, a2);
                const __m128i b2 = _mm_unpacklo_epi32(a1, a3), b3 = _mm_unpackhi_epi32(a1, a3);
                const __m128i b4 = _mm_unpacklo_epi32(a4, a6), b5 = _mm_unpackhi_epi32(a4, a6);
                const __m128i b6 = _mm_unpacklo_epi32(a5, a7), b7 = _mm_unpackhi_epi32(a5, a7);
                const __m128i c[8] = {_mm_unpacklo_epi64(b0, b4), _mm_unpackhi_epi64(b0, b4),
                                      _mm_unpacklo_epi64(b1, b5), _mm_unpackhi_epi64(b1, b5),
                                      _mm_unpacklo_epi64(b2, b6), _mm_unpackhi_epi64(b2, b6),
                                      _mm_unpacklo_epi64(b3, b7), _mm_unpackhi_epi64(b3, b7)};
                for (size_t i = 0; i < 8; ++i)
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(keys_t + (d0 + i) * key_rows + j0), c[i]);
            }
#else
        for (size_t d = 0; d < dim; ++d)
            for (size_t j = 0; j < key_rows; ++j) keys_t[d * key_rows + j] = key_rows_src[j][d];
#endif
        for (size_t r = 0; r < rows; ++r) {
            const size_t q = query + r;
            for (size_t g = 0; g < group; ++g) {
                const size_t h = head * group + g;
                const float* rel = view.relative + ((row0 + r) * heads + h) * relative_dim;
                float* out = bias + (r * group + g) * key_rows;
                for (size_t j = 0; j < key_rows; ++j) {
                    const size_t key = at + j;
                    const bool visible = key < end && key <= q && (!view.window || key + view.window > q);
                    float value = hidden;
                    if (visible) {
                        value = 0.0f;
                        if (q - key < extent) {
                            const float* t = view.table + (q - key) * relative_dim;
                            for (size_t d = 0; d < relative_dim; ++d) value += rel[d] * t[d];
                        }
                    }
                    out[j] = value;
                }
            }
        }
    }
}

struct Tier {
    size_t chunks, count;
    xrt::bo insts;

    Tier(xrt::device& device, uint32_t dma_group, const std::filesystem::path& path, size_t capacity,
         size_t query_column, size_t output_column)
        : chunks(capacity) {
        const auto words = read_stream(path);
        const std::array<size_t, 3> per_column{query_column, chunks * key_words * sizeof(uint16_t),
                                               output_column * sizeof(float)};
        std::array<size_t, 3> covered{};
        size_t bd = 0;
        for (size_t i = 4; i < words.size(); i += stream_patch::op_len(words[i])) {
            if (words[i] == 1) bd = i;
            if (words[i] != 0x81) continue;
            if (i + 11 >= words.size() || !bd || words[bd + 2] + 4 != words[i + 6])
                fail("invalid relative attention DMA patch");
            const size_t arg = words[i + 8];
            if (arg >= covered.size() || words[i + 11] || words[i + 10] != covered[arg] * per_column[arg] ||
                static_cast<size_t>(words[bd + 4]) * 4 != per_column[arg])
                fail("relative attention DMA does not match the tier capacity");
            ++covered[arg];
        }
        if (covered != std::array<size_t, 3>{kv_heads, kv_heads, kv_heads}) fail("incomplete attention stream");
        count = words.size();
        insts = xrt::bo(device, count * sizeof(uint32_t), xrt::bo::flags::cacheable, dma_group);
        std::memcpy(insts.map<void*>(), words.data(), count * sizeof(uint32_t));
        insts.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
};

struct Attention {
    struct Mode {
        xrt::kernel kernel;
        std::vector<Tier> tiers;
        size_t stride, rows_per_column, output_column;
    };
    using DecodeFormats = std::array<std::unique_ptr<CompactFormat>, 2>;

    bool shared, singleton;
    xrt::hw_context context;
    std::array<Mode, 2> modes;
    xrt::bo query, kv, result;
    std::vector<uint8_t> packed_queries;
    std::array<std::vector<uint16_t>, layers> table;

    Attention(xrt::device& device, const std::filesystem::path& root, const std::vector<size_t>& chunks,
              const DecodeFormats* decode, bool unified) : shared(decode != nullptr), singleton(unified) {
        if (!shared) {
            const xrt::xclbin image((root / "relative-attention.xclbin").string());
            context = xrt::hw_context(device, device.register_xclbin(image));
            modes[0].kernel = xrt::kernel(context, "MLIR_AIE");
        }
        size_t q_bytes = 0, out_values = 0;
        for (size_t i = 0; i < (shared && !singleton ? modes.size() : 1); ++i) {
            auto& mode = modes[i];
            std::string prefix;
            if (shared) {
                mode.kernel = (*decode)[i]->kernel;
                prefix = singleton ? "singleton-" : i == 0 ? "shared-iq4-" : "shared-q8-";
            }
            constexpr size_t singleton_frame = shared_dense::tile * 36;
            constexpr size_t singleton_stride = (query_bytes + singleton_frame - 1) / singleton_frame * singleton_frame;
            mode.stride = singleton ? singleton_stride : shared ? shared_query_stride(i == 0) : query_bytes;
            mode.rows_per_column = shared ? shared_rows : group;
            mode.output_column = shared ? shared_output_column : group * result_values;
            const size_t query_column = mode.stride * mode.rows_per_column;
            for (size_t n : chunks) {
                const auto stream = prefix + "attention-" + std::to_string(n) + ".bin";
                mode.tiers.emplace_back(device, mode.kernel.group_id(1), root / stream, n, query_column, mode.output_column);
            }
            q_bytes = std::max(q_bytes, kv_heads * query_column);
            out_values = std::max(out_values, kv_heads * mode.output_column);
        }
        query = xrt::ext::bo(device, padded_bo(q_bytes));
        kv = xrt::ext::bo(device, padded_bo(kv_heads * chunks.back() * key_words * sizeof(uint16_t)));
        result = xrt::ext::bo(device, padded_bo(out_values * sizeof(float)));
    }

    const std::vector<uint16_t>& relative_table(const Inkling2AttentionView& view) {
        auto& packed = table.at(view.layer);
        if (packed.empty()) {
            constexpr size_t size = extent * relative_dim;
            packed.resize(2 * size);
            for (size_t i = 0; i < size; ++i) {
                packed[i] = f32_to_bf16(view.table[i]);
                packed[size + i] = f32_to_bf16(view.table[i] - bf16_to_f32(packed[i]));
            }
        }
        return packed;
    }
};
struct WideOldAttention {
    static constexpr size_t rows = 256, core_rows = 32, cores = rows / core_rows;
    static constexpr size_t frame_bytes = inkling_wide_bfp::frame_bytes;
    static constexpr size_t record = (1 + 2 * cores) * frame_bytes;
    static constexpr size_t output_tile = (core_rows + 1) * dim;
    static constexpr size_t output_values = heads * cores * output_tile;
    static constexpr size_t short_keys = 4096, long_keys = 16384;
    static_assert(core_rows == inkling_wide_bfp::tile && dim == inkling_wide_bfp::width &&
                  kv_heads == inkling_wide_bfp::heads && short_keys == inkling_wide_bfp::KeyCache::segment);

    struct Stream {
        size_t count = 0;
        xrt::bo insts;
    };

    xrt::hw_context context;
    xrt::kernel kernel;
    std::array<Stream, 2> streams;
    xrt::bo query, key_value, output;
    inkling_wide_bfp::KeyCache packed_keys;

    WideOldAttention(xrt::device& device, const std::filesystem::path& root) {
        const xrt::xclbin image((root / "wide-old-attention.xclbin").string());
        context = xrt::hw_context(device, device.register_xclbin(image));
        kernel = xrt::kernel(context, "MLIR_AIE");
        const auto index = kernel.group_id(1);
        for (size_t i = 0; i < streams.size(); ++i) {
            const auto words = read_stream(root / (i ? "wide-old-attention-16384.bin" : "wide-old-attention-4096.bin"));
            auto& stream = streams[i];
            stream.count = words.size();
            const size_t bytes = stream.count * sizeof(uint32_t);
            stream.insts = xrt::bo(device, bytes, xrt::bo::flags::cacheable, index);
            std::memcpy(stream.insts.map<void*>(), words.data(), bytes);
            stream.insts.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        query = xrt::ext::bo(device, padded_bo(heads * record));
        key_value = xrt::ext::bo(device, padded_bo(2 * kv_heads * (long_keys / core_rows) * frame_bytes));
        output = xrt::ext::bo(device, padded_bo(output_values * sizeof(float)));
    }

    void upload_query(const Inkling2AttentionView& view) {
        auto* base = query.map<uint8_t*>();
        constexpr float scale = 0.08838834764831844f, log2e = 1.4426950408889634f;
        const int32_t last = long_keys - 1;
#pragma omp parallel for schedule(static)
        for (int h = 0; h < static_cast<int>(heads); ++h) {
            auto* dest = base + h * record;
            std::memset(dest, 0, record);
            std::memcpy(dest, &last, sizeof(last));
            for (size_t worker = 0; worker < cores; ++worker) {
                auto* hi = dest + (1 + worker) * frame_bytes;
                auto* lo = hi + cores * frame_bytes;
                for (size_t row = 0; row < core_rows; row += 8) {
                    for (size_t col = 0; col < dim; col += 8) {
                        for (size_t lane = 0; lane < 8; ++lane) {
                            const size_t offset = worker * core_rows + row + lane;
                            if (offset >= view.rows) {
                                hi += inkling_wide_bfp::bytes;
                                lo += inkling_wide_bfp::bytes;
                                continue;
                            }
                            const auto* input = view.query + (offset * heads + h) * dim + col;
                            alignas(32) float values[8];
                            for (size_t d = 0; d < 8; ++d) {
                                const float value = (input[d] * scale) * log2e;
                                const float rounded = bf16_to_f32(f32_to_bf16(value));
                                values[d] = rounded + bf16_to_f32(f32_to_bf16(value - rounded));
                            }
                            inkling_wide_bfp::encode_values(values, hi, lo);
                            hi += inkling_wide_bfp::bytes;
                            lo += inkling_wide_bfp::bytes;
                        }
                    }
                }
            }
        }
        query.sync(XCL_BO_SYNC_BO_TO_DEVICE, heads * record, 0);
    }

    void upload_cache(const Inkling2AttentionView& view, size_t first, size_t length) {
        auto* dst = key_value.map<uint8_t*>();
        const size_t chunks = length / core_rows;
        const size_t plane_bytes = chunks * frame_bytes;
#pragma omp parallel for schedule(static)
        for (int plane_head = 0; plane_head < static_cast<int>(2 * kv_heads); ++plane_head) {
            const size_t head = plane_head % kv_heads;
            auto* out = dst + plane_head * plane_bytes;
            if (plane_head < static_cast<int>(kv_heads)) {
                packed_keys.copy_head(view.sequence, view.visit, head, first, length, out);
                continue;
            }
            const auto* src = view.cached_value + first * kv_width + head * dim;
            for (size_t chunk = 0; chunk < chunks; ++chunk) {
                auto* frame = out + chunk * frame_bytes;
                for (size_t row = 0; row < core_rows; ++row) {
                    constexpr size_t row_bytes = dim * sizeof(uint16_t);
                    std::memcpy(frame + row * row_bytes, src + (chunk * core_rows + row) * kv_width, row_bytes);
                }
                constexpr size_t payload = core_rows * dim * sizeof(uint16_t);
                std::memset(frame + payload, 0, frame_bytes - payload);
            }
        }
        key_value.sync(XCL_BO_SYNC_BO_TO_DEVICE, 2 * kv_heads * plane_bytes, 0);
    }

    static void merge(const float* result, float* out, float* peaks, float* masses, size_t active_rows) {
        constexpr float ln2 = 0.6931471805599453f;
#pragma omp parallel for schedule(static)
        for (int h = 0; h < static_cast<int>(heads); ++h) {
            for (size_t worker = 0; worker < cores; ++worker) {
                const auto* tile = result + (h * cores + worker) * output_tile;
                const auto* state = tile + core_rows * dim;
                for (size_t row = 0; row < core_rows && worker * core_rows + row < active_rows; ++row) {
                    const size_t lane = (worker * core_rows + row) * heads + h;
                    const float peak = state[row] * ln2, mass = state[core_rows + row];
                    const float merged = std::max(peaks[lane], peak);
                    const float old_scale = std::exp(peaks[lane] - merged);
                    const float new_scale = std::exp(peak - merged);
                    const auto* value = tile + row * dim;
                    auto* dest = out + lane * dim;
                    for (size_t d = 0; d < dim; ++d)
                        dest[d] = old_scale * dest[d] + new_scale * value[d];
                    peaks[lane] = merged;
                    masses[lane] = old_scale * masses[lane] + new_scale * mass;
                }
            }
        }
    }
};

// One dispatch covers 256 query rows, all heads and 1,280 keys (inkling2/near_attention.py).
struct NearAttention {
    static constexpr size_t rows = 256, core_rows = 32, cores = rows / core_rows, chunk = 16, strip = 48;
    static constexpr size_t key_bytes = chunk * dim * sizeof(uint16_t);
    static constexpr size_t strip_words = relative_dim * strip, part_words = 4 * strip_words;
    static constexpr size_t frame_bytes = key_bytes + part_words * sizeof(uint16_t);
    static constexpr size_t q_words = (dim + relative_dim) * core_rows;
    static constexpr size_t out_words = core_rows * dim + 2 * core_rows;
    static constexpr size_t query_bytes = heads * 2 * cores * frame_bytes;
    static constexpr size_t output_values = heads * cores * out_words;
    static_assert(q_words * sizeof(uint16_t) + 64 <= frame_bytes);

    size_t chunks, span;
    xrt::hw_context context;
    xrt::kernel kernel;
    xrt::bo insts, query, kv, output;
    size_t count = 0;
    std::array<xrt::bo, layers> strips;

    NearAttention(xrt::device& device, const std::filesystem::path& root, size_t chunk_count)
        : chunks(chunk_count), span(chunk_count * chunk) {
        const xrt::xclbin image((root / "near-attention.xclbin").string());
        context = xrt::hw_context(device, device.register_xclbin(image));
        kernel = xrt::kernel(context, "MLIR_AIE");
        const auto words = read_stream(root / "near-attention.bin");
        count = words.size();
        insts = xrt::bo(device, count * sizeof(uint32_t), xrt::bo::flags::cacheable, kernel.group_id(1));
        std::memcpy(insts.map<void*>(), words.data(), count * sizeof(uint32_t));
        insts.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        query = xrt::ext::bo(device, padded_bo(query_bytes));
        kv = xrt::ext::bo(device, padded_bo(2 * kv_heads * span * dim * sizeof(uint16_t)));
        output = xrt::ext::bo(device, padded_bo(output_values * sizeof(float)));
    }

    // Strip (chunk c, core i) holds table rows for distances 1024 + 32i - 16c - 15 + x, x < 47.
    // That geometry assumes the near segment starts 1,024 keys before the block.
    const xrt::bo& layer_strips(xrt::device& device, const Inkling2AttentionView& view) {
        auto& bo = strips.at(view.layer);
        if (bo) return bo;
        const size_t words = 2 * chunks * part_words;
        bo = xrt::ext::bo(device, padded_bo(words * sizeof(uint16_t)));
        auto* dst = bo.map<uint16_t*>();
        std::fill_n(dst, words, uint16_t{0});
        for (size_t c = 0; c < chunks; ++c)
            for (size_t i = 0; i < cores; ++i) {
                auto* out = dst + ((i / 4) * chunks + c) * part_words + (i % 4) * strip_words;
                const long delta = static_cast<long>(extent + core_rows * i) - static_cast<long>(chunk * c);
                for (size_t x = 0; x + 1 < strip; ++x) {
                    const long distance = delta - static_cast<long>(chunk - 1) + static_cast<long>(x);
                    if (distance < 0 || distance >= static_cast<long>(extent)) continue;
                    for (size_t d = 0; d < relative_dim; ++d)
                        out[d * strip + x] = f32_to_bf16(view.table[distance * relative_dim + d]);
                }
            }
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, words * sizeof(uint16_t), 0);
        return bo;
    }

    void pack_query(const Inkling2AttentionView& view) {
        constexpr float scale = 0.08838834764831844f * 1.4426950408889634f, log2e = 1.4426950408889634f;
        auto* base = query.map<uint8_t*>();
#pragma omp parallel for schedule(static)
        for (int h = 0; h < static_cast<int>(heads); ++h) {
            for (size_t w = 0; w < cores; ++w) {
                auto* hi = reinterpret_cast<uint16_t*>(base + ((h * 2 + 0) * cores + w) * frame_bytes);
                auto* lo = reinterpret_cast<uint16_t*>(base + ((h * 2 + 1) * cores + w) * frame_bytes);
                std::fill_n(hi, q_words, uint16_t{0});
                std::fill_n(lo, q_words, uint16_t{0});
                for (size_t r = 0; r < core_rows && w * core_rows + r < view.rows; ++r) {
                    const size_t at = (w * core_rows + r) * heads + h;
                    for (size_t d = 0; d < dim + relative_dim; ++d) {
                        const float x = d < dim ? view.query[at * dim + d] * scale
                                                : view.relative[at * relative_dim + d - dim] * log2e;
                        const uint16_t high = f32_to_bf16(x);
                        hi[d * core_rows + r] = high;
                        lo[d * core_rows + r] = f32_to_bf16(x - bf16_to_f32(high));
                    }
                }
            }
        }
    }

    // Meta: first query position, key position of chunk 0, window, valid rows, lowest visible key.
    void set_meta(const Inkling2AttentionView& view, long kbase, long low) {
        const std::array<int32_t, 16> meta{static_cast<int32_t>(view.position), static_cast<int32_t>(kbase),
                                           static_cast<int32_t>(view.window), static_cast<int32_t>(view.rows),
                                           static_cast<int32_t>(low)};
        auto* base = query.map<uint8_t*>();
        for (size_t h = 0; h < heads; ++h)
            for (size_t w = 0; w < cores; ++w)
                std::memcpy(base + ((h * 2) * cores + w) * frame_bytes + q_words * sizeof(uint16_t), meta.data(),
                            sizeof(meta));
        query.sync(XCL_BO_SYNC_BO_TO_DEVICE, query_bytes, 0);
    }

    // K then V planes of [kv head][span keys][128]. Keys outside [low, position + rows) are zero.
    void pack_kv(const Inkling2AttentionView& view, long kbase, long low) {
        auto* dst = kv.map<uint16_t*>();
        const long end = static_cast<long>(view.position + view.rows);
#pragma omp parallel for schedule(static)
        for (int plane_head = 0; plane_head < static_cast<int>(2 * kv_heads); ++plane_head) {
            const bool values = plane_head >= static_cast<int>(kv_heads);
            const size_t head = plane_head % kv_heads;
            auto* out = dst + plane_head * span * dim;
            const uint16_t* cached = values ? view.cached_value : view.cached_key;
            const uint16_t* pending = values ? view.pending_value : view.pending_key;
            for (size_t j = 0; j < span; ++j) {
                const long key = kbase + static_cast<long>(j);
                auto* row = out + j * dim;
                if (key < low || key >= end) {
                    std::fill_n(row, dim, uint16_t{0});
                    continue;
                }
                const size_t p = static_cast<size_t>(key);
                const uint16_t* src = p < view.position
                    ? cached + (view.ring ? p % view.ring : p) * kv_width
                    : pending + (p - view.position) * kv_width;
                std::memcpy(row, src + head * dim, dim * sizeof(uint16_t));
            }
        }
        kv.sync(XCL_BO_SYNC_BO_TO_DEVICE, 2 * kv_heads * span * dim * sizeof(uint16_t), 0);
    }

    static void merge(const float* result, float* out, float* peaks, float* masses, size_t active_rows) {
        constexpr float ln2 = 0.6931471805599453f;
#pragma omp parallel for schedule(static)
        for (int h = 0; h < static_cast<int>(heads); ++h) {
            for (size_t w = 0; w < cores; ++w) {
                const auto* tile = result + (h * cores + w) * out_words;
                const auto* state = tile + core_rows * dim;
                for (size_t r = 0; r < core_rows && w * core_rows + r < active_rows; ++r) {
                    const float mass = state[core_rows + r];
                    if (!(mass > 0)) continue;
                    const size_t lane = (w * core_rows + r) * heads + h;
                    const float peak = state[r] * ln2;
                    const float merged = std::max(peaks[lane], peak);
                    const float old_scale = std::exp(peaks[lane] - merged);
                    const float new_scale = std::exp(peak - merged);
                    auto* dest = out + lane * dim;
                    for (size_t d = 0; d < dim; ++d)
                        dest[d] = old_scale * dest[d] + new_scale * tile[d * core_rows + r];
                    peaks[lane] = merged;
                    masses[lane] = old_scale * masses[lane] + new_scale * mass;
                }
            }
        }
    }
};

} // namespace attention_layout

namespace dense_layout {
constexpr size_t rows = 64;
struct Shape { const char* name; size_t inputs, outputs; };
constexpr std::array<Shape, 4> shapes{{
    {"dense-fused", width, 9216},
    {"dense-query", width, 6912},
    {"dense-kv", width, 1536},
    {"dense-output", attention_layout::q_width, width},
}};
enum Role { Fused, Query, Key, Value, Output, Count };
constexpr std::array<size_t, Count> stream_index{0, 1, 2, 2, 3};

struct Part {
    const char* tensor;
    size_t outputs;
    float* dst;
};

struct Dense {
    xrt::hw_context context;
    xrt::kernel kernel;
    std::vector<CompactStream> streams;
    std::array<std::array<xrt::bo, Count>, layers> weights;
    std::vector<float> unpacked = std::vector<float>(rows * shapes.front().outputs);

    Dense(xrt::device& device, const std::filesystem::path& root) {
        const xrt::xclbin image((root / "dense-q8.xclbin").string());
        context = xrt::hw_context(device, device.register_xclbin(image));
        kernel = xrt::kernel(context, "MLIR_AIE");
        streams.reserve(shapes.size());
        for (const auto& shape : shapes) {
            auto& stream = streams.emplace_back(device, kernel.group_id(1),
                root / (std::string(shape.name) + ".bin"), rows, shape.inputs, false, false, shape.outputs, true);
            stream.instruction.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
    }
};
} // namespace dense_layout

// Operand layouts of designs/inkling2/loop_controller.py; recipes/inkling2_controller.py packs the same.
namespace controller_layout {
constexpr size_t rows = 256, rank = 64, chunk = 32, chunks = width / chunk, object = 4096;
constexpr size_t values = 2 * chunks * object, bytes = values * sizeof(uint16_t);
constexpr size_t activation_floats = rows * 2 * width, output_floats = rows * width;
constexpr size_t first_layer = 6, core_layers = 12;
constexpr std::array<const char*, 4> kinds{"q", "k", "v", "ffn"};

struct Program {
    xrt::bo instruction;
    size_t count = 0;

    Program(const xrt::device& device, uint32_t group, const std::filesystem::path& path, size_t controllers) {
        const auto words = read_stream(path);
        const std::array<size_t, 3> spans{4 * bytes, controllers * activation_floats * sizeof(float),
                                         controllers * output_floats * sizeof(float)};
        std::array<size_t, 3> patches{};
        for (size_t i = 4; i < words.size(); i += stream_patch::op_len(words[i])) {
            if (words[i] != 0x81) continue;
            if (i + 11 >= words.size()) fail("truncated DMA patch in " + path.string());
            const size_t arg = words[i + 8], offset = words[i + 10];
            if (arg >= spans.size() || words[i + 11] || offset >= spans[arg])
                fail("controller DMA outside its buffer in " + path.string());
            ++patches[arg];
        }
        // Four weight shims, and one activation fill and one drain for each column and controller.
        if (patches != std::array<size_t, 3>{4, 8 * controllers, 8 * controllers})
            fail("controller DMA does not match the stream in " + path.string());
        count = words.size();
        instruction = xrt::bo(device, count * sizeof(uint32_t), xrt::bo::flags::cacheable, group);
        std::memcpy(instruction.map<void*>(), words.data(), count * sizeof(uint32_t));
        instruction.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
};

uint16_t exact_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if (bits & 0xffffu) fail("loop controller weights are not bf16 values");
    return static_cast<uint16_t>(bits >> 16);
}

// One layer's four controllers in stream order: phase 1 [frame][k tile][z tile][8 k][8 z], then
// phase 2 [chunk][gain, forget][z tile][h tile][8 z][8 h].
void pack(const GgufFile& file, size_t layer, uint16_t* out) {
    const std::string prefix = "blk." + std::to_string(layer) + ".loop_";
    for (size_t c = 0; c < kinds.size(); ++c, out += values) {
        const auto weights = [&](const char* part, size_t k, size_t n) {
            const auto name = prefix + kinds[c] + "_ctrl_" + part + ".weight";
            const auto& tensor = file.tensor(name);
            size_t size = 0;
            const auto* raw = file.raw(name, &size);
            if (tensor.type != GgufFile::Type::F32 || tensor.dims != std::vector<uint64_t>{k, n} ||
                size != k * n * sizeof(float))
                fail("unsupported loop controller tensor " + name);
            return reinterpret_cast<const float*>(raw);
        };
        const float* down = weights("down", 2 * width, rank);
        const std::array<const float*, 2> gates{weights("gain", rank, width), weights("forget", rank, width)};
        size_t at = 0;
        for (size_t j = 0; j < chunks; ++j)
            for (size_t kt = 0; kt < 8; ++kt)
                for (size_t zt = 0; zt < 8; ++zt)
                    for (size_t kl = 0; kl < 8; ++kl)
                        for (size_t zl = 0; zl < 8; ++zl) {
                            const size_t k = (kt < 4 ? 0 : width) + j * chunk + (kt % 4) * 8 + kl;
                            out[at++] = exact_bf16(down[(zt * 8 + zl) * 2 * width + k]);
                        }
        for (size_t j = 0; j < chunks; ++j)
            for (const float* gate : gates)
                for (size_t zt = 0; zt < 8; ++zt)
                    for (size_t ht = 0; ht < 4; ++ht)
                        for (size_t zl = 0; zl < 8; ++zl)
                            for (size_t hl = 0; hl < 8; ++hl)
                                out[at++] = exact_bf16(gate[(j * chunk + ht * 8 + hl) * rank + zt * 8 + zl]);
    }
}

struct Controller {
    xrt::hw_context context;
    xrt::kernel kernel;
    std::vector<Program> programs;  // q, k and v together, then ffn
    std::array<xrt::bo, core_layers> weights;
    xrt::bo activations, outputs;
    uint32_t eps_bits = 0;

    Controller(xrt::device& device, const std::filesystem::path& root, const nlohmann::json& info) {
        if (info.at("format") != "inkling2-controller-v1" || info.at("image") != "controller.xclbin" ||
            info.at("rows").get<size_t>() != rows || info.at("streams").at("qkv") != "controller-qkv.bin" ||
            info.at("streams").at("ffn") != "controller-ffn.bin")
            fail("unsupported loop controller bundle");
        eps_bits = info.at("eps_bits").get<uint32_t>();
        const xrt::xclbin image((root / "controller.xclbin").string());
        context = xrt::hw_context(device, device.register_xclbin(image));
        kernel = xrt::kernel(context, "MLIR_AIE");
        programs.emplace_back(device, kernel.group_id(1), root / "controller-qkv.bin", 3);
        programs.emplace_back(device, kernel.group_id(1), root / "controller-ffn.bin", 1);
        activations = xrt::ext::bo(device, padded_bo(3 * activation_floats * sizeof(float)));
        outputs = xrt::ext::bo(device, padded_bo(3 * output_floats * sizeof(float)));
    }
};
} // namespace controller_layout

class NpuExperts final : public Inkling2ExpertProjector {
public:
    NpuExperts(const std::filesystem::path& root, unsigned timeout_ms)
        : device_(0u), root_(root), timeout_(timeout_ms) {
        if (!timeout_ms) fail("dispatch timeout must be positive");
        std::ifstream manifest(root / "manifest.json");
        if (!manifest) fail("could not open the routed bundle manifest");
        const auto info = nlohmann::json::parse(manifest);
        const auto abi = info.at("abi").get<std::string>();
        if (const auto key_storage = info.find("wide_old_key_storage"); key_storage != info.end()) {
            if (*key_storage != "lossless-packed" || !info.contains("wide_old_attention"))
                fail("lossless packed keys require a wide older-key attention bundle");
            retire_wide_keys_ = true;
        }
        if (info.contains("wide_old_attention") && abi != "inkling2-routed-v11")
            fail("wide older-key attention requires a v11 bundle");
        const bool dense = abi == "inkling2-routed-v8" || abi == "inkling2-routed-v9" ||
                           abi == "inkling2-routed-v10" || abi == "inkling2-routed-v11";
        const bool grouped_prefill = abi == "inkling2-routed-v7" || dense;
        const bool shared_attention = abi == "inkling2-routed-v6" || grouped_prefill;
        const bool relative_attention = abi == "inkling2-routed-v5" || shared_attention;
        const auto alignment = info.value("attention_chunk_alignment", 0);
        if (alignment && (!shared_attention || alignment != attention_layout::key_rows))
            fail("unsupported attention chunk alignment");
        aligned_chunks_ = alignment != 0;
        const bool compact = abi == "inkling2-routed-v4" || relative_attention;
        const bool block = abi == "inkling2-routed-v3" || compact;
        bool head = abi == "inkling2-routed-v2";
        if (abi != "inkling2-routed-v1" && !head && !block) fail("unsupported routed bundle ABI");
        const size_t cores = compact ? 0 : info.at("cores").get<size_t>();
        if (!compact && (!cores || width / 64 % cores || cores > 12)) fail("unsupported routed core count");
        if (compact) {
            decode_core_rows_ = info.at("decode_core_rows").get<size_t>();
            if (decode_core_rows_ != 1 && decode_core_rows_ != 2 && decode_core_rows_ != 4)
                fail("unsupported compact decode core rows");
            if (shared_attention && decode_core_rows_ != attention_layout::shared_rows)
                fail("shared attention requires four expert core rows");
        }
        if (grouped_prefill) {
            grouped_rows_ = info.at("grouped_block_rows").get<size_t>();
            if (grouped_rows_ != block_layout::tile_m) fail("unsupported grouped expert block row count");
        } else if (block) {
            block_rows_ = info.at("block_rows").get<size_t>();
            if (block_rows_ != 16 && block_rows_ != 32 && block_rows_ != 64)
                fail("unsupported expert block row count");
        }
        constexpr std::array<const char*, 4> names{"iq4-up", "iq4-down", "q8-up", "q8-down"};
        std::vector<size_t> attention_tiers;
        if (block || info.contains("gguf")) {
            const auto& direct = info.at("gguf");
            const auto& contexts = direct.at("contexts");
            const auto& kernels = direct.at("kernels");
            if (direct.at("architecture") != "inkling2_loop")
                fail("invalid Inkling 2 GGUF artifact manifest");
            if (!compact) {
                if (contexts.at("experts") != "final.xclbin") fail("invalid Inkling 2 expert image path");
                for (const char* name : names)
                    if (kernels.at(name).at("insts") != std::string(name) + ".bin")
                        fail("invalid Inkling 2 expert stream path");
            }
            const auto check_format = [&](const std::string& name) {
                if (contexts.at(name) != name + ".xclbin") fail("invalid compact expert image path");
                for (const char* stage : {"-up", "-down"})
                    if (kernels.at(name + stage).at("insts") != name + stage + ".bin")
                        fail("invalid compact expert stream path");
            };
            if (block) {
                head = contexts.contains("head");
                check_format(grouped_prefill ? "prefill-iq4" : "block-iq4");
                check_format(grouped_prefill ? "prefill-q8" : "block-q8");
            }
            if (compact) {
                check_format(shared_attention ? "shared-iq4" : "compact-iq4");
                check_format(shared_attention ? "shared-q8" : "compact-q8");
            }
            if (relative_attention) {
                attention_tiers = info.at("attention_chunks").get<std::vector<size_t>>();
                if (attention_tiers.empty() || attention_tiers.front() != 1 || attention_tiers.back() > 128 ||
                    std::adjacent_find(attention_tiers.begin(), attention_tiers.end(), std::greater_equal<size_t>()) != attention_tiers.end() ||
                    (!shared_attention && contexts.at("relative-attention") != "relative-attention.xclbin"))
                    fail("unsupported relative attention tiers or image");
                if (const auto wide = info.find("wide_old_attention"); wide != info.end()) {
                    if (!aligned_chunks_ || attention_tiers.back() != 128 ||
                        *wide != "bfp16-qk-pv-v1" || contexts.at("wide-old-attention") != "wide-old-attention.xclbin" ||
                        kernels.at("wide-old-attention-4096").at("insts") != "wide-old-attention-4096.bin" ||
                        kernels.at("wide-old-attention-16384").at("insts") != "wide-old-attention-16384.bin")
                        fail("unsupported wide older-key attention bundle");
                    wide_old_enabled_ = true;
                }
                if (const auto mode = info.find("attention_decode"); mode != info.end()) {
                    if (*mode != "keys-as-lanes-v1" || !aligned_chunks_) fail("unsupported decode attention mode");
                    decode_attention_ = true;
                }
                if (const auto block = info.find("near_attention"); block != info.end()) {
                    if (abi != "inkling2-routed-v11" || block->at("format") != "inkling2-near-v1" ||
                        contexts.at("near-attention") != "near-attention.xclbin" ||
                        kernels.at("near-attention").at("insts") != "near-attention.bin" ||
                        block->at("chunks").get<size_t>() != 80)
                        fail("unsupported near attention bundle");
                    near_chunks_ = block->at("chunks").get<size_t>();
                }
                const auto check_tier = [&](const std::string& name) {
                    if (kernels.at(name).at("insts") != name + ".bin") fail("invalid relative attention stream path");
                };
                for (size_t chunks : attention_tiers) {
                    const auto name = "attention-" + std::to_string(chunks);
                    if (shared_attention) {
                        check_tier("shared-iq4-" + name);
                        check_tier("shared-q8-" + name);
                    } else {
                        check_tier(name);
                    }
                }
            }
            if (head &&
                (contexts.at("head") != "head.xclbin" || kernels.at("head").at("insts") != "head.bin"))
                fail("invalid Inkling 2 head artifact paths");
            head_pair_available_ = contexts.contains("head-pair") || kernels.contains("head-pair");
            if (head_pair_available_ &&
                (!head || abi != "inkling2-routed-v11" || contexts.at("head-pair") != "head-pair.xclbin" ||
                 kernels.at("head-pair").at("insts") != "head-pair.bin"))
                fail("invalid Inkling 2 paired head artifacts");
            if (dense) {
                if (info.at("dense_q8_format") != "split-scale" || contexts.at("dense-q8") != "dense-q8.xclbin")
                    fail("invalid dense projection format or image path");
                for (const auto& shape : dense_layout::shapes)
                    if (kernels.at(shape.name).at("insts") != std::string(shape.name) + ".bin")
                        fail("invalid dense projection stream path");
            }
        }
        shared_prefill_q8_ = info.value("shared_prefill_q8", false);
        if (shared_prefill_q8_ && (!info.contains("singleton_decode") || !grouped_rows_ || block_rows_))
            fail("shared grouped Q8 needs grouped prefill and singleton decode");
        if (info.contains("singleton_decode")) {
            if (abi != "inkling2-routed-v11" || !aligned_chunks_)
                fail("singleton decode requires an aligned v11 bundle");
            singleton_ = std::make_unique<SingletonDecode>(device_, root_, info.at("singleton_decode"), attention_tiers);
        }
        if (compact) {
            const auto* shared_kernel = singleton_ ? &singleton_->kernel : nullptr;
            const char* prefix = singleton_ ? "singleton" : shared_attention ? "shared" : "compact";
            decode_formats_[0] = std::make_unique<CompactFormat>(device_, root_,
                                                                  prefix, compact_m, true, true, shared_kernel);
            if (shared_attention)
                decode_formats_[1] = std::make_unique<CompactFormat>(device_, root_, prefix, compact_m, false, true, shared_kernel);
        } else {
            const xrt::xclbin image((root / "final.xclbin").string());
            context_ = xrt::hw_context(device_, device_.register_xclbin(image));
            kernel_ = xrt::kernel(context_, "MLIR_AIE");
            const auto group = kernel_.group_id(1);
            streams_.reserve(names.size());
            for (size_t i = 0; i < names.size(); ++i)
                streams_.emplace_back(device_, group, root / (std::string(names[i]) + ".bin"), cores,
                                      i % 2 ? 1280 : width, i < 2);
        }
        if (relative_attention) {
            attention_ = std::make_unique<attention_layout::Attention>(device_, root_, attention_tiers,
                                                                    shared_attention ? &decode_formats_ : nullptr, singleton_ != nullptr);
        }
        if (head) head_ = std::make_unique<Head>(device_, root, "head", 1);
        if (dense) dense_ = std::make_unique<dense_layout::Dense>(device_, root);
        if (const auto loop = info.find("loop_controller"); loop != info.end())
            controller_ = std::make_unique<controller_layout::Controller>(device_, root, *loop);
        const size_t max_rows = std::max({block_rows_, grouped_rows_ * slots, compact ? compact_m : size_t{0}});
        const size_t dense_input = dense ? block_layout::input_bytes(dense_layout::rows, attention_layout::q_width) : 0;
        const size_t dense_output = dense ? dense_layout::rows * dense_layout::shapes.front().outputs : 0;
        const size_t head_tokens = head_pair_available_ ? 2 : head ? 1 : 0;
        input_ = xrt::ext::bo(device_, padded_bo(std::max({std::max(values, head_tokens * width) * sizeof(uint16_t),
                                                         block_layout::input_bytes(max_rows, width), dense_input})));
        output_ = xrt::ext::bo(device_, padded_bo(std::max({head ? head_rows * head_tokens : values,
                                                          max_rows * width, dense_output}) * sizeof(float)));
    }

    void project(const GgufFile& file, const std::string& tensor, size_t layer, bool down,
                 const ExpertIds& ids, const float* input, float* output) override {
        if (failed_) fail("the previous dispatch failed; recreate the projector");
        const size_t cols = down ? expert_width : width, padded_k = down ? 1280 : width;
        const auto& info = file.tensor(tensor);
        const bool iq4 = info.type == GgufFile::Type::IQ4_NL;
        if ((!iq4 && info.type != GgufFile::Type::Q8_0) ||
            info.dims != std::vector<uint64_t>{cols, width, experts} || layer >= layers)
            fail("unsupported weight tensor " + tensor);
        if (decode_core_rows_) {
            project_compact(file, tensor, layer, down, iq4, cols, ids, input, output);
            return;
        }
        const size_t plane_bytes = width * (cols / 32) * (iq4 ? 18 : 34);
        size_t raw_bytes = 0;
        const auto* src = file.raw(tensor, &raw_bytes);
        if (raw_bytes != experts * plane_bytes) fail("unexpected weight size for " + tensor);

        auto& stream = streams_[(iq4 ? 0 : 2) + down];
        auto& pool = pools_[layer * 2 + down];
        if (!pool.bo) pool.bo = xrt::ext::bo(device_, padded_bo(experts * stream.expert_bytes));
        auto* dst = pool.bo.map<uint8_t*>();
        for (const size_t id : ids) {
            if (id >= experts) fail("invalid expert ID");
            if (pool.packed[id]) continue;
            const size_t offset = id * stream.expert_bytes;
            pools::pack_inkling2_expert(src + id * plane_bytes, iq4, width, cols, padded_k, dst + offset);
            pool.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, stream.expert_bytes, offset);
            pool.packed.set(id);
        }

        auto* words = stream.instruction.map<uint32_t*>();
        for (const auto& fill : stream.fills)
            words[fill.word] = static_cast<uint32_t>(ids[fill.slot] * stream.expert_bytes + fill.intra);
        stream.instruction.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto* activation = input_.map<uint16_t*>();
        for (size_t slot = 0; slot < slots; ++slot) {
            auto* row = activation + slot * width;
            const auto* x = input + slot * cols;
            std::transform(x, x + cols, row, f32_to_bf16);
            std::fill(row + cols, row + width, uint16_t{0});
        }
        input_.sync(XCL_BO_SYNC_BO_TO_DEVICE, values * sizeof(uint16_t), 0);

        const auto* result = dispatch(kernel_, stream.instruction, stream.count, pool.bo, values, tensor);
        std::copy_n(result, values, output);
    }

    ~NpuExperts() override {
        // Buffers are released only after the launch in flight has retired.
        release_lanes();
    }

    bool grouped_overlap() const override { return grouped_rows_ && prefill_overlap_available(); }

    void stage_grouped(const GgufFile& file, const std::string& tensor, size_t layer, bool down, size_t index,
                       const ExpertIds& group, const ExpertRows& counts, const float* input) override {
        if (failed_) fail("the previous dispatch failed; recreate the projector");
        const size_t cols = down ? expert_width : width;
        const auto& info = file.tensor(tensor);
        const bool iq4 = info.type == GgufFile::Type::IQ4_NL;
        if (!grouped_rows_ || index > 1 || (!iq4 && info.type != GgufFile::Type::Q8_0) ||
            info.dims != std::vector<uint64_t>{cols, width, experts} || layer >= layers || !counts.front())
            fail("unsupported overlapped expert group for " + tensor);
        const bool split = shared_prefill_q8_ && !iq4;
        auto& format = block_formats_[iq4 ? 0 : 1];
        auto& pool = (split ? singleton_pools_ : compact_pools_)[layer * 2 + down];
        bool cold = !format || !pool.bo;
        for (size_t slot = 0; slot < slots && !cold; ++slot)
            cold = counts[slot] && (group[slot] >= experts || !pool.packed[group[slot]]);
        // Create contexts and buffers, and upload weights, with nothing in flight.
        if (cold) settle();
        if (!format)
            format = std::make_unique<CompactFormat>(device_, root_, "prefill", slots * grouped_rows_, iq4, true, nullptr, split);
        auto& stream = format->streams[down];
        auto& lane = lane_at((down ? 2 : 0) + index);
        if (&lane == overlap_flight_ || lane.done) fail("an overlapped expert lane is still in use");
        const size_t tile_m = std::max(grouped_rows_, block_layout::decode_tile_m);
        const size_t a_bytes = block_layout::input_bytes(tile_m, cols);
        ExpertIds ids = group;
        auto* activation = lane.input.map<uint8_t*>();
        for (size_t slot = 0; slot < slots; ++slot) {
            if (counts[slot]) pack_compact_expert(pool, stream, file, tensor, iq4, cols, ids[slot], split);
            else ids[slot] = ids.front();
            block_layout::pack_input(input + slot * grouped_rows_ * cols, counts[slot], tile_m, cols,
                                     activation + slot * a_bytes);
        }
        lane.input.sync(XCL_BO_SYNC_BO_TO_DEVICE, slots * a_bytes, 0);
        // The stream takes these expert addresses in launch(), between launches.
        lane.patch = &stream;
        lane.ids = ids;
        lane.instruction = stream.instruction;
        lane.kernel = &format->kernel;
        lane.weights = &pool.bo;
        lane.count = stream.count;
        lane.floats = slots * tile_m * width;
        lane.tensor = tensor;
        lane.expert = true;
        lane.staged = true;
    }

    void start_grouped(bool down, size_t index) override {
        if (failed_) fail("the previous dispatch failed; recreate the projector");
        if (index > 1) fail("invalid overlapped expert lane");
        launch(lanes_[(down ? 2 : 0) + index]);
    }

    void wait_grouped() override {
        settle();
    }

    void read_grouped(bool down, size_t index, const ExpertRows& counts, float* output) override {
        if (index > 1) fail("invalid overlapped expert lane");
        const float* result = fetch(lanes_[(down ? 2 : 0) + index]);
        block_layout::unpack_grouped_output(result, counts.data(), slots, grouped_rows_, decode_core_rows_, width,
                                            output);
    }

    void finish_grouped() noexcept override { release_lanes(); }

    bool block_experts_available() const override { return block_rows_ || grouped_rows_; }
    size_t grouped_block_rows() const override { return grouped_rows_; }
    bool dense_attention_available() const override { return dense_ != nullptr; }
    bool singleton_attention_available() const override { return singleton_ != nullptr; }
    bool aligned_attention_chunks() const override { return aligned_chunks_; }

    void reset_attention_cache(uint64_t sequence) noexcept override {
        if (wide_old_) wide_old_->packed_keys.clear(sequence);
        wide_old_cache_exhausted_ = false;
    }

    bool project_attention_inputs(const GgufFile& file, size_t layer, size_t rows,
                                  const std::array<const float*, 3>& inputs,
                                  const Inkling2AttentionProjections& out) override {
        if (!dense_) return false;
        using namespace attention_layout;
        using namespace dense_layout;
        const Part q{"attn_q", q_width, out.query}, k{"attn_k", kv_width, out.key}, v{"attn_v", kv_width, out.value};
        const Part gate{"attn_gate", q_width, out.gate}, rel{"attn_rel", heads * relative_dim, out.relative};
        const Part sink{"attn_sink_proj", heads, out.sinks};
        if (singleton_ && rows == 1) {
            project_singleton(file, layer, 0, inputs, {
                {q.tensor, q.outputs, 0, q.dst}, {k.tensor, k.outputs, 1, k.dst}, {v.tensor, v.outputs, 2, v.dst},
                {gate.tensor, gate.outputs, 0, gate.dst}, {rel.tensor, rel.outputs, 0, rel.dst},
                {sink.tensor, sink.outputs, 0, sink.dst},
            });
            return true;
        }
        if (inputs[0] == inputs[1] && inputs[0] == inputs[2]) {
            project_dense(file, layer, Fused, rows, inputs[0], {q, k, v, gate, rel, sink});
        } else {
            project_dense(file, layer, Query, rows, inputs[0], {q, gate, rel, sink});
            project_dense(file, layer, Key, rows, inputs[1], {k});
            project_dense(file, layer, Value, rows, inputs[2], {v});
        }
        return true;
    }

    bool project_attention_output(const GgufFile& file, size_t layer, size_t rows, const float* input,
                                  float* output) override {
        if (!dense_) return false;
        if (singleton_ && rows == 1)
            project_singleton(file, layer, 1, {input, input, input}, {{"attn_output", width, 0, output}});
        else project_dense(file, layer, dense_layout::Output, rows, input, {{"attn_output", width, output}});
        return true;
    }

    bool apply_loop_controllers(const GgufFile& file, size_t layer, size_t first, size_t count, size_t rows, float eps,
                                const float* const* x, const float* const* message, float* const* out) override {
        using namespace controller_layout;
        uint32_t eps_bits;
        std::memcpy(&eps_bits, &eps, sizeof(eps_bits));
        const bool known = (first == 0 && count == 3) || (first == 3 && count == 1);
        if (!controller_ || eps_bits != controller_->eps_bits || !known) return false;
        if (failed_) fail("the previous dispatch failed; recreate the projector");
        if (layer < first_layer || layer >= first_layer + core_layers) fail("loop controller outside the core layers");
        auto& weights = controller_->weights[layer - first_layer];
        if (!weights) {
            auto packed = xrt::ext::bo(device_, padded_bo(4 * bytes));
            pack(file, layer, packed.map<uint16_t*>());
            packed.sync(XCL_BO_SYNC_BO_TO_DEVICE, 4 * bytes, 0);
            weights = std::move(packed);
        }
        const auto& program = controller_->programs[first == 3];
        auto* staged = controller_->activations.map<float*>();
        for (size_t at = 0; at < rows; at += controller_layout::rows) {
            const size_t valid = std::min(controller_layout::rows, rows - at);
            for (size_t n = 0; n < count; ++n) {
                float* dst = staged + n * activation_floats;
                for (size_t row = 0; row < valid; ++row) {
                    const float* xs = x[n] + (at + row) * width;
                    const float* ms = message[n] + (at + row) * width;
                    for (size_t j = 0; j < width; j += chunk, dst += 2 * chunk) {
                        std::memcpy(dst, xs + j, chunk * sizeof(float));
                        std::memcpy(dst + chunk, ms + j, chunk * sizeof(float));
                    }
                }
                std::fill_n(dst, (controller_layout::rows - valid) * 2 * width, 0.0f);
            }
            controller_->activations.sync(XCL_BO_SYNC_BO_TO_DEVICE, count * activation_floats * sizeof(float), 0);
            const float* result = dispatch(controller_->kernel, program.instruction, program.count, weights,
                                           controller_->activations, nullptr, controller_->outputs,
                                           count * output_floats, "loop controller");
            for (size_t n = 0; n < count; ++n)
                std::memcpy(out[n] + at * width, result + n * output_floats, valid * width * sizeof(float));
        }
        return true;
    }

    std::vector<float> project_attention(const Inkling2AttentionView& view) override {
        if (!attention_) return {};
        if (failed_) fail("the previous dispatch failed; recreate the projector");
        // Drop prefill-only copies on decode. Retired rows still need their archive.
        if (view.rows == 1 && wide_old_ && !wide_old_->packed_keys.has_retired())
            wide_old_->packed_keys.clear(view.sequence);
        using namespace attention_layout;
        if (near_chunks_ && !near_bypass_ && near_eligible(view)) {
            auto result = project_near(view);
            if (std::getenv("INKLING2_TEST_NEAR_CHECK")) check_near(view, result);
            return result;
        }
        std::vector<float> out(view.rows * q_width);
        auto& attn = *attention_;
        const auto& mode = attn.modes[attn.shared && !attn.singleton && !view.iq4_gate_up ? 1 : 0];
        const size_t query_size = kv_heads * mode.rows_per_column * mode.stride;
        const auto& table = attn.relative_table(view);
        std::vector<float> peaks(view.rows * heads), masses(peaks.size());
        const size_t tier_keys = mode.tiers.back().chunks * key_rows;
        const size_t old_keys = view.position >= extent ? view.position - extent + 1 : 0;
        const size_t shared_end = !view.window && view.rows > query_rows ? old_keys / tier_keys * tier_keys : 0;
        const size_t wide_end = project_wide_old(view, shared_end, out.data(), peaks.data(), masses.data());
        auto* query = attn.query.map<uint8_t*>();
        // A query group's BF16 payload is independent of the K/V tier. With interleaved
        // groups, retain each packed payload on the host and write the tier header on use.
        if (shared_end) {
            attn.packed_queries.resize((view.rows + query_rows - 1) / query_rows * query_size);
            for (size_t token = 0; token < view.rows; token += query_rows)
                pack_query(view, token, std::min(query_rows, view.rows - token), {},
                           attn.packed_queries.data() + token / query_rows * query_size,
                           mode.stride, mode.rows_per_column);
        }
        size_t resident_query = view.rows;
        // Single-tile decode and verify use the keys-as-lanes kernel mode.
        const bool row_mode = decode_attention_ && aligned_chunks_ && attn.shared && view.rows <= decode_rows &&
                              !(wide_old_ && wide_old_->packed_keys.has_retired());
        const auto upload_keys = [&](size_t position, size_t first, size_t last, const Tier& tier) {
            const auto* archive = wide_old_ && wide_old_->packed_keys.has_retired() ? &wide_old_->packed_keys : nullptr;
            if (row_mode) pack_kv_rows(view, position, view.rows, first, last, tier.chunks, attn.kv.map<uint16_t*>());
            else pack_kv(view, table, position, first, last, tier.chunks, attn.kv.map<uint16_t*>(), archive);
            attn.kv.sync(XCL_BO_SYNC_BO_TO_DEVICE, kv_heads * tier.chunks * key_words * sizeof(uint16_t), 0);
        };
        const auto attend_tile = [&](size_t token, size_t rows, size_t first, size_t last, bool sink, const Tier& tier) {
            const QueryHeader header{static_cast<int32_t>(view.position + token), static_cast<int32_t>(first),
                                     static_cast<int32_t>(last), static_cast<int32_t>(view.window), sink,
                                     0, row_mode ? static_cast<int32_t>(rows) : 0, aligned_chunks_};
            if (resident_query != token && !shared_end) {
                pack_query(view, token, rows, header, query, mode.stride, mode.rows_per_column);
            } else {
                if (resident_query != token)
                    std::memcpy(query, attn.packed_queries.data() + token / query_rows * query_size, query_size);
                for (size_t head = 0; head < heads; ++head)
                    std::memcpy(query_record(query, head, mode.stride, mode.rows_per_column), header.data(), sizeof(header));
            }
            resident_query = token;
            if (row_mode)
                for (size_t head = 0; head < heads; ++head)
                    reinterpret_cast<int32_t*>(query_record(query, head, mode.stride, mode.rows_per_column))[5] =
                        static_cast<int32_t>(head % group);
            attn.query.sync(XCL_BO_SYNC_BO_TO_DEVICE, query_size, 0);
            const auto* partial = dispatch(mode.kernel, tier.insts, tier.count, attn.query, attn.kv, attn.result,
                                           kv_heads * mode.output_column, "relative attention");
            if (attn.shared) merge_attention_tile<true>(partial, token, rows, out.data(), peaks.data(), masses.data());
            else merge_attention_tile<false>(partial, token, rows, out.data(), peaks.data(), masses.data());
        };

        // Reuse complete full-attention tiers that are outside the relative
        // extent of every query in the block. Each query still merges the same
        // tiers in ascending key order.
        for (size_t first = wide_end; first < shared_end; first += tier_keys) {
            const size_t last = first + tier_keys;
            const auto& tier = mode.tiers.back();
            upload_keys(view.position, first, last, tier);
            for (size_t token = 0; token < view.rows; token += query_rows)
                attend_tile(token, std::min(query_rows, view.rows - token), first, last, first == 0, tier);
        }
        for (size_t token = 0; token < view.rows; token += query_rows) {
            const size_t rows = std::min(query_rows, view.rows - token);
            const size_t position = view.position + token, end = position + rows;
            const size_t visible_first = view.window && position + 1 > view.window ? position + 1 - view.window : 0;
            const size_t first_key = aligned_chunks_ ? visible_first / key_rows * key_rows : visible_first;
            for (size_t first = std::max(first_key, shared_end); first < end;) {
                const size_t last = std::min(end, first + tier_keys);
                const size_t chunks = (last - first + key_rows - 1) / key_rows;
                const auto& tier = *std::lower_bound(mode.tiers.begin(), mode.tiers.end(), chunks,
                    [](const Tier& t, size_t n) { return t.chunks < n; });
                upload_keys(position, first, last, tier);
                attend_tile(token, rows, first, last, first == first_key, tier);
                first = last;
            }
        }
        for (size_t lane = 0; lane < masses.size(); ++lane) {
            if (!(masses[lane] > 0) || !std::isfinite(masses[lane])) fail("invalid relative attention denominator");
            const float recip = 1.0f / masses[lane];
            auto* dest = out.data() + lane * dim;
            for (size_t d = 0; d < dim; ++d) dest[d] *= recip;
        }
        if (row_mode && std::getenv("INKLING2_TEST_NEAR_CHECK")) {
            const auto err = reference_errors(view, out);
            std::fprintf(stderr, "row-check visit %zu position %zu rows %zu window %zu median %e max %e\n",
                         view.visit, view.position, view.rows, view.window, err[err.size() / 2], err.back());
        }
        return out;
    }

    std::vector<float> project_block(const GgufFile& file, const std::string& tensor, size_t layer, bool down,
                                     size_t expert, const float* input, size_t rows) override {
        if (!block_rows_) return {};
        if (failed_) fail("the previous dispatch failed; recreate the projector");
        const size_t cols = down ? expert_width : width;
        const auto& info = file.tensor(tensor);
        const bool iq4 = info.type == GgufFile::Type::IQ4_NL;
        if ((!iq4 && info.type != GgufFile::Type::Q8_0) || info.dims != std::vector<uint64_t>{cols, width, experts} ||
            layer >= layers || expert >= experts)
            fail("unsupported block weight tensor or expert ID for " + tensor);

        auto& format = block_formats_[iq4 ? 0 : 1];
        if (!format) format = std::make_unique<CompactFormat>(device_, root_, "block", block_rows_, iq4, false);
        auto& stream = format->streams[down];
        auto& pool = compact_pools_[layer * 2 + down];
        const size_t offset = expert * stream.stride;
        pack_compact_expert(pool, stream, file, tensor, iq4, cols, expert);
        auto* words = stream.instruction.map<uint32_t*>();
        for (const auto& [word, within] : stream.fills) words[word] = static_cast<uint32_t>(offset + within);
        stream.instruction.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        std::vector<float> output(rows * width);
        for (size_t at = 0; at < rows; at += block_rows_) {
            const size_t real = std::min(block_rows_, rows - at);
            block_layout::pack_input(input + at * cols, real, block_rows_, cols, input_.map<uint8_t*>());
            input_.sync(XCL_BO_SYNC_BO_TO_DEVICE, block_layout::input_bytes(block_rows_, cols), 0);
            const auto* result = dispatch(format->kernel, stream.instruction, stream.count, pool.bo,
                                          block_rows_ * width, tensor);
            block_layout::unpack_output(result, real, block_rows_, width, output.data() + at * width);
        }
        return output;
    }

    bool project_grouped_blocks(const GgufFile& file, const std::string& tensor, size_t layer, bool down,
                                const ExpertIds& ids, const ExpertRows& counts, const float* input,
                                float* output) override {
        if (!grouped_rows_) return false;
        if (failed_) fail("the previous dispatch failed; recreate the projector");
        const size_t cols = down ? expert_width : width;
        const auto& info = file.tensor(tensor);
        const bool iq4 = info.type == GgufFile::Type::IQ4_NL;
        if ((!iq4 && info.type != GgufFile::Type::Q8_0) || info.dims != std::vector<uint64_t>{cols, width, experts} ||
            layer >= layers || !counts.front())
            fail("unsupported grouped block tensor or empty batch for " + tensor);
        const bool split = shared_prefill_q8_ && !iq4;
        auto& format = block_formats_[iq4 ? 0 : 1];
        if (!format)
            format = std::make_unique<CompactFormat>(device_, root_, "prefill", slots * grouped_rows_, iq4, true, nullptr, split);
        project_packed_group(*format, file, tensor, layer, down, iq4, cols, grouped_rows_, ids, counts, input, output, split);
        return true;
    }

    std::vector<float> project_logits(const GgufFile& file, const float* input) override {
        if (!head_) return {};
        const auto* result = run_head(*head_, file, input, 1);
        return {result, result + vocab};
    }

    std::vector<float> project_logits_pair(const GgufFile& file, const float* input) override {
        if (!head_pair_available_) return {};
        if (!head_pair_) head_pair_ = std::make_unique<Head>(device_, root_, "head-pair", 2);
        const auto* result = run_head(*head_pair_, file, input, 2);
        std::vector<float> logits(2 * vocab);
        for (size_t start = 0; start < vocab; start += head_band) {
            const size_t count = std::min(head_band, vocab - start);
            std::copy_n(result + 2 * start, count, logits.data() + start);
            std::copy_n(result + 2 * start + head_band, count, logits.data() + vocab + start);
        }
        return logits;
    }

private:
    const float* run_head(const Head& head, const GgufFile& file, const float* input, size_t rows) {
        if (failed_) fail("the previous dispatch failed; recreate the projector");
        const std::string tensor = "output.weight";
        if (!head_weights_) {
            const auto& info = file.tensor(tensor);
            if (info.type != GgufFile::Type::Q8_0 || info.dims != std::vector<uint64_t>{width, file_vocab})
                fail("unsupported Q8 head weight tensor " + tensor);
            size_t bytes = 0;
            const auto* src = file.raw(tensor, &bytes);
            if (bytes != file_vocab * (width / 32) * 34) fail("unexpected Q8 head weight size");
            auto packed = xrt::ext::bo(device_, padded_bo(head_bytes));
            pools::pack_q8_head_tiles(src, vocab, width, head_rows, packed.map<uint8_t*>());
            packed.sync(XCL_BO_SYNC_BO_TO_DEVICE, head_bytes, 0);
            head_weights_ = std::move(packed);
        }
        std::transform(input, input + rows * width, input_.map<uint16_t*>(), f32_to_bf16);
        input_.sync(XCL_BO_SYNC_BO_TO_DEVICE, rows * width * sizeof(uint16_t), 0);
        return dispatch(head.kernel, head.instruction, head.count, head_weights_, rows * head_rows, tensor);
    }

    void project_singleton(const GgufFile& file, size_t layer, size_t role, const std::array<const float*, 3>& input,
                           std::initializer_list<shared_dense::Part> parts) {
        auto& weights = singleton_->weights.at(layer).at(role);
        const auto& stream = singleton_->dense.at(role);
        const auto shape = stream.shape;
        if (failed_) fail("the previous dispatch failed; recreate the projector");
        if (!weights) {
            auto packed = xrt::ext::bo(device_, padded_bo(shape.weight_bytes()));
            shared_dense::pack_dense(file, layer, shape, parts, packed.map<uint8_t*>());
            packed.sync(XCL_BO_SYNC_BO_TO_DEVICE, shape.weight_bytes(), 0);
            weights = std::move(packed);
        }
        shared_dense::pack_inputs(shape, input, singleton_->activation, input_.map<uint8_t*>());
        input_.sync(XCL_BO_SYNC_BO_TO_DEVICE, shape.input_bytes(), 0);
        const auto* result = dispatch(singleton_->kernel, stream.instruction, stream.count, weights,
                                      shape.output_bytes() / sizeof(float), "singleton dense");
        const bool separate = input[0] != input[1] || input[0] != input[2];
        shared_dense::unpack_parts(shape, result, separate, parts);
    }

    void project_dense(const GgufFile& file, size_t layer, dense_layout::Role role, size_t count, const float* input,
                       std::initializer_list<dense_layout::Part> parts) {
        using namespace dense_layout;
        if (failed_) fail("the previous dispatch failed; recreate the projector");
        if (layer >= layers || !count) fail("invalid dense projection layer or row count");
        const size_t index = stream_index[role];
        const auto& shape = shapes[index];
        const auto& stream = dense_->streams[index];
        auto& weights = dense_->weights[layer][role];
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        if (!weights) {
            // Assemble the GGUF rows in the same order used to scatter the output.
            const size_t row_bytes = block_layout::weight_bytes(false, 1, shape.inputs);
            std::vector<uint8_t> raw(shape.outputs * row_bytes);
            size_t row = 0;
            for (const auto& part : parts) {
                const auto name = prefix + part.tensor + ".weight";
                const auto& tensor = file.tensor(name);
                size_t bytes = 0;
                const auto* source = file.raw(name, &bytes);
                if (tensor.type != GgufFile::Type::Q8_0 ||
                    tensor.dims != std::vector<uint64_t>{shape.inputs, part.outputs} ||
                    bytes != part.outputs * row_bytes || part.outputs > shape.outputs - row)
                    fail("unsupported dense weight tensor " + name);
                std::memcpy(raw.data() + row * row_bytes, source, bytes);
                row += part.outputs;
            }
            auto packed = xrt::ext::bo(device_, padded_bo(stream.bytes));
            block_layout::pack_weights(raw.data(), false, shape.outputs, shape.inputs, packed.map<uint8_t*>(), true);
            packed.sync(XCL_BO_SYNC_BO_TO_DEVICE, stream.bytes, 0);
            weights = std::move(packed);
        }
        const auto name = prefix + shape.name;
        const size_t input_bytes = block_layout::input_bytes(rows, shape.inputs);
        if (prefill_overlap_available() && count > rows) {
            // The host packs wave w + 1 and scatters wave w - 1 while wave w runs.
            const size_t waves = (count + rows - 1) / rows;
            const auto stage_wave = [&](size_t wave) {
                auto& lane = lane_at(wave & 1);
                if (&lane == overlap_flight_ || lane.done) fail("an overlapped dense lane is still in use");
                const size_t at = wave * rows;
                block_layout::pack_input(input + at * shape.inputs, std::min(rows, count - at), rows, shape.inputs,
                                         lane.input.map<uint8_t*>(), true);
                lane.input.sync(XCL_BO_SYNC_BO_TO_DEVICE, input_bytes, 0);
                lane.patch = nullptr;
                lane.instruction = stream.instruction;
                lane.kernel = &dense_->kernel;
                lane.weights = &weights;
                lane.count = stream.count;
                lane.floats = rows * shape.outputs;
                lane.tensor = name;
                lane.expert = false;
                lane.staged = true;
            };
            try {
                stage_wave(0);
                launch(lanes_[0]);
                for (size_t wave = 0; wave < waves; ++wave) {
                    if (wave + 1 < waves) stage_wave(wave + 1);
                    settle();
                    if (wave + 1 < waves) launch(lanes_[(wave + 1) & 1]);
                    const size_t at = wave * rows, valid = std::min(rows, count - at);
                    const float* result = fetch(lanes_[wave & 1]);
                    if (parts.size() == 1 && parts.begin()->outputs == shape.outputs) {
                        block_layout::unpack_output(result, valid, rows, shape.outputs,
                                                    parts.begin()->dst + at * shape.outputs);
                        continue;
                    }
                    auto* unpacked = dense_->unpacked.data();
                    block_layout::unpack_output(result, valid, rows, shape.outputs, unpacked);
                    size_t col = 0;
                    for (const auto& part : parts) {
                        for (size_t row = 0; row < valid; ++row)
                            std::copy_n(unpacked + row * shape.outputs + col, part.outputs,
                                        part.dst + (at + row) * part.outputs);
                        col += part.outputs;
                    }
                }
            } catch (...) {
                release_lanes();
                throw;
            }
            return;
        }
        for (size_t at = 0; at < count; at += rows) {
            const size_t valid = std::min(rows, count - at);
            block_layout::pack_input(input + at * shape.inputs, valid, rows, shape.inputs, input_.map<uint8_t*>(), true);
            input_.sync(XCL_BO_SYNC_BO_TO_DEVICE, input_bytes, 0);
            const auto* result = dispatch(dense_->kernel, stream.instruction, stream.count, weights,
                                          rows * shape.outputs, name);
            if (parts.size() == 1 && parts.begin()->outputs == shape.outputs) {
                block_layout::unpack_output(result, valid, rows, shape.outputs, parts.begin()->dst + at * shape.outputs);
                continue;
            }
            auto* unpacked = dense_->unpacked.data();
            block_layout::unpack_output(result, valid, rows, shape.outputs, unpacked);
            size_t col = 0;
            for (const auto& part : parts) {
                for (size_t row = 0; row < valid; ++row)
                    std::copy_n(unpacked + row * shape.outputs + col, part.outputs, part.dst + (at + row) * part.outputs);
                col += part.outputs;
            }
        }
    }

    void pack_compact_expert(Pool& pool, const CompactStream& stream, const GgufFile& file,
                             const std::string& tensor, bool iq4, size_t cols, size_t expert, bool split_scale = false) {
        if (expert >= experts) fail("invalid expert ID");
        if (pool.packed[expert]) return;
        size_t bytes = 0;
        const auto* src = file.raw(tensor, &bytes);
        const size_t plane_bytes = block_layout::weight_bytes(iq4, width, cols);
        if (bytes != experts * plane_bytes) fail("unexpected compact expert weight size for " + tensor);
        if (!pool.bo) pool.bo = xrt::ext::bo(device_, padded_bo(experts * stream.stride));
        const size_t offset = expert * stream.stride;
        block_layout::pack_weights(src + expert * plane_bytes, iq4, width, cols, pool.bo.map<uint8_t*>() + offset, split_scale);
        pool.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, stream.bytes, offset);
        pool.packed.set(expert);
    }

    void project_compact(const GgufFile& file, const std::string& tensor, size_t layer, bool down, bool iq4,
                         size_t cols, const ExpertIds& ids, const float* input, float* output) {
        auto& format = decode_formats_[iq4 ? 0 : 1];
        if (!format) format = std::make_unique<CompactFormat>(device_, root_, "compact", compact_m, iq4, true);
        constexpr ExpertRows counts{1, 1, 1, 1, 1, 1, 1, 1};
        project_packed_group(*format, file, tensor, layer, down, iq4, cols, 1, ids, counts, input, output, singleton_ && !iq4);
    }

    void project_packed_group(CompactFormat& format, const GgufFile& file, const std::string& tensor,
                              size_t layer, bool down, bool iq4, size_t cols, size_t slot_rows,
                              ExpertIds ids, const ExpertRows& counts, const float* input, float* output, bool split_scale = false) {
        auto& stream = format.streams[down];
        auto& pool = (split_scale ? singleton_pools_ : compact_pools_)[layer * 2 + down];
        const size_t tile_m = std::max(slot_rows, block_layout::decode_tile_m);
        const size_t a_bytes = block_layout::input_bytes(tile_m, cols);
        auto* activation = input_.map<uint8_t*>();
        for (size_t slot = 0; slot < slots; ++slot) {
            if (counts[slot]) pack_compact_expert(pool, stream, file, tensor, iq4, cols, ids[slot], split_scale);
            else ids[slot] = ids.front();
            block_layout::pack_input(input + slot * slot_rows * cols, counts[slot], tile_m, cols,
                                     activation + slot * a_bytes);
        }
        input_.sync(XCL_BO_SYNC_BO_TO_DEVICE, slots * a_bytes, 0);
        auto* words = stream.instruction.map<uint32_t*>();
        for (const auto& [word, within] : stream.fills)
            words[word] = static_cast<uint32_t>(ids[within / stream.bytes] * stream.stride);
        stream.instruction.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        const auto* result = dispatch(format.kernel, stream.instruction, stream.count, pool.bo,
                                      slots * tile_m * width, tensor);
        block_layout::unpack_grouped_output(result, counts.data(), slots, slot_rows, decode_core_rows_, width, output);
    }

    // Test-only: per-head relative L2 of attention rows against an FP64 reference (every seventh row).
    static std::vector<double> reference_errors(const Inkling2AttentionView& view, const std::vector<float>& result) {
        using namespace attention_layout;
        std::vector<double> errors;
        for (size_t t = 0; t < view.rows; t += 7) {
            const size_t q = view.position + t;
            const size_t first = view.window && q + 1 > view.window ? q + 1 - view.window : 0;
            for (size_t h = 0; h < heads; ++h) {
                const size_t lane = t * heads + h, head = h / group;
                const auto row_of = [&](const uint16_t* cached, const uint16_t* pending, size_t k) {
                    return k < view.position ? cached + (view.ring ? k % view.ring : k) * kv_width
                                             : pending + (k - view.position) * kv_width;
                };
                std::vector<double> scores;
                double peak = view.sinks[lane];
                for (size_t k = first; k <= q; ++k) {
                    const uint16_t* key = row_of(view.cached_key, view.pending_key, k);
                    double s = 0;
                    for (size_t d = 0; d < dim; ++d)
                        s += double(view.query[lane * dim + d]) * bf16_to_f32(key[head * dim + d]);
                    s *= 0.08838834764831845;
                    if (q - k < extent)
                        for (size_t d = 0; d < relative_dim; ++d)
                            s += double(view.relative[lane * relative_dim + d]) * view.table[(q - k) * relative_dim + d];
                    scores.push_back(s);
                    peak = std::max(peak, s);
                }
                std::vector<double> want(dim, 0.0);
                double mass = std::exp(double(view.sinks[lane]) - peak);
                for (size_t k = first; k <= q; ++k) {
                    const double p = std::exp(scores[k - first] - peak);
                    mass += p;
                    const uint16_t* value = row_of(view.cached_value, view.pending_value, k);
                    for (size_t d = 0; d < dim; ++d) want[d] += p * bf16_to_f32(value[head * dim + d]);
                }
                double norm = 0, err = 0;
                for (size_t d = 0; d < dim; ++d) {
                    want[d] /= mass;
                    norm += want[d] * want[d];
                    err += std::pow(result[lane * dim + d] - want[d], 2);
                }
                errors.push_back(std::sqrt(err / norm));
            }
        }
        std::sort(errors.begin(), errors.end());
        return errors;
    }

    // Test-only: compare the near route and the relative-attention route with an FP64 reference.
    void check_near(const Inkling2AttentionView& view, const std::vector<float>& result) {
        near_bypass_ = true;
        const auto old = project_attention(view);
        near_bypass_ = false;
        const auto near_err = reference_errors(view, result), old_err = reference_errors(view, old);
        std::fprintf(stderr, "near-check layer %zu visit %zu position %zu window %zu near %f %f old %f %f\n",
                     view.layer, view.visit, view.position, view.window, near_err[near_err.size() / 2],
                     near_err.back(), old_err[old_err.size() / 2], old_err.back());
    }

    bool near_eligible(const Inkling2AttentionView& view) const {
        using Near = attention_layout::NearAttention;
        // Verify and decode keep their own arithmetic.
        return view.rows > attention_layout::query_rows && view.rows <= Near::rows && view.position % Near::chunk == 0;
    }

    // Older full-attention keys run first, oldest segment first, then the relative segment that ends at the block.
    std::vector<float> project_near(const Inkling2AttentionView& view) {
        using namespace attention_layout;
        using Near = NearAttention;
        if (!near_) near_ = std::make_unique<Near>(device_, root_, near_chunks_);
        auto& attention = *near_;
        std::vector<float> out(view.rows * q_width, 0.0f), peaks(view.rows * heads), masses(view.rows * heads);
        std::copy_n(view.sinks, peaks.size(), peaks.begin());
        std::fill(masses.begin(), masses.end(), 1.0f);
        const long kbase = static_cast<long>(view.position) - static_cast<long>(extent);
        long covered = 0;
        if (!view.window && kbase > 0)
            covered = static_cast<long>(project_wide_old(view, static_cast<size_t>(kbase), out.data(), peaks.data(),
                                                         masses.data()));
        // Retired raw keys are readable only through the wide route or the archive in the 16-row route.
        if (wide_old_ && !view.window && covered < static_cast<long>(wide_old_->packed_keys.retired_end(view.visit))) {
            if (covered) fail("near attention cannot follow a partial wide older-key range");
            near_bypass_ = true;
            auto fallback = project_attention(view);
            near_bypass_ = false;
            return fallback;
        }
        const auto& strips = attention.layer_strips(device_, view);
        attention.pack_query(view);
        const long span = static_cast<long>(attention.span);
        long first = view.window ? kbase : kbase - (kbase > covered ? (kbase - covered + span - 1) / span * span : 0);
        for (; first <= kbase; first += span) {
            const long low = std::max(first, std::max(covered, 0L));
            attention.set_meta(view, first, low);
            attention.pack_kv(view, first, low);
            const auto* state = dispatch(attention.kernel, attention.insts, attention.count, attention.query,
                                         attention.kv, &strips, attention.output, Near::output_values,
                                         "near attention");
            Near::merge(state, out.data(), peaks.data(), masses.data(), view.rows);
        }
        for (size_t lane = 0; lane < masses.size(); ++lane) {
            if (!(masses[lane] > 0) || !std::isfinite(masses[lane])) fail("invalid near attention denominator");
            const float recip = 1.0f / masses[lane];
            auto* dest = out.data() + lane * dim;
            for (size_t d = 0; d < dim; ++d) dest[d] *= recip;
        }
        return out;
    }

    size_t project_wide_old(const Inkling2AttentionView& view, size_t end, float* out, float* peaks, float* masses) {
        using namespace attention_layout;
        using Wide = WideOldAttention;
        if (!wide_old_enabled_ || wide_old_cache_exhausted_ || view.window || view.ring ||
            view.rows < Wide::rows / 2 || view.rows > Wide::rows)
            return 0;
        end = end / Wide::short_keys * Wide::short_keys;
        if (!end) return 0;
        if (!wide_old_) wide_old_ = std::make_unique<Wide>(device_, root_);
        auto& wide = *wide_old_;
#ifdef _OPENMP
        const int threads = std::min(omp_get_max_threads(), static_cast<int>(kv_heads));
#else
        const int threads = 1;
#endif
        try {
            wide.packed_keys.prepare(view.sequence, view.visit, view.cached_key, view.position, end, threads,
                                     view.key_storage, retire_wide_keys_ ? view.retire_key_rows : nullptr);
        } catch (const std::bad_alloc&) {
            if (!wide.packed_keys.has_retired()) wide.packed_keys.clear(view.sequence);
            wide_old_cache_exhausted_ = true;
            return 0;
        }
        const size_t lanes = view.rows * heads;
        std::copy_n(view.sinks, lanes, peaks);
        std::fill_n(masses, lanes, 1.0f);
        wide.upload_query(view);
        for (size_t first = 0; first < end;) {
            const bool large = end - first >= Wide::long_keys;
            const size_t count = large ? Wide::long_keys : Wide::short_keys;
            const auto& stream = wide.streams[large];
            wide.upload_cache(view, first, count);
            const auto* state = dispatch(wide.kernel, stream.insts, stream.count, wide.query, wide.key_value,
                                         wide.output, Wide::output_values, "wide older-key attention");
            Wide::merge(state, out, peaks, masses, view.rows);
            first += count;
        }
        return end;
    }

    const float* dispatch(const xrt::kernel& kernel, const xrt::bo& instruction, size_t count,
                          const xrt::bo& weights, size_t rows, const std::string& tensor) {
        return dispatch(kernel, instruction, count, weights, input_, output_, rows, tensor);
    }

    const float* dispatch(const xrt::kernel& kernel, const xrt::bo& instruction, size_t count,
                          const xrt::bo& first, const xrt::bo& second, xrt::bo& output,
                          size_t rows, const std::string& tensor) {
        return dispatch(kernel, instruction, count, first, second, nullptr, output, rows, tensor);
    }

    const float* dispatch(const xrt::kernel& kernel, const xrt::bo& instruction, size_t count,
                          const xrt::bo& first, const xrt::bo& second, const xrt::bo* third, xrt::bo& output,
                          size_t rows, const std::string& tensor) {
        xrt::run run(kernel);
        run.set_arg(0, 3);
        run.set_arg(1, instruction);
        run.set_arg(2, static_cast<int>(count));
        run.set_arg(3, first);
        run.set_arg(4, second);
        if (third) run.set_arg(5, *third);
        run.set_arg(third ? 6 : 5, output);
        failed_ = true;
        run.start();
        const auto state = run.wait(timeout_);
        if (state != ERT_CMD_STATE_COMPLETED) {
            if (state == ERT_CMD_STATE_TIMEOUT) run.abort();
            fail("dispatch failed for " + tensor + " (state " + std::to_string(state) + ")");
        }
        output.sync(XCL_BO_SYNC_BO_FROM_DEVICE, rows * sizeof(float), 0);
        NpuExperts::read_fence();
        const auto* result = output.map<const float*>();
        if (!all_finite(result, rows))
            fail("non-finite output from " + tensor);
        failed_ = false;
        return result;
    }

    xrt::device device_;
    xrt::hw_context context_;
    xrt::kernel kernel_;
    std::vector<Stream> streams_;
    std::array<Pool, 2 * layers> pools_;
    std::unique_ptr<Head> head_, head_pair_;
    xrt::bo input_, output_, head_weights_;
    bool head_pair_available_ = false;
    std::filesystem::path root_;
    bool shared_prefill_q8_ = false;
    size_t block_rows_ = 0, grouped_rows_ = 0, decode_core_rows_ = 0;
    std::array<std::unique_ptr<CompactFormat>, 2> block_formats_, decode_formats_;
    std::unique_ptr<attention_layout::Attention> attention_;
    std::unique_ptr<attention_layout::WideOldAttention> wide_old_;
    std::unique_ptr<attention_layout::NearAttention> near_;
    size_t near_chunks_ = 0;
    bool decode_attention_ = false;
    static constexpr size_t decode_rows = 8;
    bool near_bypass_ = false;
    std::unique_ptr<dense_layout::Dense> dense_;
    std::unique_ptr<controller_layout::Controller> controller_;
    std::unique_ptr<SingletonDecode> singleton_;
    std::array<Pool, 2 * layers> compact_pools_, singleton_pools_;
    std::chrono::milliseconds timeout_;
    bool failed_ = false;
    bool aligned_chunks_ = false;
    bool wide_old_enabled_ = false;
    bool wide_old_cache_exhausted_ = false;
    bool retire_wide_keys_ = false;

    struct OverlapLane {
        xrt::bo input, output, instruction;
        const xrt::kernel* kernel = nullptr;
        const xrt::bo* weights = nullptr;
        // An expert stream whose weight fills take `ids` at launch. Dense streams need no patch.
        const CompactStream* patch = nullptr;
        ExpertIds ids{};
        size_t count = 0, floats = 0;
        std::string tensor;
        bool expert = false, staged = false, done = false, fresh = false;
    };

    static constexpr bool prefill_overlap_available() {
#ifdef _WIN32
        return true;
#else
        return false;
#endif
    }

    static bool all_finite(const float* data, size_t count) {
        size_t i = 0;
#if defined(__AVX2__)
        {
            // A float is non-finite exactly when all eight exponent bits are set.
            const __m256i exponent = _mm256_set1_epi32(0x7f800000);
            __m256i invalid = _mm256_setzero_si256();
            for (; count - i >= 8; i += 8) {
                const __m256i bits = _mm256_castps_si256(_mm256_loadu_ps(data + i));
                invalid = _mm256_or_si256(invalid, _mm256_cmpeq_epi32(_mm256_and_si256(bits, exponent), exponent));
            }
            if (!_mm256_testz_si256(invalid, invalid)) return false;
        }
#endif
        return std::all_of(data + i, data + count, [](float value) { return std::isfinite(value); });
    }

    // The Windows driver's small device-to-host sync is an unfenced CLFLUSH loop (OPEN-REQUEST-ISOLATION).
    static void read_fence() {
#if defined(_M_X64) || defined(__x86_64__)
        _mm_mfence();
#endif
    }

    OverlapLane& lane_at(size_t index) {
        auto& lane = lanes_[index];
        if (!lane.input) {
            settle();
            lane.input = xrt::ext::bo(device_, input_.size());
            lane.output = xrt::ext::bo(device_, output_.size());
        }
        return lane;
    }

    void launch(OverlapLane& lane) {
        if (overlap_flight_) fail("an overlapped launch is already in flight");
        if (!lane.staged || lane.done) fail("an overlapped lane is not ready to launch");
        if (lane.patch) {
            // A stream is patched only while no launch uses it (OPEN-DECODE-PIPELINE).
            auto* words = lane.instruction.map<uint32_t*>();
            for (const auto& [word, within] : lane.patch->fills)
                words[word] = static_cast<uint32_t>(lane.ids[within / lane.patch->bytes] * lane.patch->stride);
            lane.instruction.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        xrt::run next(*lane.kernel);
        next.set_arg(0, 3);
        next.set_arg(1, lane.instruction);
        next.set_arg(2, static_cast<int>(lane.count));
        next.set_arg(3, *lane.weights);
        next.set_arg(4, lane.input);
        next.set_arg(5, lane.output);
        lane.staged = false;
        overlap_run_ = std::move(next);
        overlap_flight_ = &lane;
        failed_ = true;
        overlap_run_.start();
        failed_ = false;
    }

    // Wait for the launch in flight. A second queued launch gave different bits from run to run
    // in the block GEMM route (OPEN-PREFILL-BATCH), so this route never queues one.
    void settle() {
        if (!overlap_flight_) return;
        auto& lane = *overlap_flight_;
        failed_ = true;
        const auto state = overlap_run_.wait(timeout_);
        if (state != ERT_CMD_STATE_COMPLETED) {
            if (state == ERT_CMD_STATE_TIMEOUT) overlap_run_.abort();
            overlap_flight_ = nullptr;
            fail("dispatch failed for " + lane.tensor + " (state " + std::to_string(state) + ")");
        }
        overlap_flight_ = nullptr;
        failed_ = false;
        lane.done = lane.fresh = true;
    }

    void read_back(OverlapLane& lane) {
        if (!lane.fresh) return;
        lane.output.sync(XCL_BO_SYNC_BO_FROM_DEVICE, lane.floats * sizeof(float), 0);
        read_fence();
        lane.fresh = false;
    }

    const float* fetch(OverlapLane& lane) {
        if (&lane == overlap_flight_ || !lane.done) fail("an overlapped result is not ready");
        read_back(lane);
        const auto* result = lane.output.map<const float*>();
        const bool finite = all_finite(result, lane.floats);
        lane.done = false;
        if (!finite) {
            failed_ = true;
            fail("non-finite output from " + lane.tensor);
        }
        return result;
    }

    void release_lanes() noexcept {
        if (overlap_flight_) {
            const bool previously_failed = failed_;
            try {
                settle();
                failed_ = previously_failed;
            } catch (...) {
                failed_ = true;
                if (overlap_flight_) {
                    try { overlap_run_.abort(); } catch (...) {}
                    overlap_flight_ = nullptr;
                }
            }
        }
        for (auto& lane : lanes_) lane.staged = lane.done = lane.fresh = false;
    }

    // Lanes 0 and 1 serve gate/up and dense waves. Lanes 2 and 3 serve down.
    std::array<OverlapLane, 4> lanes_;
    xrt::run overlap_run_;
    OverlapLane* overlap_flight_ = nullptr;
};

}  // namespace

std::unique_ptr<Inkling2ExpertProjector> make_inkling2_npu_experts(const std::string& bundle, unsigned timeout_ms) {
    return std::make_unique<NpuExperts>(bundle, timeout_ms);
}

}  // namespace open_qwen36
