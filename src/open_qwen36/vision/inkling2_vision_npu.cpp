#include "open_qwen36/vision/inkling2_vision_npu.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <utility>

#include "nlohmann/json.hpp"
#include "open_qwen36/q4nx_file.hpp"
#include "open_qwen36/vision/vit.hpp"
#include "open_qwen36/weight_file.hpp"
#include "xrt/experimental/xrt_ext.h"
#include "xrt/experimental/xrt_xclbin.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

namespace open_qwen36::vision {
namespace {

constexpr int block = 256, heads = 16, dim = 64, width = heads * dim, max_rows = 42437;
constexpr int max_projection = 4096;
constexpr int key_chunk = 64;
constexpr int query_lanes = 32, query_record = 64 * dim, query_plane = block * dim;
constexpr int query_head = query_record + 2 * query_plane;
constexpr float query_scale = 1.4426950408889634f / 8.0f;  // log2(e) / sqrt(dim)
constexpr std::array<std::pair<int, int>, 6> projections{{
    {1024, 1024}, {1024, 3072}, {1024, 4096}, {4096, 1024}, {4096, 2304}, {4096, 4096},
}};

[[noreturn]] void fail(const std::string& reason) {
    throw std::runtime_error("inkling2 vision NPU: " + reason);
}

struct Image {
    xrt::hw_context context;
    xrt::kernel kernel;

    Image(xrt::device& device, const std::filesystem::path& path)
        : context(device, device.register_xclbin(xrt::xclbin(path.string()))), kernel(context, "MLIR_AIE") {}
};

struct Stream {
    xrt::bo instruction;
    int words;

    Stream(const xrt::device& device, const Image& image, const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        const auto bytes = file.tellg();
        if (bytes < 16 || bytes > 1024 * 1024 || bytes % 4) fail("invalid instruction stream " + path.string());
        const size_t size = static_cast<size_t>(bytes);
        words = static_cast<int>(size / sizeof(uint32_t));
        instruction = xrt::bo(device, size, xrt::bo::flags::cacheable, image.kernel.group_id(1));
        file.seekg(0);
        if (!file.read(instruction.map<char*>(), bytes)) fail("could not read " + path.string());
        instruction.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
};

xrt::bo device_buffer(const xrt::device& device, size_t bytes) {
    constexpr size_t alignment = 1u << 20;
    return xrt::ext::bo(device, (bytes + alignment - 1) / alignment * alignment);
}

// The pretiled BF16 GEMM streams [K/64][N/32][8][4][8][8] from DDR.
void pack_weight(const Linear& weights, uint16_t* packed) {
    constexpr int tile_k = 64, tile_n = 32, mac = 8;
#pragma omp parallel for schedule(static)
    for (int k = 0; k < weights.in; k += tile_k) {
        auto* dst = packed + static_cast<size_t>(k) * weights.out;
        for (int n = 0; n < weights.out; n += tile_n)
            for (int kb = 0; kb < tile_k; kb += mac)
                for (int nb = 0; nb < tile_n; nb += mac)
                    for (int ki = 0; ki < mac; ++ki)
                        for (int ni = 0; ni < mac; ++ni)
                            *dst++ = weights.w[static_cast<size_t>(n + nb + ni) * weights.in + k + kb + ki];
    }
}

float rotate(const float* row, int d, const float* cos, const float* sin) {
    const float other = (d & 1) ? row[d - 1] : -row[d + 1];
    return row[d] * cos[d] + other * sin[d];
}

float round_to_fp16(float value) {
    return fp16_to_f32(f32_to_fp16(value));
}

class NpuVision final : public Inkling2VisionBackend {
public:
    NpuVision(const std::filesystem::path& root, const std::vector<int>& tiers,
              bool split_keys, bool split_values, unsigned timeout_ms)
        : gemm_(device_, root / "vision-gemm.xclbin"), attention_(device_, root / "vision-attention.xclbin"),
          activations_(device_buffer(device_, block * max_projection * sizeof(uint16_t))),
          query_(device_buffer(device_, heads * query_head * sizeof(uint16_t))),
          output_(device_buffer(device_, block * max_projection * sizeof(float))), timeout_(timeout_ms),
          split_keys_(split_keys), split_values_(split_values) {
        for (const auto& shape : projections)
            gemm_streams_.try_emplace(shape, device_, gemm_, root / (gemm_name(shape) + ".bin"));
        for (const int tier : tiers)
            attention_streams_.try_emplace(tier, device_, attention_, root / ("vision-attention-" + std::to_string(tier) + ".bin"));
        std::fill_n(output_.map<float*>(), block * max_projection, 0.0f);
        output_.sync(XCL_BO_SYNC_BO_TO_DEVICE, block * max_projection * sizeof(float), 0);
    }

    static std::string gemm_name(const std::pair<int, int>& shape) {
        return "vision-gemm-" + std::to_string(shape.first) + "-" + std::to_string(shape.second);
    }

    void project(const float* input, int rows, const Linear& weights, float* output) override {
        if (failed_) fail("previous dispatch failed");
        const auto& stream = gemm_streams_.at({weights.in, weights.out});
        const auto& matrix = resident_weight(weights);
        const size_t count = block * static_cast<size_t>(weights.in);
        auto* dst = activations_.map<uint16_t*>();
        for (int at = 0; at < rows; at += block) {
            const int valid = std::min(block, rows - at);
            const size_t values = valid * static_cast<size_t>(weights.in);
            const auto* src = input + static_cast<size_t>(at) * weights.in;
#pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(values); ++i) dst[i] = f32_to_bf16(src[i]);
            std::fill(dst + values, dst + count, uint16_t{0});
            activations_.sync(XCL_BO_SYNC_BO_TO_DEVICE, count * sizeof(uint16_t), 0);
            const auto* result = dispatch(gemm_, stream, activations_, matrix, valid * weights.out);
#pragma omp parallel for schedule(static)
            for (int row = 0; row < valid; ++row)
                for (int col = 0; col < weights.out; ++col)
                    output[static_cast<size_t>(at + row) * weights.out + col] = result[row * weights.out + col] + weights.b[col];
        }
    }

    void attend(const float* qkv, int rows, const std::vector<float>& cos,
                const std::vector<float>& sin, float* output) override {
        if (failed_) fail("previous dispatch failed");
        const auto tier = attention_streams_.lower_bound(rows);
        if (rows < 1 || rows > max_rows || tier == attention_streams_.end()) fail("unsupported attention length");
        pack_kv(qkv, rows, tier->first, cos, sin);
        for (int at = 0; at < rows; at += block) {
            const int valid = std::min(block, rows - at);
            pack_queries(qkv, at, valid, rows, cos, sin);
            const auto* result = dispatch(attention_, tier->second, query_, kv_, valid * width);
            std::copy_n(result, valid * width, output + static_cast<size_t>(at) * width);
        }
    }

private:
    xrt::device device_{0};
    Image gemm_, attention_;
    std::map<std::pair<int, int>, Stream> gemm_streams_;
    std::map<int, Stream> attention_streams_;
    std::map<const Linear*, xrt::bo> weights_;
    xrt::bo activations_, query_, output_, kv_;
    int cached_tier_ = 0;
    const std::chrono::milliseconds timeout_;
    const bool split_keys_, split_values_;
    bool failed_ = false;

    const xrt::bo& resident_weight(const Linear& weight) {
        auto found = weights_.find(&weight);
        if (found != weights_.end()) return found->second;
        const size_t bytes = weight.w.size() * sizeof(uint16_t);
        auto buffer = device_buffer(device_, bytes);
        pack_weight(weight, buffer.map<uint16_t*>());
        buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, 0);
        return weights_.emplace(&weight, std::move(buffer)).first->second;
    }

    void pack_kv(const float* qkv, int rows, int tier, const std::vector<float>& cos, const std::vector<float>& sin) {
        const size_t plane = key_chunk * width;
        const size_t planes = 2 + split_keys_ + split_values_;
        const size_t bytes = static_cast<size_t>(tier) * planes * width * sizeof(uint16_t);
        if (tier > cached_tier_) {
            kv_ = device_buffer(device_, bytes);
            cached_tier_ = tier;
        }
        auto* dst = kv_.map<uint16_t*>();
#pragma omp parallel for schedule(static)
        for (int row = 0; row < tier; ++row) {
            // The split layout keeps [K hi, K lo, V hi, optional V lo] together for each 64 keys.
            // The FIFOs read both parts in one descriptor while each core holds one at a time.
            auto* key = split_keys_ ? dst + (row / key_chunk) * planes * plane + (row % key_chunk) * width
                                    : dst + static_cast<size_t>(row) * 2 * width;
            auto* low = split_keys_ ? key + plane : nullptr;
            auto* value = split_keys_ ? key + 2 * plane : key + width;
            auto* value_low = split_values_ ? value + plane : nullptr;
            if (row >= rows) {
                std::fill_n(key, width, uint16_t{0});
                if (low) std::fill_n(low, width, uint16_t{0});
                std::fill_n(value, width, uint16_t{0});
                if (value_low) std::fill_n(value_low, width, uint16_t{0});
                continue;
            }
            const float* src = qkv + static_cast<size_t>(row) * 3 * width + width;
            const auto* c = cos.data() + row * dim;
            const auto* s = sin.data() + row * dim;
            for (int head = 0; head < heads; ++head)
                for (int d = 0; d < dim; ++d) {
                    const int lane = head * dim + d;
                    const float k = rotate(src + head * dim, d, c, s);
                    // v11 matches the GGUF oracle's FP16 attention operands
                    // before representing them in the NPU's two BF16 planes.
                    const float rotated = split_values_ ? round_to_fp16(k) : k;
                    const float v = split_values_ ? round_to_fp16(src[width + lane]) : src[width + lane];
                    key[lane] = f32_to_bf16(rotated);
                    if (low) low[lane] = f32_to_bf16(rotated - bf16_to_f32(key[lane]));
                    value[lane] = f32_to_bf16(v);
                    if (value_low) value_low[lane] = f32_to_bf16(v - bf16_to_f32(value[lane]));
                }
        }
        kv_.sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, 0);
    }

    void pack_queries(const float* qkv, int at, int valid, int rows,
                      const std::vector<float>& cos, const std::vector<float>& sin) {
        auto* dst = query_.map<uint16_t*>();
        std::fill_n(dst, heads * query_head, uint16_t{0});
        const int32_t last_key = rows - 1;
#pragma omp parallel for schedule(static)
        for (int head = 0; head < heads; ++head) {
            auto* header = dst + head * query_head;
            std::memcpy(header, &last_key, sizeof(last_key));
            auto* hi = header + query_record;
            auto* lo = hi + query_plane;
            for (int row = 0; row < valid; ++row) {
                const int pos = at + row;
                const float* src = qkv + static_cast<size_t>(pos) * 3 * width + head * dim;
                const auto* c = cos.data() + pos * dim;
                const auto* s = sin.data() + pos * dim;
                for (int d = 0; d < dim; ++d) {
                    const int index = (row / query_lanes * dim + d) * query_lanes + row % query_lanes;
                    const float rotated = rotate(src, d, c, s);
                    const float q = (split_values_ ? round_to_fp16(rotated) : rotated) * query_scale;
                    hi[index] = f32_to_bf16(q);
                    lo[index] = f32_to_bf16(q - bf16_to_f32(hi[index]));
                }
            }
        }
        query_.sync(XCL_BO_SYNC_BO_TO_DEVICE, heads * query_head * sizeof(uint16_t), 0);
    }

    const float* dispatch(const Image& image, const Stream& stream, const xrt::bo& a, const xrt::bo& b, int count) {
        xrt::run run(image.kernel);
        run.set_arg(0, 3);
        run.set_arg(1, stream.instruction);
        run.set_arg(2, stream.words);
        run.set_arg(3, a);
        run.set_arg(4, b);
        run.set_arg(5, output_);
        failed_ = true;
        run.start();
        const auto state = run.wait(timeout_);
        if (state != ERT_CMD_STATE_COMPLETED) {
            if (state == ERT_CMD_STATE_TIMEOUT) run.abort();
            fail("dispatch returned state " + std::to_string(state));
        }
        output_.sync(XCL_BO_SYNC_BO_FROM_DEVICE, count * sizeof(float), 0);
        const auto* result = output_.map<const float*>();
        if (!std::all_of(result, result + count, [](float value) { return std::isfinite(value); }))
            fail("nonfinite dispatch result");
        failed_ = false;
        return result;
    }
};

}  // namespace

std::unique_ptr<Inkling2VisionBackend> make_inkling2_vision_npu(const std::string& bundle, unsigned timeout_ms) {
    const std::filesystem::path root(bundle);
    std::ifstream manifest(root / "manifest.json");
    if (!manifest) fail("missing bundle manifest");
    const auto info = nlohmann::json::parse(manifest);
    const auto abi = info.value("abi", std::string{});
    const bool split_values = abi == "inkling2-routed-v11";
    const bool split_keys = split_values || abi == "inkling2-routed-v10";
    if (!split_keys && abi != "inkling2-routed-v9") return {};
    const auto& vision = info.at("vision");
    const auto tiers = vision.at("attention_tiers").get<std::vector<int>>();
    if (!timeout_ms || vision.at("projection_rows") != block || vision.value("key_planes", 1) != (split_keys ? 2 : 1) ||
        vision.value("value_planes", 1) != (split_values ? 2 : 1) ||
        tiers.empty() || tiers.front() != block ||
        tiers.back() != ((max_rows + block - 1) / block) * block ||
        !std::is_sorted(tiers.begin(), tiers.end()) || std::adjacent_find(tiers.begin(), tiers.end()) != tiers.end() ||
        !std::all_of(tiers.begin(), tiers.end(), [](int n) { return n > 0 && n % block == 0; }))
        fail("unsupported vision stream geometry");
    const auto& direct = info.at("gguf");
    const auto& contexts = direct.at("contexts");
    const auto& kernels = direct.at("kernels");
    for (const std::string name : {"vision-gemm", "vision-attention"})
        if (contexts.at(name) != name + ".xclbin") fail("invalid context path for " + name);
    const auto check = [&](const std::string& name) {
        if (kernels.at(name).at("insts") != name + ".bin") fail("invalid stream path for " + name);
    };
    for (const auto& shape : projections) check(NpuVision::gemm_name(shape));
    for (const int tier : tiers) check("vision-attention-" + std::to_string(tier));
    return std::make_unique<NpuVision>(root, tiers, split_keys, split_values, timeout_ms);
}

}  // namespace open_qwen36::vision
