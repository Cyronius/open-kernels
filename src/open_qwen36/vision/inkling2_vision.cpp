#include "open_qwen36/vision/inkling2_vision.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "open_qwen36/gguf_file.hpp"
#include "open_qwen36/inkling2_host.hpp"
#include "open_qwen36/q4nx_file.hpp"
#include "open_qwen36/vision/vit.hpp"
#include "open_qwen36/vision/vit_ops.hpp"

namespace open_qwen36::vision {
namespace {

constexpr int hidden = 1024, heads = 16, head_dim = hidden / heads, ff = 4096, depth = 23;
constexpr int patch = 20, merge = 2, align = patch * merge, side = 32, channels = 3;
constexpr int patch_values = channels * patch * patch, merged = hidden * merge * merge;
constexpr int output_width = Inkling2Host::embedding_width;
using Type = GgufFile::Type;

[[noreturn]] void fail(const std::string& reason) {
    throw std::runtime_error("inkling2 vision: " + reason);
}

const uint8_t* tensor(const GgufFile& file, const std::string& name, Type type, const std::vector<uint64_t>& shape) {
    const auto& info = file.tensor(name);
    if (info.type != type || info.dims != shape) fail("unexpected tensor type or shape for " + name);
    return file.raw(name);
}

std::vector<float> f32(const GgufFile& file, const std::string& name, const std::vector<uint64_t>& shape) {
    tensor(file, name, Type::F32, shape);
    return file.f32(name);
}

Linear linear(const GgufFile& file, const std::string& name, uint64_t in, uint64_t out, bool bias = true) {
    const auto* weights = tensor(file, name + ".weight", Type::BF16, {in, out});
    Linear value;
    value.in = static_cast<int>(in);
    value.out = static_cast<int>(out);
    value.w.resize(in * out);
    std::memcpy(value.w.data(), weights, value.w.size() * sizeof(uint16_t));
    value.b = bias ? f32(file, name + ".bias", {out}) : std::vector<float>(out);
    return value;
}

Linear qkv_linear(const GgufFile& file, const std::string& name) {
    Linear result;
    result.in = hidden;
    result.out = 3 * hidden;
    result.w.reserve(static_cast<size_t>(result.in) * result.out);
    result.b.reserve(result.out);
    for (const char* leg : {"attn_q", "attn_k", "attn_v"}) {
        const auto key = name + leg;
        const auto* weights = tensor(file, key + ".weight", Type::BF16, {hidden, hidden});
        const auto offset = result.w.size();
        result.w.resize(offset + hidden * hidden);
        std::memcpy(result.w.data() + offset, weights, hidden * hidden * sizeof(uint16_t));
        const auto bias = f32(file, key + ".bias", {hidden});
        result.b.insert(result.b.end(), bias.begin(), bias.end());
    }
    return result;
}

struct Norm {
    std::vector<float> weight, bias;

    Norm() = default;
    Norm(const GgufFile& file, const std::string& name)
        : weight(f32(file, name + ".weight", {hidden})), bias(f32(file, name + ".bias", {hidden})) {}

    void apply(const std::vector<float>& input, int rows, float eps, std::vector<float>& output) const {
        ops::normalize(input.data(), rows, hidden, weight.data(), bias.data(), eps, output.data());
    }
};

struct Layer {
    Norm attention_norm, ffn_norm;
    Linear qkv, attention_out, up, down;
    std::vector<float> attention_scale, ffn_scale;

    Layer(const GgufFile& file, const std::string& name)
        : attention_norm(file, name + "ln1"), ffn_norm(file, name + "ln2"),
          qkv(qkv_linear(file, name)),
          attention_out(linear(file, name + "attn_out", hidden, hidden)),
          up(linear(file, name + "ffn_up", hidden, ff)), down(linear(file, name + "ffn_down", ff, hidden)),
          attention_scale(f32(file, name + "ls1.weight", {hidden})), ffn_scale(f32(file, name + "ls2.weight", {hidden})) {}
};

struct GridPoint {
    int low, high;
    float fraction;
};

GridPoint interpolate(int position, int grid_side) {
    const float scale = static_cast<float>(grid_side) / side;
    const float source = (position + 0.5f) / scale - 0.5f;
    const int base = static_cast<int>(std::floor(source));
    const int low = std::clamp(base, 0, side - 1);
    return {low, std::clamp(base + 1, 0, side - 1), std::clamp(source - low, 0.0f, 1.0f)};
}

void rotary_tables(int grid_w, int grid_h, std::vector<float>& cs, std::vector<float>& sn) {
    constexpr int axis_dim = head_dim / 2;
    const int rows = grid_w * grid_h + 1;
    cs.resize(static_cast<size_t>(rows) * head_dim, 1.0f);
    sn.resize(cs.size(), 0.0f);
    std::array<float, axis_dim / 2> frequency;
    for (int i = 0; i < axis_dim / 2; ++i) frequency[i] = std::pow(10000.0f, -2.0f * i / axis_dim);
    for (int row = 1; row < rows; ++row) {
        const int xy[] = {(row - 1) % grid_w + 1, (row - 1) / grid_w + 1};
        for (int axis = 0; axis < 2; ++axis)
            for (int i = 0; i < axis_dim / 2; ++i) {
                const int lane = row * head_dim + axis * axis_dim + 2 * i;
                const float angle = xy[axis] * frequency[i];
                cs[lane] = cs[lane + 1] = std::cos(angle);
                sn[lane] = sn[lane + 1] = std::sin(angle);
            }
    }
}

void add_scaled(std::vector<float>& residual, const std::vector<float>& update, const std::vector<float>& scale) {
#pragma omp parallel for schedule(static)
    for (int row = 0; row < static_cast<int>(residual.size() / hidden); ++row)
        for (int d = 0; d < hidden; ++d) residual[row * hidden + d] += update[row * hidden + d] * scale[d];
}

// The BF16 GGUF projections round their inputs to BF16 while accumulating in FP32.
void round_projection_input(std::vector<float>& values) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(values.size()); ++i)
        values[i] = bf16_to_f32(f32_to_bf16(values[i]));
}

}  // namespace

struct Inkling2Vision::Impl {
    float eps;
    std::vector<float> patch_weight, cls, pos;
    Norm pre_norm;
    std::vector<Layer> layers;
    Linear projector_up, projector_down;
    std::unique_ptr<Inkling2VisionBackend> backend;
    mutable std::mutex encode_lock;

    Impl(const std::string& path, std::unique_ptr<Inkling2VisionBackend> implementation)
        : backend(std::move(implementation)) {
        const GgufFile file(path);
        if (file.kv_str("general.architecture") != "clip" || file.kv_str("clip.vision.projector_type") != "inkling2v")
            fail("expected an inkling2v mmproj GGUF");
        const std::pair<const char*, uint64_t> fixed[] = {
            {"patch_size", patch}, {"spatial_merge_size", merge}, {"embedding_length", hidden},
            {"feed_forward_length", ff}, {"attention.head_count", heads}, {"block_count", depth},
            {"projection_dim", output_width},
        };
        for (const auto& [name, value] : fixed) {
            const std::string key = std::string("clip.vision.") + name;
            if (file.kv_u64(key) != value) fail("unsupported " + key);
        }
        eps = static_cast<float>(file.kv_float("clip.vision.attention.layer_norm_epsilon"));
        if (!std::isfinite(eps) || eps <= 0) fail("invalid normalization epsilon");
        patch_weight = f32(file, "v.patch_embd.weight", {patch, patch, channels, hidden});
        // The reference's still-image convolution uses an FP16 im2col and FP16 dot inputs.
        for (float& value : patch_weight) value = fp16_to_f32(f32_to_fp16(value));
        cls = f32(file, "v.class_embd", {hidden});
        const auto cls_pos = f32(file, "v.class_position_embd.weight", {hidden});
        for (int d = 0; d < hidden; ++d) cls[d] += cls_pos[d];
        pos = f32(file, "v.position_embd.weight", {hidden, side * side});
        pre_norm = Norm(file, "v.pre_ln");
        layers.reserve(depth);
        for (int i = 0; i < depth; ++i) layers.emplace_back(file, "v.blk." + std::to_string(i) + ".");
        projector_up = linear(file, "mm.1", merged, merged, false);
        projector_down = linear(file, "mm.2", merged, output_width, false);
    }

    void embed_patches(const std::vector<uint8_t>& rgb, int width, int height, int grid_w, int grid_h,
                       std::vector<float>& out) const {
        std::copy(cls.begin(), cls.end(), out.begin());
#pragma omp parallel for schedule(static)
        for (int t = 0; t < grid_w * grid_h; ++t) {
            const int px = t % grid_w, py = t / grid_w;
            std::array<float, patch_values> pixels;
            for (int c = 0; c < channels; ++c)
                for (int y = 0; y < patch; ++y)
                    for (int x = 0; x < patch; ++x) {
                        const int sx = px * patch + x, sy = py * patch + y;
                        const float pixel = sx < width && sy < height ? rgb[(sy * width + sx) * channels + c] : -1.0f;
                        const float normalized = (pixel - 127.5f) / 127.5f;
                        pixels[(c * patch + y) * patch + x] = fp16_to_f32(f32_to_fp16(normalized));
                    }
            const GridPoint pw = interpolate(px, grid_w), ph = interpolate(py, grid_h);
            const float dx = pw.fraction, dy = ph.fraction;
            const float* a = pos.data() + (ph.low * side + pw.low) * hidden;
            const float* b = pos.data() + (ph.low * side + pw.high) * hidden;
            const float* c = pos.data() + (ph.high * side + pw.low) * hidden;
            const float* d = pos.data() + (ph.high * side + pw.high) * hidden;
            float* row = out.data() + (t + 1) * hidden;
            for (int j = 0; j < hidden; ++j) {
                const float learned = a[j] * (1 - dx) * (1 - dy) + b[j] * dx * (1 - dy) +
                                      c[j] * (1 - dx) * dy + d[j] * dx * dy;
                row[j] = ops::dot(pixels.data(), patch_weight.data() + j * patch_values, patch_values) + learned;
            }
        }
    }

    void project(std::vector<float>& input, int rows, const Linear& weight, std::vector<float>& output) const {
        if (backend) {
            backend->project(input.data(), rows, weight, output.data());
        } else {
            round_projection_input(input);
            ops::project(input.data(), rows, weight, output.data());
        }
    }

    std::vector<float> encode(const std::vector<uint8_t>& rgb, int width, int height) const {
        if (width < align || height < align || width > 4096 || height > 4096) fail("unsupported image dimensions");
        if (rgb.size() != static_cast<size_t>(width) * height * channels) fail("invalid RGB byte count");
        std::lock_guard<std::mutex> locked(encode_lock);
        const int grid_w = (width + align - 1) / align * merge, grid_h = (height + align - 1) / align * merge;
        const int rows = 1 + grid_w * grid_h;
        std::vector<float> residual(static_cast<size_t>(rows) * hidden), cs, sn;
        embed_patches(rgb, width, height, grid_w, grid_h, residual);
        pre_norm.apply(residual, rows, eps, residual);
        rotary_tables(grid_w, grid_h, cs, sn);
        VitConfig transformer;
        transformer.hidden = hidden;
        transformer.heads = heads;
        transformer.head_dim = head_dim;
        {
            std::vector<float> normed(residual.size()), update(residual.size());
            std::vector<float> qkv(residual.size() * 3), mid(static_cast<size_t>(rows) * ff);
            for (const auto& layer : layers) {
                layer.attention_norm.apply(residual, rows, eps, normed);
                project(normed, rows, layer.qkv, qkv);
                if (backend) backend->attend(qkv.data(), rows, cs, sn, normed.data());
                else ops::full_attention(transformer, qkv.data(), rows, cs, sn, normed.data());
                project(normed, rows, layer.attention_out, update);
                add_scaled(residual, update, layer.attention_scale);
                layer.ffn_norm.apply(residual, rows, eps, normed);
                project(normed, rows, layer.up, mid);
                ops::gelu(mid);
                project(mid, rows, layer.down, update);
                add_scaled(residual, update, layer.ffn_scale);
            }
        }

        const int merged_w = grid_w / merge, merged_rows = merged_w * (grid_h / merge);
        std::vector<float> packed(static_cast<size_t>(merged_rows) * merged);
#pragma omp parallel for schedule(static)
        for (int t = 0; t < merged_rows; ++t) {
            const int x = (t % merged_w) * merge, y = (t / merged_w) * merge;
            const float* top = residual.data() + (1 + y * grid_w + x) * hidden;
            float* row = packed.data() + t * merged;
            std::copy_n(top, merge * hidden, row);
            std::copy_n(top + grid_w * hidden, merge * hidden, row + merge * hidden);
        }
        std::vector<float> mid(packed.size()), output(static_cast<size_t>(merged_rows) * output_width);
        project(packed, merged_rows, projector_up, mid);
        ops::gelu(mid);
        project(mid, merged_rows, projector_down, output);
        return output;
    }
};

Inkling2Vision::Inkling2Vision(const std::string& path, std::unique_ptr<Inkling2VisionBackend> backend)
    : impl_(std::make_unique<Impl>(path, std::move(backend))) {}
Inkling2Vision::~Inkling2Vision() = default;

std::vector<float> Inkling2Vision::encode(const std::vector<uint8_t>& rgb, int width, int height) const {
    return impl_->encode(rgb, width, height);
}

}  // namespace open_qwen36::vision
