#pragma once

#include "open_qwen36/gguf_file.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>
#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace open_qwen36::inkling2_cpu {

using Type = GgufFile::Type;

[[noreturn]] inline void fail(const std::string& reason) {
    throw std::runtime_error("inkling2: " + reason);
}

struct Rows {
    size_t width = 0;
    std::vector<float> data;

    Rows() = default;
    Rows(size_t count, size_t dim) : width(dim), data(count * dim) {}
    size_t count() const { return data.size() / width; }
    float* operator[](size_t row) { return data.data() + row * width; }
    const float* operator[](size_t row) const { return data.data() + row * width; }
};

inline float dot(const float* a, const float* b, size_t n) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    size_t j = 0;
    for (; j + 4 <= n; j += 4) {
        s0 += a[j] * b[j];
        s1 += a[j + 1] * b[j + 1];
        s2 += a[j + 2] * b[j + 2];
        s3 += a[j + 3] * b[j + 3];
    }
    for (; j < n; ++j) s0 += a[j] * b[j];
    return (s0 + s1) + (s2 + s3);
}

#ifdef __AVX2__
// The four lanes preserve the scalar dot's accumulation and reduction order.
template <size_t count>
inline std::array<float, count> dot_f32_rows(const float* input, const float* weight, size_t width) {
    __m128 sums[count];
    for (auto& sum : sums) sum = _mm_setzero_ps();
    for (size_t j = 0; j < width; j += 4) {
        const auto weights = _mm_loadu_ps(weight + j);
        for (size_t row = 0; row < count; ++row) {
            const auto values = _mm_loadu_ps(input + row * width + j);
            sums[row] = _mm_add_ps(sums[row], _mm_mul_ps(values, weights));
        }
    }
    std::array<float, count> result;
    for (size_t row = 0; row < count; ++row) {
        float lanes[4];
        _mm_storeu_ps(lanes, sums[row]);
        result[row] = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    }
    return result;
}

// Eight-lane FMA dots for batched prefill products; every row of a block uses the same order.
template <size_t count>
inline std::array<float, count> dot_f32_fma_rows(const float* input, const float* weight, size_t width) {
    __m256 sums[count];
    for (auto& sum : sums) sum = _mm256_setzero_ps();
    size_t j = 0;
    for (; j + 8 <= width; j += 8) {
        const auto weights = _mm256_loadu_ps(weight + j);
        for (size_t row = 0; row < count; ++row)
            sums[row] = _mm256_fmadd_ps(_mm256_loadu_ps(input + row * width + j), weights, sums[row]);
    }
    std::array<float, count> result;
    for (size_t row = 0; row < count; ++row) {
        const __m128 half = _mm_add_ps(_mm256_castps256_ps128(sums[row]), _mm256_extractf128_ps(sums[row], 1));
        float lanes[4];
        _mm_storeu_ps(lanes, half);
        float total = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
        for (size_t k = j; k < width; ++k) total += input[row * width + k] * weight[k];
        result[row] = total;
    }
    return result;
}

constexpr size_t q8_block_values = 32, q8_block_bytes = 34;

template <size_t count>
inline std::array<float, count> dot_q8_rows(const float* input, const uint8_t* weights, size_t n) {
    // Decode each weight vector once and retain the scalar accumulation order for each row.
    __m128 sums[count];
    for (auto& sum : sums) sum = _mm_setzero_ps();
    for (size_t at = 0; at < n; at += q8_block_values, weights += q8_block_bytes) {
        uint16_t scale;
        std::memcpy(&scale, weights, sizeof(scale));
        const __m128 d = _mm_set1_ps(fp16_to_f32(scale));
        for (size_t j = 0; j < q8_block_values; j += 4) {
            int32_t packed;
            std::memcpy(&packed, weights + sizeof(scale) + j, sizeof(packed));
            const auto codes = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(packed));
            const auto v = _mm_mul_ps(d, _mm_cvtepi32_ps(codes));
            for (size_t row = 0; row < count; ++row) {
                const auto a = _mm_loadu_ps(input + row * n + at + j);
                sums[row] = _mm_add_ps(sums[row], _mm_mul_ps(a, v));
            }
        }
    }
    std::array<float, count> output;
    for (size_t row = 0; row < count; ++row) {
        float lanes[4];
        _mm_storeu_ps(lanes, sums[row]);
        output[row] = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    }
    return output;
}
#endif

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float softcap(float x, float cap) { return cap * std::tanh(x / cap); }

inline void rms(float* x, size_t n, float eps, const float* weight = nullptr) {
    const float scale = 1.0f / std::sqrt(dot(x, x, n) / static_cast<float>(n) + eps);
    if (weight)
        for (size_t j = 0; j < n; ++j) x[j] *= scale * weight[j];
    else
        for (size_t j = 0; j < n; ++j) x[j] *= scale;
}

inline void norm(Rows& rows, float eps, const std::vector<float>& weight) {
    for (size_t t = 0; t < rows.count(); ++t) rms(rows[t], rows.width, eps, weight.data());
}

inline const GgufFile::TensorInfo& checked_tensor(const GgufFile& file, const std::string& name,
                                                 const std::vector<uint64_t>& shape) {
    if (!file.has(name)) fail("missing tensor " + name);
    const auto& tensor = file.tensor(name);
    if (tensor.dims != shape) fail("unexpected shape for " + name);
    switch (tensor.type) {
        case Type::F32: case Type::F16: case Type::BF16: case Type::Q8_0:
        case Type::IQ4_NL: case Type::IQ4_XS: case Type::NVFP4: break;
        default: fail("unsupported tensor type for " + name + ": " + GgufFile::type_name(tensor.type));
    }
    if ((tensor.type == Type::Q8_0 || tensor.type == Type::IQ4_NL) && shape.front() % 32)
        fail("misaligned quantized rows in " + name);
    if (tensor.type == Type::IQ4_XS && shape.front() % 256)
        fail("misaligned IQ4_XS rows in " + name);
    return tensor;
}

inline std::vector<float> load_small(const GgufFile& file, const std::string& name,
                                     const std::vector<uint64_t>& shape) {
    checked_tensor(file, name, shape);
    return file.f32(name);
}

// The host reads quantized rows directly. Dequantize each row once per input
// block, including when several tokens select the same expert.
struct Projection {
    const GgufFile* file = nullptr;
    std::string name;
    size_t inputs = 0, outputs = 0;
    const float* f32 = nullptr;
    std::vector<float> expert_scales;
#ifdef __AVX2__
    const uint8_t* q8 = nullptr;
#endif

    Projection() = default;
    Projection(const GgufFile& data, std::string tensor, size_t cols, size_t rows, size_t planes = 0)
        : file(&data), name(std::move(tensor)), inputs(cols), outputs(rows) {
        std::vector<uint64_t> shape{cols, rows};
        if (planes) shape.push_back(planes);
        const auto type = checked_tensor(data, name, shape).type;
        if (type == Type::F32) f32 = reinterpret_cast<const float*>(data.raw(name));
        if (type == Type::NVFP4) {
            constexpr std::string_view suffix = ".weight";
            if (!planes || name.size() < suffix.size() ||
                name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0)
                fail("expected an NVFP4 expert projection for " + name);
            const std::string scale_name = name.substr(0, name.size() - suffix.size()) + ".scale";
            if (checked_tensor(data, scale_name, {planes}).type != Type::F32)
                fail("expected F32 expert scales in " + scale_name);
            expert_scales = data.f32(scale_name);
            if (!std::all_of(expert_scales.begin(), expert_scales.end(),
                             [](float scale) { return scale > 0 && std::isfinite(scale); }))
                fail("invalid expert scale in " + scale_name);
        }
#ifdef __AVX2__
        if (type == Type::Q8_0) q8 = data.raw(name);
#endif
    }

    Rows apply(const Rows& x, size_t plane = 0, size_t rows = 0) const {
        if (!rows) rows = outputs;
        const size_t count = x.count();
        Rows y(count, rows);
#ifdef __AVX2__
        // Small drafts and verify blocks benefit from reading packed Q8_0 directly.
        // Larger blocks reuse one dequantized weight row through the general path.
        if (q8 && count <= 8 && file->fold(name) == 1.0f) {
            const size_t stride = inputs / q8_block_values * q8_block_bytes;
            const auto* weights = q8 + plane * outputs * stride;
#pragma omp parallel for if(rows >= 512) schedule(static)
            for (long long r = 0; r < static_cast<long long>(rows); ++r) {
                const auto* weight = weights + static_cast<size_t>(r) * stride;
                if (count == 8) {
                    const auto sums = dot_q8_rows<8>(x[0], weight, inputs);
                    for (size_t row = 0; row < sums.size(); ++row) y[row][r] = sums[row];
                    continue;
                }
                constexpr size_t tile = 4;
                size_t t = 0;
                for (; t + tile <= count; t += tile) {
                    const auto sums = dot_q8_rows<tile>(x[t], weight, inputs);
                    for (size_t row = 0; row < tile; ++row) y[t + row][r] = sums[row];
                }
                for (; t < count; ++t) y[t][r] = dot_q8_rows<1>(x[t], weight, inputs)[0];
            }
            return y;
        }
#endif
#ifdef __AVX2__
        // A small F32 matrix in a prefill batch walks token tiles outside so tiles and output rows stay in cache.
        if (f32 && inputs % 4 == 0 && rows * inputs <= 512 * 1024 && count >= 64) {
            constexpr size_t tile = 4;
            const long long tiles = static_cast<long long>(count / tile);
#pragma omp parallel for schedule(static)
            for (long long i = 0; i < tiles; ++i) {
                const size_t t = static_cast<size_t>(i) * tile;
                for (size_t r = 0; r < rows; ++r) {
                    const auto values = dot_f32_fma_rows<tile>(x[t], f32 + (plane * outputs + r) * inputs, inputs);
                    for (size_t row = 0; row < tile; ++row) y[t + row][r] = values[row];
                }
            }
            for (size_t t = count / tile * tile; t < count; ++t)
                for (size_t r = 0; r < rows; ++r)
                    y[t][r] = dot_f32_fma_rows<1>(x[t], f32 + (plane * outputs + r) * inputs, inputs)[0];
            if (!expert_scales.empty())
                for (auto& value : y.data) value *= expert_scales[plane];
            return y;
        }
#endif
        // A narrow F32 controller or router can still be a large batched product.
        // Waking a thread team costs more than a small single-row F32 product.
        const bool threaded = f32 ? rows * inputs * count >= 2 * 1024 * 1024 : rows >= 512;
#pragma omp parallel if(threaded)
        {
            std::array<float, 2 * 2304> stack;
            std::vector<float> large;
            if (!f32 && inputs > stack.size()) large.resize(inputs);
            float* scratch = large.empty() ? stack.data() : large.data();
#pragma omp for
            for (long long r = 0; r < static_cast<long long>(rows); ++r) {
                const auto index = plane * outputs + static_cast<size_t>(r);
                const float* weight = f32 ? f32 + index * inputs : scratch;
                if (!f32) file->embed_row(name, index, inputs, scratch);
#ifdef __AVX2__
                if (f32 && inputs % 4 == 0) {
                    constexpr size_t tile = 4;
                    size_t t = 0;
                    for (; t + tile <= count; t += tile) {
                        const auto values = dot_f32_rows<tile>(x[t], weight, inputs);
                        for (size_t row = 0; row < tile; ++row) y[t + row][r] = values[row];
                    }
                    for (; t < count; ++t) y[t][r] = dot_f32_rows<1>(x[t], weight, inputs)[0];
                    continue;
                }
#endif
                for (size_t t = 0; t < count; ++t)
                    y[t][r] = dot(x[t], weight, inputs);
            }
        }
        // The gate/up scales are applied before SITU, and the down scales
        // before the corresponding expert's contribution is accumulated.
        if (!expert_scales.empty())
            for (auto& value : y.data) value *= expert_scales[plane];
        return y;
    }
};

}  // namespace open_qwen36::inkling2_cpu
