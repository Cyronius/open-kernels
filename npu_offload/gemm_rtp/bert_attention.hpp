// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "bert_attention_geometry.hpp"
#include "../../open_kernels/harness/stream_patch.hpp"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_xclbin.h"

namespace npue::npu {

// The caller supplies the checked xclbin and instruction bytes. This
// adapter consumes the NPUE projection where Q already carries the attention scale.
// Use one instance per serialized NPU route.
class BertAttention {
public:
    using Narrow = void (*)(void*, const float*, size_t);
    using Widen = void (*)(float*, const void*, size_t);

private:
    static constexpr int64_t max_head_dim = 64;
    static constexpr uint32_t marker = 381;

    xrt::device device{0u};
    xrt::hw_context context;
    xrt::kernel kernel;
    xrt::bo instructions;
    std::array<xrt::bo, 4> tensor;
    std::vector<size_t> patches;
    const int64_t capacity, head_dim, width;
    const float undo_q_scale;
    Narrow narrow;
    Widen widen;
    uint32_t instruction_count = 0;
    uint32_t current_length = marker;

    static double now_s() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

public:
    double io_ms = 0, dispatch_ms = 0;
    BertAttention(const std::vector<char>& image_bytes, const std::vector<char>& bytes,
                  int64_t sequence, int64_t heads, int64_t head_width,
                  Narrow narrow_bf16, Widen widen_bf16)
        : capacity(sequence), head_dim(head_width),
          width(bert_attention_geometry(sequence, heads, head_width) ? heads * head_width : 0),
          undo_q_scale(std::sqrt(static_cast<float>(head_width))),
          narrow(narrow_bf16), widen(widen_bf16) {
        if (!width || !narrow || !widen)
            throw std::invalid_argument("unsupported BERT attention geometry or conversion");
        if (bytes.empty() || bytes.size() % sizeof(uint32_t) ||
                bytes.size() / sizeof(uint32_t) > static_cast<size_t>((std::numeric_limits<int>::max)()))
            throw std::runtime_error("invalid BERT attention stream");
        std::vector<uint32_t> words(bytes.size() / sizeof(uint32_t));
        std::memcpy(words.data(), bytes.data(), bytes.size());
        patches = stream_patch::bert_length_table(words, marker);
        instruction_count = static_cast<uint32_t>(words.size());
        auto image = xrt::xclbin(image_bytes);
        context = xrt::hw_context(device, device.register_xclbin(image));
        kernel = xrt::kernel(context, "MLIR_AIE");
        instructions = xrt::bo(device, bytes.size(), xrt::bo::flags::cacheable, kernel.group_id(1));
        std::memcpy(instructions.map<void*>(), bytes.data(), bytes.size());
        instructions.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        const auto data_bytes = static_cast<size_t>(capacity * width * sizeof(uint16_t));
        constexpr size_t alignment = 1u << 20;
        const auto rounded_bytes = (data_bytes + alignment - 1) & ~(alignment - 1);
        for (unsigned i = 0; i < tensor.size(); ++i) {
            tensor[i] = xrt::bo(device, rounded_bytes, xrt::bo::flags::host_only, kernel.group_id(i + 3));
            std::memset(tensor[i].map<void*>(), 0, rounded_bytes);
        }
    }

    void operator()(const std::vector<float>& qkv, const std::vector<float>& mask,
                    std::vector<float>& output, int64_t batch) {
        if (batch < 0 || static_cast<uint64_t>(batch) >
                (std::numeric_limits<size_t>::max)() / (capacity * width * 3))
            throw std::invalid_argument("invalid BERT attention batch");
        const size_t rows = static_cast<size_t>(batch) * capacity;
        if (mask.size() != rows || output.size() != rows * width || qkv.size() != rows * width * 3)
            throw std::invalid_argument("BERT attention input sizes disagree with its geometry");
        std::fill(output.begin(), output.end(), 0.0f);
        for (int64_t b = 0; b < batch; ++b) {
            const auto start = now_s();
            const auto begin = mask.begin() + b * capacity;
            const uint32_t length = static_cast<uint32_t>(std::count(begin, begin + capacity, 0.0f));
            if (!length) continue;
            if (std::find_if(begin, begin + length, [](float value) { return value != 0; }) != begin + length)
                throw std::runtime_error("BERT attention expects right padding");
            if (length != current_length) {
                auto* words = instructions.map<uint32_t*>();
                for (const auto patch : patches) words[patch] = length;
                instructions.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                current_length = length;
            }
            std::array<uint16_t*, 3> mapped;
            for (size_t i = 0; i < mapped.size(); ++i)
                mapped[i] = tensor[i].map<uint16_t*>();
            for (int64_t h = 0; h < width / head_dim; ++h) {
                for (int64_t t = 0; t < length; ++t) {
                    const auto* src = qkv.data() + ((b * capacity + t) * 3 * width + h * head_dim);
                    const auto to = (h * capacity + t) * head_dim;
                    float query[max_head_dim];
                    std::transform(src, src + head_dim, query, [this](float v) { return v * undo_q_scale; });
                    narrow(mapped[0] + to, query, head_dim);
                    narrow(mapped[1] + to, src + width, head_dim);
                    narrow(mapped[2] + to, src + 2 * width, head_dim);
                }
            }
            for (unsigned i = 0; i < mapped.size(); ++i)
                tensor[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
            const auto launch = now_s();
            xrt::run run(kernel);
            run.set_arg(0, 3);
            run.set_arg(1, instructions);
            run.set_arg(2, static_cast<int>(instruction_count));
            for (unsigned i = 0; i < tensor.size(); ++i) run.set_arg(i + 3, tensor[i]);
            run.start();
            if (run.wait(std::chrono::milliseconds(180000)) != ERT_CMD_STATE_COMPLETED)
                throw std::runtime_error("BERT attention dispatch failed");
            const auto finished = now_s();
            tensor.back().sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            const auto* result = tensor.back().map<uint16_t*>();
            for (int64_t h = 0; h < width / head_dim; ++h)
                for (int64_t t = 0; t < length; ++t)
                    widen(output.data() + (b * capacity + t) * width + h * head_dim,
                                         result + (h * capacity + t) * head_dim, head_dim);
            dispatch_ms += (finished - launch) * 1000;
            io_ms += (launch - start + now_s() - finished) * 1000;
        }
    }
};

}  // namespace npue::npu
