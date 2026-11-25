#pragma once

#include "open_qwen36/gguf_file.hpp"
#include "open_qwen36/pools.hpp"
#include "stream_patch.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace open_qwen36::shared_dense {

namespace block = pools::inkling2_block;
inline constexpr size_t columns = block::array_cols, workers = 4, tile = block::tile_n;
inline constexpr size_t activations = block::decode_tile_m, outputs_per_band = columns * workers * tile;

struct Shape {
    size_t inputs, bands;
    size_t outputs() const { return outputs_per_band * bands; }
    size_t weight_bytes() const { return block::weight_bytes(false, outputs(), inputs, true); }
    size_t input_bytes() const { return columns * block::input_bytes(activations, inputs); }
    size_t output_bytes() const { return activations * outputs() * sizeof(float); }
};

struct Part {
    const char* tensor;
    size_t outputs, input_row;
    float* destination = nullptr;
};

inline constexpr Shape fused{2304, 3};
inline constexpr Shape output{3072, 1};

[[noreturn]] inline void fail(const std::string& message) {
    throw std::runtime_error("shared dense Inkling 2: " + message);
}

inline void check_stream(const std::vector<uint32_t>& words, const std::array<size_t, 3>& spans) {
    std::array<size_t, 3> covered{};
    size_t weight_transfers = 0;
    size_t bd = 0;
    for (size_t at = 4; at < words.size();) {
        const size_t op_size = stream_patch::op_len(words[at]);
        if (op_size > words.size() - at) fail("truncated instruction stream");
        if (words[at] == 1) {
            if (op_size < 5) fail("truncated DMA descriptor");
            bd = at;
        }
        if (words[at] == 0x81) {
            if (op_size < 12 || !bd || words[bd + 2] + 4 != words[at + 6])
                fail("missing matching DMA descriptor");
            const size_t arg = words[at + 8], offset = words[at + 10];
            const size_t size = static_cast<size_t>(words[bd + 4]) * sizeof(uint32_t);
            if (arg >= spans.size() || spans[arg] % columns || words[at + 11] ||
                offset != covered[arg] || size != spans[arg] / columns || size > spans[arg] - covered[arg])
                fail("DMA span differs from the shared stream geometry");
            covered[arg] += size;
            if (arg == 0) ++weight_transfers;
        }
        at += op_size;
    }
    if (covered != spans || weight_transfers != columns) fail("incomplete shared stream DMA coverage");
}

inline void pack_dense(const GgufFile& file, size_t layer, Shape shape, std::initializer_list<Part> parts,
                       uint8_t* destination) {
    const size_t row_bytes = block::weight_bytes(false, 1, shape.inputs);
    std::vector<uint8_t> reordered(shape.outputs() * row_bytes);
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    size_t at = 0;
    for (const auto& part : parts) {
        const std::string name = prefix + part.tensor + ".weight";
        size_t source_bytes = 0;
        const auto* source = file.raw(name, &source_bytes);
        const auto& weight = file.tensor(name);
        if (weight.type != GgufFile::Type::Q8_0 ||
            weight.dims != std::vector<uint64_t>{shape.inputs, part.outputs} ||
            source_bytes != part.outputs * row_bytes || part.outputs > shape.outputs() - at)
            fail("unexpected dense tensor " + name);

        // Place each output tile in worker-major order before using the existing GGUF packer.
        for (size_t row = 0; row < part.outputs; ++row) {
            const size_t output_row = at + row, group = output_row / tile;
            const size_t col = group % columns, worker = group / columns % workers, band = group / (columns * workers);
            const size_t target_tile = (worker * shape.bands + band) * columns + col;
            const size_t target_row = target_tile * tile + output_row % tile;
            std::memcpy(reordered.data() + target_row * row_bytes, source + row * row_bytes, row_bytes);
        }
        at += part.outputs;
    }
    block::pack_weights(reordered.data(), false, shape.outputs(), shape.inputs, destination, true);
}

inline void pack_inputs(Shape shape, const std::array<const float*, 3>& inputs,
                        std::vector<float>& scratch, uint8_t* destination) {
    const bool shared = inputs[0] == inputs[1] && inputs[0] == inputs[2];
    const float* values = inputs[0];
    const size_t rows = shared ? 1 : inputs.size();
    if (!shared) {
        scratch.resize(rows * shape.inputs);
        for (size_t row = 0; row < rows; ++row)
            std::copy_n(inputs[row], shape.inputs, scratch.data() + row * shape.inputs);
        values = scratch.data();
    }
    block::pack_input(values, rows, activations, shape.inputs, destination, true);
    const size_t bytes = shape.input_bytes() / columns;
    for (size_t col = 1; col < columns; ++col) std::memcpy(destination + col * bytes, destination, bytes);
}

inline void unpack_parts(Shape shape, const float* source, bool separate, std::initializer_list<Part> parts) {
    constexpr size_t lanes = 8, frame = activations * tile;
    size_t at = 0;
    for (const auto& part : parts) {
        if (part.outputs % lanes || part.outputs > shape.outputs() - at || part.input_row >= activations)
            fail("bad dense output part");
        const size_t activation = separate ? part.input_row : 0;
        for (size_t row = 0; row < part.outputs; row += lanes) {
            const size_t output_row = at + row, group = output_row / tile;
            const size_t col = group % columns, worker = group / columns % workers, band = group / (columns * workers);
            const size_t wire_frame = (col * shape.bands + band) * workers + worker;
            const size_t offset = wire_frame * frame + (output_row % tile / lanes * activations + activation) * lanes;
            std::copy_n(source + offset, lanes, part.destination + row);
        }
        at += part.outputs;
    }
}
} // namespace open_qwen36::shared_dense
