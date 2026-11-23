// SPDX-License-Identifier: MIT
#pragma once

#include "bert_attention_geometry.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#include <nlohmann/json.hpp>

namespace npue::npu {

struct BertAttentionFiles {
    int64_t sequence;
    int64_t heads;
    int64_t head_width;
    std::vector<char> image;
    std::vector<char> instructions;
};

inline std::vector<char> read_attention_payload(const std::filesystem::path& path,
                                                const nlohmann::json& expected) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const auto length = input.tellg();
    if (length <= 0 || length > (std::numeric_limits<ULONG>::max)() ||
        expected.at("bytes") != static_cast<uint64_t>(length))
        throw std::runtime_error("BERT attention payload size mismatch: " + path.string());
    std::vector<char> bytes(static_cast<size_t>(length));
    if (!input.seekg(0) || !input.read(bytes.data(), length) || input.peek() != EOF)
        throw std::runtime_error("could not read the BERT attention payload: " + path.string());
    std::array<unsigned char, 32> digest{};
    if (BCryptHash(BCRYPT_SHA256_ALG_HANDLE, nullptr, 0,
                   reinterpret_cast<PUCHAR>(bytes.data()), static_cast<ULONG>(bytes.size()),
                   digest.data(), static_cast<ULONG>(digest.size())) != 0)
        throw std::runtime_error("could not hash the BERT attention payload: " + path.string());
    constexpr char hex[] = "0123456789abcdef";
    std::string hash;
    for (auto byte : digest) {
        hash.push_back(hex[byte >> 4]);
        hash.push_back(hex[byte & 15]);
    }
    if (expected.at("sha256") != hash)
        throw std::runtime_error("BERT attention payload SHA-256 mismatch: " + path.string());
    return bytes;
}

inline BertAttentionFiles load_bert_attention(const std::filesystem::path& directory) {
    std::ifstream input(directory / "attention.json", std::ios::binary);
    if (!input)
        throw std::runtime_error("missing BERT attention manifest in " + directory.string());
    auto manifest = nlohmann::json::parse(input);
    const auto& files = manifest.at("files");
    const auto sequence = manifest.at("sequence");
    const auto heads = manifest.at("heads");
    const auto width = manifest.at("head_width");
    if (manifest.at("format") != "bert-attention-v1" ||
        !sequence.is_number_integer() || !heads.is_number_integer() || !width.is_number_integer() ||
        !bert_attention_geometry(sequence.get<int64_t>(), heads.get<int64_t>(), width.get<int64_t>()) ||
        manifest.at("length_marker") != 381 || manifest.at("dtype") != "bf16" ||
        manifest.at("layout") != "head_sequence_width" ||
        manifest.at("attention_scale") != "inverse_sqrt_head_width")
        throw std::runtime_error("unsupported BERT attention manifest: " + directory.string());

    auto image = read_attention_payload(directory / "final.xclbin", files.at("final.xclbin"));
    auto instructions = read_attention_payload(directory / "insts.bin", files.at("insts.bin"));
    if (instructions.size() % sizeof(uint32_t))
        throw std::runtime_error("BERT attention instruction stream is not word-aligned");
    return {sequence.get<int64_t>(), heads.get<int64_t>(), width.get<int64_t>(),
            std::move(image), std::move(instructions)};
}

}  // namespace npue::npu
