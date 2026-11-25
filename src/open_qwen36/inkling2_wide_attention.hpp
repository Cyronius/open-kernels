#pragma once

#include "open_qwen36/bfp16.hpp"
#include "open_qwen36/q4nx_file.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace inkling_wide_bfp {

inline constexpr size_t heads = 8, width = 128, tile = 32;
inline constexpr size_t bytes = open_qwen36::BFP16_BLOCK_BYTES;
inline constexpr size_t frame_bytes = tile * width / 8 * bytes * 2;

inline void encode_values(const float* x, uint8_t* out_hi, uint8_t* out_lo) {
    alignas(32) float hi[8], residual[8];
    open_qwen36::bfp16_encode_block(x, out_hi);
    open_qwen36::bfp16_decode_block(out_hi, hi);
#ifdef __AVX2__
    const __m256 r = _mm256_sub_ps(_mm256_loadu_ps(x), _mm256_load_ps(hi));
    const __m256i bits = _mm256_castps_si256(r);
    const __m256i parity = _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
    const __m256i rounded = _mm256_add_epi32(bits, _mm256_add_epi32(_mm256_set1_epi32(0x7fff), parity));
    _mm256_store_ps(residual, _mm256_castsi256_ps(_mm256_and_si256(rounded, _mm256_set1_epi32(-65536))));
#else
    for (size_t i = 0; i < 8; ++i)
        residual[i] = open_qwen36::bf16_to_f32(open_qwen36::f32_to_bf16(x[i] - hi[i]));
#endif
    open_qwen36::bfp16_encode_block(residual, out_lo);
}

inline void encode(const uint16_t* input, uint8_t* out_hi, uint8_t* out_lo) {
    alignas(32) float values[8];
#ifdef __AVX2__
    const __m128i bf16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input));
    _mm256_store_ps(values, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16), 16)));
#else
    for (size_t i = 0; i < 8; ++i) values[i] = open_qwen36::bf16_to_f32(input[i]);
#endif
    encode_values(values, out_hi, out_lo);
}

inline void pack(const uint16_t* source, uint8_t* dest, size_t count, int threads) {
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int frame = 0; frame < static_cast<int>(heads * count / tile); ++frame) {
        const size_t head = frame / (count / tile), first = frame % (count / tile) * tile;
        auto* frame_out = dest + frame * frame_bytes;
        for (size_t row = 0; row < tile; row += 8) {
            for (size_t col = 0; col < width; col += 8) {
                for (size_t lane = 0; lane < 8; ++lane) {
                    const size_t panel = (row / 8 * (width / 8) + col / 8) * 16 * bytes;
                    auto* hi = frame_out + panel + lane * bytes;
                    const auto* key = source + ((first + row + lane) * heads + head) * width + col;
                    encode(key, hi, hi + 8 * bytes);
                }
            }
        }
    }
}

class KeyCache {
public:
    static constexpr size_t visit_count = 48, segment = 4096;
    static constexpr size_t frames_per_head = segment / tile, head_bytes = frames_per_head * frame_bytes;
    using Retire = bool (*)(const void*, size_t, size_t) noexcept;

    void clear(uint64_t sequence) noexcept {
        if (resident_) {
            for (auto& pieces : pieces_) pieces.clear();
            resident_ = false;
        }
        retired_ = false;
        sequence_ = sequence;
    }

    void prepare(uint64_t sequence, size_t visit, const uint16_t* committed, size_t committed_position,
                 size_t end, int threads, const void* owner = nullptr, Retire retire = nullptr) {
        if (end % segment || end > committed_position || (end && !committed))
            throw std::invalid_argument("wide attention key range is outside the committed prefix");
        auto& target = pieces_.at(visit);
        if (sequence != sequence_) clear(sequence);
        while (target.size() < end / segment) {
            const size_t start = target.size() * segment;
            const auto* raw = committed + start * heads * width;
            Piece piece;
            piece.wire.resize(heads * head_bytes);
            pack(raw, piece.wire.data(), segment, threads);
            if (retire) record_corrections(piece, raw);
            // Publish the complete exact representation before decommitting any original pages.
            target.push_back(std::move(piece));
            if (retire && retire(owner, start, segment)) {
                target.back().retired = true;
                retired_ = true;
            } else {
                std::vector<uint32_t>().swap(target.back().offsets);
                std::vector<Correction>().swap(target.back().corrections);
            }
            resident_ = true;
        }
    }

    bool has_retired() const noexcept { return retired_; }

    // Keys below this position have retired raw pages for this visit.
    size_t retired_end(size_t visit) const {
        const auto& pieces = pieces_.at(visit);
        for (size_t i = pieces.size(); i > 0; --i)
            if (pieces[i - 1].retired) return i * segment;
        return 0;
    }

    bool restore_head_frame(size_t visit, size_t head, size_t first, uint16_t* dst) const {
        const auto& pieces = pieces_.at(visit);
        const auto index = first / segment;
        if (index >= pieces.size() || !pieces[index].retired) return false;
        if (head >= heads || first % tile) throw std::invalid_argument("unaligned archived key frame");
        const auto& piece = pieces[index];
        const size_t frame = head * frames_per_head + first % segment / tile;
        decode_frame(piece.wire.data() + frame * frame_bytes, dst);
        for (size_t i = piece.offsets[frame]; i < piece.offsets[frame + 1]; ++i)
            dst[piece.corrections[i].index] = piece.corrections[i].bits;
        return true;
    }

    void copy_head(uint64_t sequence, size_t visit, size_t head, size_t first, size_t count, uint8_t* out) const {
        const auto& pieces = pieces_.at(visit);
        if (sequence != sequence_ || head >= heads || first % segment || count % segment ||
            first / segment > pieces.size() || count / segment > pieces.size() - first / segment)
            throw std::invalid_argument("wide attention packed key range is unavailable");
        for (size_t at = first / segment; at < (first + count) / segment; ++at) {
            std::memcpy(out, pieces[at].wire.data() + head * head_bytes, head_bytes);
            out += head_bytes;
        }
    }

private:
    struct Correction { uint16_t index, bits; };
    struct Piece {
        std::vector<uint8_t> wire;
        std::vector<uint32_t> offsets;
        std::vector<Correction> corrections;
        bool retired = false;
    };

    static void decode_pair(const uint8_t* hi, const uint8_t* lo, uint16_t* dst) {
#ifdef __AVX2__
        if (hi[0] >= 7 && lo[0] >= 7) {
            const auto values = [](const uint8_t* block) {
                const auto q = _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(block + 1)));
                const auto scale = _mm256_castsi256_ps(_mm256_set1_epi32((block[0] - 6) << 23));
                return _mm256_mul_ps(_mm256_cvtepi32_ps(q), scale);
            };
            const auto bits = _mm256_castps_si256(_mm256_add_ps(values(hi), values(lo)));
            const auto high = _mm256_srli_epi32(bits, 16);
            const auto parity = _mm256_and_si256(high, _mm256_set1_epi32(1));
            const auto rounded = _mm256_add_epi32(bits, _mm256_add_epi32(_mm256_set1_epi32(0x7fff), parity));
            const auto words = _mm256_srli_epi32(rounded, 16);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst),
                             _mm_packus_epi32(_mm256_castsi256_si128(words), _mm256_extracti128_si256(words, 1)));
            return;
        }
#endif
        std::array<float, 8> a, b;
        open_qwen36::bfp16_decode_block(hi, a.data());
        open_qwen36::bfp16_decode_block(lo, b.data());
        for (size_t i = 0; i < 8; ++i) dst[i] = open_qwen36::f32_to_bf16(a[i] + b[i]);
    }

    static void decode_frame(const uint8_t* wire, uint16_t* dst) {
        for (size_t row = 0; row < tile; row += 8) {
            for (size_t col = 0; col < width; col += 8) {
                const auto* panel = wire + (row / 8 * (width / 8) + col / 8) * 16 * bytes;
                for (size_t lane = 0; lane < 8; ++lane)
                    decode_pair(panel + lane * bytes, panel + (8 + lane) * bytes, dst + (row + lane) * width + col);
            }
        }
    }

    static void record_corrections(Piece& piece, const uint16_t* raw) {
        std::array<uint16_t, tile * width> restored;
        piece.offsets.reserve(heads * frames_per_head + 1);
        piece.offsets.push_back(0);
        for (size_t head = 0; head < heads; ++head) {
            for (size_t chunk = 0; chunk < frames_per_head; ++chunk) {
                const size_t frame = head * frames_per_head + chunk;
                decode_frame(piece.wire.data() + frame * frame_bytes, restored.data());
                for (size_t row = 0; row < tile; ++row) {
                    const auto* source = raw + ((chunk * tile + row) * heads + head) * width;
                    for (size_t col = 0; col < width; ++col) {
                        const auto index = row * width + col;
                        if (source[col] != restored[index])
                            piece.corrections.push_back({static_cast<uint16_t>(index), source[col]});
                    }
                }
                piece.offsets.push_back(static_cast<uint32_t>(piece.corrections.size()));
            }
        }
    }

    uint64_t sequence_ = 0;
    bool resident_ = false, retired_ = false;
    std::array<std::vector<Piece>, visit_count> pieces_;
};

} // namespace inkling_wide_bfp
