/// \file gguf_pack_test.cpp
/// \brief No-XRT unit test for the GGUF-direct path: a synthetic GGUF goes
///        through pools::apply's std_perm_gguf and the packed pool bytes must
///        dequantize BIT-EXACTLY to the GGUF blocks' values (the pack is a byte
///        permutation plus an exact fp16 -> f32 widening of the scales;
///        open_kernels/gguf_pool.py holds the same law in NumPy).
///
/// Also covers GgufFile::embed_row (Q8_0) and pack_norm's f32 -> bf16.
///
///     cmake -S . -B build && cmake --build build && ctest --test-dir build
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gguf/gguf_file.hpp"
#include "open_qwen36/pools.hpp"
#include "open_qwen36/q4nx_file.hpp"

namespace fs = std::filesystem;

using namespace open_qwen36;

namespace {

void put_u32(std::vector<uint8_t>& f, uint32_t v) {
    f.insert(f.end(), reinterpret_cast<const uint8_t*>(&v), reinterpret_cast<const uint8_t*>(&v) + 4);
}
void put_u64(std::vector<uint8_t>& f, uint64_t v) {
    f.insert(f.end(), reinterpret_cast<const uint8_t*>(&v), reinterpret_cast<const uint8_t*>(&v) + 8);
}
void put_str(std::vector<uint8_t>& f, const std::string& s) {
    put_u64(f, s.size());
    f.insert(f.end(), s.begin(), s.end());
}
void put_kv_str(std::vector<uint8_t>& f, const std::string& k, const std::string& v) {
    put_str(f, k);
    put_u32(f, 8);                                   // GGUF_TYPE_STRING
    put_str(f, v);
}
void put_kv_f32(std::vector<uint8_t>& f, const std::string& k, float v);
void put_kv_u32(std::vector<uint8_t>& f, const std::string& k, uint32_t v) {
    put_str(f, k);
    put_u32(f, 4);                                   // GGUF_TYPE_UINT32
    put_u32(f, v);
}

uint16_t f32_to_f16(float f) {                       // truncating (test values only)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return static_cast<uint16_t>(((u >> 16) & 0x8000) | ((((u >> 23) - 127 + 15) & 0x1F) << 10) |
                                 ((u >> 13) & 0x3FF));
}

/// GGUF blocks for a [rows, cols] tensor and the reference dequant (computed
/// from the EXACT fp16 scales that hit the file).
struct Blocks {
    std::vector<uint8_t> bytes;
    std::vector<float> values;                       // rows x cols
    unsigned rows, cols, blk, type;
};

Blocks make_blocks(unsigned rows, unsigned cols, const std::string& type, std::mt19937_64& rng) {
    const unsigned nb = cols / 32;
    const bool q4 = type != "Q8_0";
    const bool has_min = type == "Q4_1";
    const unsigned blk = q4 ? (has_min ? 20u : 18u) : 34u;
    Blocks out;
    out.rows = rows;
    out.cols = cols;
    out.blk = blk;
    out.type = type == "Q4_0" ? 2u : type == "Q4_1" ? 3u : 8u;
    out.bytes.assign(static_cast<size_t>(rows) * nb * blk, 0);
    out.values.resize(static_cast<size_t>(rows) * cols);
    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned b = 0; b < nb; ++b) {
            uint8_t* p = out.bytes.data() + (static_cast<size_t>(r) * nb + b) * blk;
            const uint16_t dh = f32_to_f16(0.01f + 0.02f * static_cast<float>(rng() % 1000) / 1000.f);
            const uint16_t mh = has_min ? f32_to_f16(-0.05f + 0.04f * static_cast<float>(rng() % 1000) / 1000.f) : 0;
            std::memcpy(p, &dh, 2);
            if (has_min) std::memcpy(p + 2, &mh, 2);
            const float df = fp16_to_f32(dh);        // the exact value that hits the file
            const float mf = has_min ? fp16_to_f32(mh) : 0.f;
            for (unsigned j = 0; j < 32; ++j) {
                if (q4) {
                    const uint8_t q = static_cast<uint8_t>(rng() % 16);
                    if (j < 16) p[has_min ? 4 + j : 2 + j] |= q;
                    else p[has_min ? 4 + (j - 16) : 2 + (j - 16)] |= static_cast<uint8_t>(q << 4);
                    out.values[static_cast<size_t>(r) * cols + b * 32 + j] = df *
                        (has_min ? static_cast<float>(q) : static_cast<float>(static_cast<int>(q) - 8)) + mf;
                } else {
                    const int8_t q = static_cast<int8_t>(rng() % 256) - 128;
                    p[2 + j] = static_cast<uint8_t>(q);
                    out.values[static_cast<size_t>(r) * cols + b * 32 + j] = df * q;
                }
            }
        }
    }
    return out;
}

/// Q6_K tensor blocks and the reference dequant. Q6_K is the one super-block
/// quant whose embed_row had a real bug (the `qh` pointer advanced 16 B per
/// 128-value half instead of 32, skewing every second half of a super-block),
/// so this test is deliberately self-contained: it synthesizes the block with a
/// known code layout and checks GgufFile::embed_row against it.
struct BlocksQ6K {
    unsigned rows, cols;                 // cols is a multiple of 256
    std::vector<uint8_t> bytes;          // rows * (cols/256) * 210
    std::vector<float> values;           // rows * cols
};

BlocksQ6K make_blocks_q6k(unsigned rows, unsigned cols, std::mt19937_64& rng) {
    const unsigned nbs = cols / 256;                 // super-blocks per row
    BlocksQ6K b;
    b.rows = rows;
    b.cols = cols;
    b.bytes.assign(static_cast<size_t>(rows) * nbs * 210, 0);
    b.values.resize(static_cast<size_t>(rows) * cols);
    const uint16_t dh = f32_to_f16(0.5f);            // exactly representable
    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned sb = 0; sb < nbs; ++sb) {
            uint8_t* p = b.bytes.data() + (static_cast<size_t>(r) * nbs + sb) * 210;
            std::memcpy(p + 208, &dh, 2);            // d
            std::memset(p + 192, 1, 16);             // scales: all 1 (int8), so y = d * 1 * (code-32)
            float* y = b.values.data() + static_cast<size_t>(r) * cols + sb * 256;
            for (int v = 0; v < 256; ++v) {
                const uint8_t code = static_cast<uint8_t>(rng() % 64);
                // ql: one byte per value's LOW 4 bits; low nibble for the first 64 of a
                // 128-half, high nibble for the next 64 (the dequant's q1/q2 vs q3/q4).
                const unsigned ql_byte = (v >= 128 ? 64u : 0u) + static_cast<unsigned>(v & 63);
                if ((v % 128) >= 64) p[ql_byte] |= static_cast<uint8_t>((code & 0xF) << 4);
                else p[ql_byte] |= static_cast<uint8_t>(code & 0xF);
                // qh: 32 bytes per 128-half, each covers 4 values (bits 0/2/4/6).
                const unsigned rel = static_cast<unsigned>((v >= 128 ? v - 128 : v) % 128);
                const unsigned l = rel % 32, which = rel / 32;
                p[128 + (v >= 128 ? 32u + l : l)] |=
                    static_cast<uint8_t>(((code >> 4) & 0x3) << (2 * which));
                y[v] = 0.5f * static_cast<float>(static_cast<int>(code) - 32);
            }
        }
    }
    return b;
}

/// A one-tensor GGUF carrying a single Q6_K tensor (rows x 256).
std::string write_gguf_q6k(const fs::path& dir, const BlocksQ6K& b) {
    std::vector<uint8_t> f;
    f.insert(f.end(), {'G', 'G', 'U', 'F'});
    put_u32(f, 3);
    put_u64(f, 1);                                    // tensors
    put_u64(f, 1);                                    // kv pairs
    put_kv_str(f, "general.architecture", "llama");
    const auto align32 = [](size_t n) { return (n + 31) / 32 * 32; };
    put_str(f, "blk.0.attn_q.weight");
    put_u32(f, 2);
    put_u64(f, b.cols);                               // cols (fastest)
    put_u64(f, b.rows);
    put_u32(f, 14);                                   // GGML_TYPE_Q6_K
    put_u64(f, 0);                                    // RELATIVE offset (0: data sits at the data base)
    const size_t data_off = align32(f.size());        // == the GGUF's data base
    while (f.size() < data_off) f.push_back(0);
    f.insert(f.end(), b.bytes.begin(), b.bytes.end());
    const fs::path out = dir / "test-q6k.gguf";
    std::ofstream of(out, std::ios::binary);
    of.write(reinterpret_cast<const char*>(f.data()), static_cast<std::streamsize>(f.size()));
    return out.string();
}

std::string write_gguf(const fs::path& dir, const Blocks& mm, const Blocks& emb, const std::vector<float>& norm,
                       uint32_t heads = 0) {
    std::vector<uint8_t> f;
    f.insert(f.end(), {'G', 'G', 'U', 'F'});
    put_u32(f, 3);                                   // v3: no alignment field, fixed 32
    put_u64(f, 3);                                   // tensors
    const int n_kv = heads ? 4 : 2;
    put_u64(f, static_cast<uint64_t>(n_kv));         // kv pairs
    put_kv_str(f, "general.architecture", "llama");
    put_kv_u32(f, "llama.embedding_length", 512);
    if (heads) {
        put_kv_u32(f, "llama.attention.head_count", heads);
        put_kv_u32(f, "llama.attention.head_count_kv", heads);
    }
    // tensor infos with placeholder offsets (patched once the data layout is known)
    const auto put_tensor = [&](const std::string& name, const std::vector<uint64_t>& dims, uint32_t type) {
        put_str(f, name);
        put_u32(f, static_cast<uint32_t>(dims.size()));
        for (uint64_t d : dims) put_u64(f, d);
        put_u32(f, type);
        put_u64(f, 0);
    };
    put_tensor("blk.0.attn_q.weight", {512, 64}, mm.type);
    put_tensor("token_embd.weight", {256, 32}, 8);           // Q8_0
    put_tensor("blk.0.attn_norm.weight", {512}, 0);          // F32
    // data section: 32-aligned right after the header; the tensor offsets are
    // RELATIVE to it (GGUF semantics)
    const auto align32 = [](size_t n) { return (n + 31) / 32 * 32; };
    const size_t off = align32(f.size());
    std::vector<uint64_t> offs;                  // relative offsets, patched into the header
    size_t rel = 0;
    auto emit = [&](const std::vector<uint8_t>& b) {
        offs.push_back(rel);
        rel = align32(rel + b.size());
    };
    emit(mm.bytes);
    emit(emb.bytes);
    offs.push_back(rel);
    // patch the three tensor offsets
    const size_t n_off_fields = 3;
    for (size_t t = 0; t < n_off_fields; ++t) {
        // the t-th tensor info's offset field: walk the header
        size_t p = 4 + 4 + 8 + 8;
        // skip the kv pairs
        for (int kv = 0; kv < n_kv; ++kv) {
            uint64_t ln;
            std::memcpy(&ln, f.data() + p, 8);
            p += 8 + ln;                             // key
            uint32_t ty;
            std::memcpy(&ty, f.data() + p, 4);
            p += 4;
            if (ty == 8) {                           // string value
                std::memcpy(&ln, f.data() + p, 8);
                p += 8 + ln;
            } else {                                 // u32 value
                p += 4;
            }
        }
        for (size_t i = 0; i < t; ++i) {
            uint64_t ln;
            uint32_t nd;
            std::memcpy(&ln, f.data() + p, 8);
            p += 8 + ln;                             // name
            std::memcpy(&nd, f.data() + p, 4);
            p += 4 + 8 * nd;                         // dims
            p += 4;                                  // type
            p += 8;                                  // the offset field itself
        }
        uint64_t ln;
        uint32_t nd;
        std::memcpy(&ln, f.data() + p, 8);
        p += 8 + ln;                                 // name
        std::memcpy(&nd, f.data() + p, 4);
        p += 4 + 8 * nd + 4;                         // dims, type -> the offset field
        std::memcpy(f.data() + p, &offs[t], 8);
    }
    while (f.size() < off + offs[0]) f.push_back(0);   // pad to the first tensor
    f.insert(f.end(), mm.bytes.begin(), mm.bytes.end());
    while (f.size() < off + offs[1]) f.push_back(0);
    f.insert(f.end(), emb.bytes.begin(), emb.bytes.end());
    while (f.size() < off + offs[2]) f.push_back(0);
    for (float v : norm) {
        const uint8_t* b = reinterpret_cast<const uint8_t*>(&v);
        f.insert(f.end(), b, b + 4);
    }
    const fs::path out = dir / ("test-" + std::to_string(mm.type) + (heads ? "-h" + std::to_string(heads) : "") + ".gguf");
    std::ofstream of(out, std::ios::binary);
    of.write(reinterpret_cast<const char*>(f.data()), static_cast<std::streamsize>(f.size()));
    return out.string();
}

void put_kv_f32(std::vector<uint8_t>& f, const std::string& k, float v) {
    put_str(f, k);
    put_u32(f, 6);                                   // GGUF_TYPE_FLOAT32
    f.insert(f.end(), reinterpret_cast<const uint8_t*>(&v), reinterpret_cast<const uint8_t*>(&v) + 4);
}

/// A tensor-less granite GGUF: just the metadata granite_folds reads.
std::string write_granite_meta(const fs::path& dir, const std::vector<std::pair<std::string, float>>& scales) {
    std::vector<uint8_t> f;
    f.insert(f.end(), {'G', 'G', 'U', 'F'});
    put_u32(f, 3);
    put_u64(f, 0);
    put_u64(f, 4 + scales.size());
    put_kv_str(f, "general.architecture", "granite");
    put_kv_u32(f, "granite.embedding_length", 2560);
    put_kv_u32(f, "granite.attention.head_count", 40);
    put_kv_u32(f, "granite.attention.key_length", 64);
    for (const auto& [k, v] : scales) put_kv_f32(f, "granite." + k, v);
    const fs::path out = dir / "granite-meta.gguf";
    std::ofstream of(out, std::ios::binary);
    of.write(reinterpret_cast<const char*>(f.data()), static_cast<std::streamsize>(f.size()));
    return out.string();
}

uint32_t f32_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

/// A GGUF of arbitrary tensors (the MoE test needs 3-D expert tensors and a 2-D router;
/// write_gguf above is fixed at three tensors).
struct SynthTensor {
    std::string name;
    std::vector<uint64_t> dims;                  // fastest first, as GGUF stores them
    uint32_t type;
    std::vector<uint8_t> bytes;
};

std::string write_gguf_tensors(const fs::path& dir, const std::string& file,
                               const std::vector<SynthTensor>& ts) {
    std::vector<uint8_t> f;
    f.insert(f.end(), {'G', 'G', 'U', 'F'});
    put_u32(f, 3);
    put_u64(f, ts.size());
    put_u64(f, 1);
    put_kv_str(f, "general.architecture", "qwen35moe");
    for (const auto& t : ts) {
        put_str(f, t.name);
        put_u32(f, static_cast<uint32_t>(t.dims.size()));
        for (uint64_t d : t.dims) put_u64(f, d);
        put_u32(f, t.type);
        put_u64(f, 0);                            // patched below
    }
    const auto align32 = [](size_t n) { return (n + 31) / 32 * 32; };
    const size_t data_off = align32(f.size());
    std::vector<uint64_t> offs;
    uint64_t rel = 0;
    for (const auto& t : ts) {
        offs.push_back(rel);
        rel = align32(rel + t.bytes.size());
    }
    // walk the header to the offset field of each tensor info
    size_t p = 4 + 4 + 8 + 8;
    {
        uint64_t ln;
        uint32_t ty;
        std::memcpy(&ln, f.data() + p, 8);
        p += 8 + ln;
        std::memcpy(&ty, f.data() + p, 4);
        p += 4;
        if (ty == 8) {
            std::memcpy(&ln, f.data() + p, 8);
            p += 8 + ln;
        } else {
            p += 4;
        }
    }
    for (size_t i = 0; i < ts.size(); ++i) {
        uint64_t ln;
        uint32_t nd;
        std::memcpy(&ln, f.data() + p, 8);
        p += 8 + ln;
        std::memcpy(&nd, f.data() + p, 4);
        p += 4 + 8 * nd + 4;
        std::memcpy(f.data() + p, &offs[i], 8);
        p += 8;
    }
    while (f.size() < data_off) f.push_back(0);
    for (size_t i = 0; i < ts.size(); ++i) {
        while (f.size() < data_off + offs[i]) f.push_back(0);
        f.insert(f.end(), ts[i].bytes.begin(), ts[i].bytes.end());
    }
    const fs::path out = dir / file;
    std::ofstream of(out, std::ios::binary);
    of.write(reinterpret_cast<const char*>(f.data()), static_cast<std::streamsize>(f.size()));
    return out.string();
}

/// The q4_1 value a pool chunk holds at (row r, column k), in the bf16-scale law (5120 B).
float pool_q4_1_value(const uint8_t* d, size_t r, size_t k) {
    const size_t kb = k / 32;
    uint16_t db, mb;
    std::memcpy(&db, d + 2 * (kb * 32 + r), 2);
    std::memcpy(&mb, d + 512 + 2 * (kb * 32 + r), 2);
    const size_t p0 = (r / 16) * 4096 + k * 16 + (r % 16);
    const uint8_t byte = d[1024 + (p0 >> 1)];
    const uint8_t nib = (p0 & 1) ? byte >> 4 : byte & 0xF;
    return nib * bf16_to_f32(db) + bf16_to_f32(mb);
}

/// The same value from a Q4_1 source block, with the fp16 scales rounded to bf16 exactly
/// as the packer writes them into a 5120-byte chunk. (The 6144 f32 law is bit-exact; the
/// q4_1 law the GGUF-direct MoE uses is the one with this one rounding.)
float source_q4_1_value(const Blocks& b, size_t row, size_t col) {
    const size_t nb = b.cols / 32;
    const uint8_t* p = b.bytes.data() + (row * nb + col / 32) * b.blk;
    uint16_t dh, mh;
    std::memcpy(&dh, p, 2);
    std::memcpy(&mh, p + 2, 2);
    const float d = bf16_to_f32(f32_to_bf16(fp16_to_f32(dh)));
    const float m = bf16_to_f32(f32_to_bf16(fp16_to_f32(mh)));
    const uint8_t byte = p[4 + (col % 16)];
    const uint8_t q = (col % 32 < 16) ? (byte & 0xF) : (byte >> 4);
    return d * q + m;
}

// A container in memory: just the tensors' bytes, as a .q4nx stores small weights (bf16 or f32).
class MemFile : public WeightFile {
public:
    std::map<std::string, std::vector<uint8_t>> t;
    bool has(const std::string& name) const override { return t.count(name) != 0; }
    const uint8_t* raw(const std::string& name, size_t* nbytes) const override {
        const auto& b = t.at(name);
        if (nbytes) *nbytes = b.size();
        return b.data();
    }
    void embed_row(const std::string&, size_t, size_t, float*) const override { throw std::logic_error("unused"); }
    std::vector<float> bf16(const std::string&) const override { throw std::logic_error("unused"); }
    std::vector<float> f32(const std::string&) const override { throw std::logic_error("unused"); }
    const TensorMeta& meta(const std::string&) const override { throw std::logic_error("unused"); }
    void bf16_row(const std::string&, size_t, size_t, float*) const override { throw std::logic_error("unused"); }
    void drop_pages() override {}
};

template <typename T>
std::vector<uint8_t> bytes_of(const std::vector<T>& v) {
    std::vector<uint8_t> b(v.size() * sizeof(T));
    std::memcpy(b.data(), v.data(), b.size());
    return b;
}

}  // namespace

int main() {
    std::mt19937_64 rng(42);
    const Blocks mm = make_blocks(64, 512, "Q4_1", rng);
    std::vector<float> norm(512);
    for (float& v : norm) v = 0.9f + 0.1f * static_cast<float>(rng() % 1000) / 1000.f;
    const Blocks emb = make_blocks(32, 256, "Q8_0", rng);
    const fs::path dir = fs::temp_directory_path() / "gguf_pack_test";
    fs::create_directories(dir);
    GgufFile g(write_gguf(dir, mm, emb, norm));

    {
        const size_t rows = 35, padded_rows = 64, cols = 256, chunk_bytes = 6144;
        const uint32_t q8 = static_cast<uint32_t>(GgufFile::Type::Q8_0);
        const Blocks head = make_blocks(rows, cols, "Q8_0", rng);
        auto padded = head.bytes;
        padded.resize(padded_rows * cols / 32 * 34);
        GgufFile tied(write_gguf_tensors(dir, "tied-head.gguf", {{"token_embd.weight", {cols, rows}, q8, head.bytes}}));
        GgufFile explicit_head(write_gguf_tensors(
            dir, "padded-head.gguf", {{"output.weight", {cols, padded_rows}, q8, padded}}));
        PackOp op;
        op.op = "std_perm_gguf";
        op.tensor = "lm_head.weight";
        op.nch = padded_rows / 32 * (cols / 256);
        op.in_dim = cols;
        Manifest m;
        m.vocab = padded_rows;
        m.real_vocab = rows;
        m.chunk_bytes = chunk_bytes;
        m.lmhead_pool_bytes = op.nch * chunk_bytes;
        m.lmhead_ops = {op};

        std::vector<uint8_t> out(m.lmhead_pool_bytes, 0xff), want(out.size());
        pools::pack_lmhead(m, tied, out.data());
        pools::apply(op, explicit_head, 0, want.data(), want.size(), chunk_bytes);
        if (out != want || std::all_of(out.begin(), out.end(), [](uint8_t byte) { return byte == 0; })) {
            std::printf("FAIL zero padding for a tied GGUF head\n");
            return 1;
        }

        m.real_vocab = rows + 1;
        bool short_head_refused = false;
        try {
            pools::pack_lmhead(m, tied, out.data());
        } catch (const std::runtime_error& e) {
            short_head_refused = std::string(e.what()).find("real vocabulary rows") != std::string::npos;
        }
        if (!short_head_refused) {
            std::printf("FAIL short GGUF head must refuse missing real rows\n");
            return 1;
        }

        bool worker_error_rethrown = false;
        try {
            pools::apply(op, tied, 0, out.data(), out.size(), chunk_bytes);
        } catch (const std::runtime_error& e) {
            worker_error_rethrown = std::string(e.what()) == "gguf: row 35 past lm_head.weight";
        }
        if (!worker_error_rethrown) {
            std::printf("FAIL GGUF pack worker error was not rethrown\n");
            return 1;
        }
        std::printf("ok    GGUF head padding and OpenMP exception propagation\n");
    }

    // ---- header sanity
    if (g.kv_str("general.architecture") != "llama" || g.kv_u64("llama.embedding_length") != 512 ||
        !g.has("model.layers.0.self_attn.q_proj.weight")) {   // found through the HF-name map
        std::printf("FAIL header\n");
        return 1;
    }
    if (GgufFile::gguf_name("model.embed_tokens.weight") != "token_embd.weight" ||
        GgufFile::gguf_name("lm_head.weight") != "output.weight" ||
        GgufFile::gguf_name("model.layers.3.mlp.down_proj.weight") != "blk.3.ffn_down.weight" ||
        GgufFile::gguf_name("model.layers.1.input_layernorm.weight") != "blk.1.attn_norm.weight" ||
        GgufFile::gguf_name("model.layers.2.self_attn.q_norm.weight") != "blk.2.attn_q_norm.weight" ||
        GgufFile::gguf_name("model.layers.2.post_attention_layernorm.weight") != "blk.2.ffn_norm.weight" ||
        GgufFile::gguf_name("model.layers.2.pre_feedforward_layernorm.weight") != "blk.2.ffn_norm.weight" ||
        GgufFile::gguf_name("model.layers.2.post_feedforward_layernorm.weight") != "blk.2.post_ffw_norm.weight" ||
        GgufFile::gguf_name("model.norm.weight") != "output_norm.weight" ||
        GgufFile::gguf_name("model.layers.2.post_attention_layernorm.weight", true) != "blk.2.post_attention_norm.weight" ||
        GgufFile::gguf_name("model.layers.2.pre_feedforward_layernorm.weight", true) != "blk.2.ffn_norm.weight" ||
        GgufFile::gguf_name("model.layers.2.post_feedforward_layernorm.weight", true) != "blk.2.post_ffw_norm.weight" ||
        GgufFile::gguf_name("model.layers.2.input_layernorm.weight", true) != "blk.2.attn_norm.weight") {
        std::printf("FAIL name map\n");
        return 1;
    }
    std::printf("ok    header (kv, tensor index, HF-name map)\n");

    // ---- std_perm_gguf: the pool bytes dequantize to the GGUF values bit-exactly
    PackOp op;
    op.op = "std_perm_gguf";
    op.tensor = "model.layers.0.self_attn.q_proj.weight";   // the manifest HF-style name
    op.in_dim = 512;
    op.nch = 64ull * 512 / (32 * 256);               // 4 chunks of 32x256
    std::vector<uint8_t> pool(static_cast<size_t>(op.nch) * 6144, 0);
    pools::apply(op, g, 0, pool.data(), pool.size(), 6144);

    size_t bad = 0;
    for (size_t c = 0; c < op.nch; ++c) {
        const uint8_t* d = pool.data() + c * 6144;
        // the band law: rows0 = 64*(c/per_band) + 32*(c%2), cols0 = 256*((c%per_band)/2)
        const size_t per_band = 4;
        const size_t rows0 = 64 * (c / per_band) + 32 * (c % 2), cols0 = 256 * ((c % per_band) / 2);
        for (size_t r = 0; r < 32; ++r) {
            for (size_t k = 0; k < 256; ++k) {
                const size_t kb = k / 32;
                const float df = *reinterpret_cast<const float*>(d + 4 * (kb * 32 + r));
                const float mf = *reinterpret_cast<const float*>(d + 1024 + 4 * (kb * 32 + r));
                const size_t p = (r / 16) * 4096 + k * 16 + (r % 16);
                const uint8_t byte = d[2048 + (p >> 1)];
                const uint8_t nib = (p & 1) ? byte >> 4 : byte & 0xF;
                const float got = nib * df + mf;
                const float want = mm.values[(rows0 + r) * mm.cols + cols0 + k];
                if (f32_bits(got) != f32_bits(want)) {
                    if (bad < 5)
                        std::printf("  chunk %zu r=%zu k=%zu: got %.9g want %.9g\n", c, r, k, got, want);
                    ++bad;
                }
            }
        }
    }
    std::printf("%s std_perm_gguf (%zu of %u values differ)\n", bad ? "FAIL" : "ok", bad, op.nch * 32u * 256u);
    if (bad) return 1;

    // Q4_0 has a two-byte header (d only), unlike Q4_1's d+m header. Exercise
    // the same pack path separately so its code offset cannot drift to +4.
    {
        const Blocks mm0 = make_blocks(64, 512, "Q4_0", rng);
        GgufFile g0(write_gguf(dir, mm0, emb, norm));
        std::vector<uint8_t> p0(static_cast<size_t>(op.nch) * 6144, 0);
        pools::apply(op, g0, 0, p0.data(), p0.size(), 6144);
        size_t bad0 = 0;
        for (size_t c = 0; c < op.nch; ++c) {
            const uint8_t* d = p0.data() + c * 6144;
            const size_t per_band = 4;
            const size_t rows0 = 64 * (c / per_band) + 32 * (c % 2);
            const size_t cols0 = 256 * ((c % per_band) / 2);
            for (size_t r = 0; r < 32; ++r)
                for (size_t k = 0; k < 256; ++k) {
                    const size_t kb = k / 32;
                    const float df = *reinterpret_cast<const float*>(d + 4 * (kb * 32 + r));
                    const size_t p = (r / 16) * 4096 + k * 16 + (r % 16);
                    const uint8_t byte = d[2048 + (p >> 1)];
                    const uint8_t nib = (p & 1) ? byte >> 4 : byte & 0xF;
                    if (f32_bits(nib * df - 8.f * df) != f32_bits(mm0.values[(rows0 + r) * mm0.cols + cols0 + k]))
                        ++bad0;
                }
        }
        std::printf("%s std_perm_gguf Q4_0 (%zu values differ)\n", bad0 ? "FAIL" : "ok", bad0);
        if (bad0) return 1;

        // The q4_1 law (5120 B, the compact GGUF view): the same codes, and each f32 scale
        // plane value RNE-rounded to bf16 -- what the f32-scale kernels narrow it to on the core.
        std::vector<uint8_t> pc(static_cast<size_t>(op.nch) * 5120, 0);
        pools::apply(op, g0, 0, pc.data(), pc.size(), 5120);
        size_t badc = 0;
        for (size_t c = 0; c < op.nch; ++c) {
            const uint8_t* w = p0.data() + c * 6144;
            const uint8_t* n = pc.data() + c * 5120;
            for (size_t j = 0; j < 512; ++j) {          // 256 d then 256 m
                float f;
                std::memcpy(&f, w + 4 * j, 4);
                uint32_t u;
                std::memcpy(&u, &f, 4);
                const uint16_t want = static_cast<uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
                uint16_t got;
                std::memcpy(&got, n + 2 * j, 2);
                badc += got != want;
            }
            badc += std::memcmp(w + 2048, n + 1024, 4096) != 0;
        }
        std::printf("%s std_perm_gguf Q4_0 into the q4_1 law (%zu mismatches)\n", badc ? "FAIL" : "ok", badc);
        if (badc) return 1;

        // The Q4_0-native law (4608 B): the q4_1 law's d plane and codes with no m plane, and
        // that m plane is exactly -8 d (the value the cores form), so both views compute alike.
        std::vector<uint8_t> pq(static_cast<size_t>(op.nch) * 4608, 0);
        pools::apply(op, g0, 0, pq.data(), pq.size(), 4608);
        size_t badq = 0;
        for (size_t c = 0; c < op.nch; ++c) {
            const uint8_t* n = pc.data() + c * 5120;
            const uint8_t* q = pq.data() + c * 4608;
            badq += std::memcmp(n, q, 512) != 0;                 // d
            badq += std::memcmp(n + 1024, q + 512, 4096) != 0;   // codes
            for (size_t j = 0; j < 256; ++j) {                   // m == -8 d, in bf16
                uint16_t d16, m16;
                std::memcpy(&d16, n + 2 * j, 2);
                std::memcpy(&m16, n + 512 + 2 * j, 2);
                const float df = bf16_to_f32(d16), mf = bf16_to_f32(m16);
                badq += mf != -8.0f * df;
            }
        }
        std::printf("%s std_perm_gguf Q4_0 into the Q4_0-native law (%zu mismatches)\n", badq ? "FAIL" : "ok", badq);
        if (badq) return 1;
        bool q41_refused = false;                                // a Q4_1 block's min is not -8 d
        try {
            pools::apply(op, g, 0, pq.data(), pq.size(), 4608);
        } catch (const std::exception&) {
            q41_refused = true;
        }
        std::printf("%s std_perm_gguf refuses a Q4_1 tensor in the Q4_0-native law\n", q41_refused ? "ok   " : "FAIL");
        if (!q41_refused) return 1;
        std::vector<uint8_t> odd(static_cast<size_t>(op.nch) * 4736, 0);
        bool refused = false;
        try {
            pools::apply(op, g0, 0, odd.data(), odd.size(), 4736);
        } catch (const std::exception&) {
            refused = true;
        }
        std::printf("%s std_perm_gguf refuses a pool chunk of neither law\n", refused ? "ok   " : "FAIL");
        if (!refused) return 1;

        // bfp16_gemm (OPEN-PREFILL-BFP16) from the GGUF's own Q4_0 blocks: the [64, 512] tensor at row 256 of a
        // 1024-row B operand. Checked against pack.py's order (pack_b's reshape / transpose written out) and a
        // frexp encoder: every written block is the GGUF values' encoding, every other block zero.
        PackOp bo;
        bo.op = "bfp16_gemm";
        bo.tensor = op.tensor;
        bo.in_dim = 512;
        bo.nch = op.nch;
        bo.row0 = 256;
        bo.n_total = 1024;
        std::vector<uint8_t> bb(bo.n_total * bo.in_dim / 8 * 9, 0);
        pools::apply(bo, g0, 0, bb.data(), bb.size(), 5120);
        const size_t n = 128, kt = 64, cols = 8, nk = bo.in_dim / kt, ntg = bo.n_total / (n * cols);
        size_t badb = 0, nonzero = 0;
        for (size_t R = 0; R < bo.n_total; ++R)
            for (size_t cb = 0; cb < bo.in_dim / 8; ++cb) {
                const size_t t = R / (n * cols), c = (R / n) % cols, rr = R % n;
                const size_t blk = (((((c * ntg + t) * nk + cb / (kt / 8)) * (n / 16) + rr / 16) * (kt / 8) + cb % (kt / 8)) * 2 +
                                    (rr % 16) / 8) * 8 + rr % 8;
                const uint8_t* b = bb.data() + blk * 9;
                uint8_t want[9] = {0};
                if (R >= bo.row0 && R < bo.row0 + 64) {
                    const float* v = mm0.values.data() + (R - bo.row0) * mm0.cols + cb * 8;
                    int e = 0;
                    for (int i = 0; i < 8; ++i) e = std::max(e, static_cast<int>((f32_bits(v[i]) >> 23) & 0xFF));
                    want[0] = static_cast<uint8_t>(e);
                    for (int i = 0; i < 8; ++i) {
                        const double q = std::nearbyint(static_cast<double>(v[i]) * std::ldexp(1.0, 133 - e));
                        want[1 + i] = static_cast<uint8_t>(static_cast<int8_t>(std::min(127.0, std::max(-128.0, q))));
                    }
                }
                badb += std::memcmp(b, want, 9) != 0;
                for (int i = 0; i < 9; ++i) nonzero += b[i] != 0;
            }
        std::printf("%s bfp16_gemm from GGUF Q4_0 (%zu of %zu blocks differ; %zu nonzero bytes)\n", badb ? "FAIL" : "ok",
                    badb, static_cast<size_t>(bo.n_total * bo.in_dim / 8), nonzero);
        if (badb || !nonzero) return 1;
    }

    // ---- std_perm_gguf on a Llama-arch q/k tensor: llama.cpp's paired rows come out half-split
    {
        const uint32_t heads = 2;
        GgufFile gh(write_gguf(dir, mm, emb, norm, heads));
        std::vector<uint8_t> ph(static_cast<size_t>(op.nch) * 6144, 0);
        pools::apply(op, gh, 0, ph.data(), ph.size(), 6144);
        const size_t hd = 64 / heads, half = hd / 2;
        size_t badh = 0;
        for (size_t c = 0; c < op.nch; ++c) {
            const uint8_t* d = ph.data() + c * 6144;
            const size_t per_band = 4;
            const size_t rows0 = 64 * (c / per_band) + 32 * (c % 2), cols0 = 256 * ((c % per_band) / 2);
            for (size_t r = 0; r < 32; ++r) {
                const size_t o = rows0 + r;
                const size_t stored = (o / hd) * hd + (o % half) * 2 + (o % hd) / half;
                for (size_t k = 0; k < 256; ++k) {
                    const size_t kb = k / 32;
                    const float df = *reinterpret_cast<const float*>(d + 4 * (kb * 32 + r));
                    const float mf = *reinterpret_cast<const float*>(d + 1024 + 4 * (kb * 32 + r));
                    const size_t p = (r / 16) * 4096 + k * 16 + (r % 16);
                    const uint8_t byte = d[2048 + (p >> 1)];
                    const uint8_t nib = (p & 1) ? byte >> 4 : byte & 0xF;
                    if (f32_bits(nib * df + mf) != f32_bits(mm.values[stored * mm.cols + cols0 + k])) ++badh;
                }
            }
        }
        std::printf("%s std_perm_gguf llama q rows un-permuted (%zu values differ)\n", badh ? "FAIL" : "ok", badh);
        if (badh) return 1;
    }

    // ---- expert_stripes_gguf / expert_down_gguf: 3-D [in, n, experts] sources
    {
        const unsigned hid = 512, ff = 256, E = 3;
        const unsigned ns = ff / 128, ncol = hid / 256;
        Blocks up = make_blocks(ff * E, hid, "Q4_1", rng);      // dims [hid, ff, E]
        Blocks gate = make_blocks(ff * E, hid, "Q4_1", rng);
        Blocks down = make_blocks(hid * E, ff, "Q4_1", rng);    // dims [ff, hid, E]
        GgufFile gm(write_gguf_tensors(dir, "test-moe.gguf",
            {{"blk.0.ffn_up_exps.weight", {hid, ff, E}, 3, up.bytes},
             {"blk.0.ffn_gate_exps.weight", {hid, ff, E}, 3, gate.bytes},
             {"blk.0.ffn_down_exps.weight", {ff, hid, E}, 3, down.bytes}}));

        // the container's MoE names (model.layer.{l}., not the dense model.layers.{l}.)
        // must resolve to llama.cpp's *_exps / *_shexp tensors
        if (GgufFile::gguf_name("model.layer.0.mlp.up_exps_proj.weight") != "blk.0.ffn_up_exps.weight" ||
            GgufFile::gguf_name("model.layer.0.mlp.share_down_exps_proj.weight") != "blk.0.ffn_down_shexp.weight" ||
            GgufFile::gguf_name("model.layer.0.moe_router.weight") != "blk.0.ffn_gate_inp.weight" ||
            GgufFile::gguf_name("model.layer.0.shared_expert_gate.weight") != "blk.0.ffn_gate_inp_shexp.weight") {
            std::printf("FAIL MoE name map\n");
            return 1;
        }

        const size_t stripe = static_cast<size_t>(128) * hid / 8192 * 5120;
        const size_t B = static_cast<size_t>(hid) * ff / 8192 * 5120;
        PackOp es;
        es.op = "expert_stripes_gguf";
        es.up = "model.layer.0.mlp.up_exps_proj.weight";
        es.gate = "model.layer.0.mlp.gate_exps_proj.weight";
        es.in_dim = hid;
        es.experts = E;
        es.stripes = ns;
        es.stripe_bytes = stripe;
        std::vector<uint8_t> sp(E * 2 * ns * stripe, 0);
        pools::apply(es, gm, 0, sp.data(), sp.size(), 5120);
        size_t bads = 0;
        for (unsigned e = 0; e < E; ++e)
            for (unsigned k = 0; k < ns; ++k) {
                const uint8_t* ud = sp.data() + (2 * ns * e + 2 * k) * stripe;
                const uint8_t* gd = ud + stripe;
                for (size_t c = 0; c < stripe / 5120; ++c) {
                    const size_t tp = ncol * (c % 4) + c / 4;
                    const size_t row0 = e * ff + k * 128 + 32 * (tp / ncol);
                    const size_t col0 = 256 * (tp % ncol);
                    for (size_t r = 0; r < 32; ++r)
                        for (size_t kk = 0; kk < 256; ++kk) {
                            const size_t row = row0 + r, col = col0 + kk;
                            if (f32_bits(pool_q4_1_value(ud + c * 5120, r, kk)) !=
                                f32_bits(source_q4_1_value(up, row, col))) ++bads;
                            if (f32_bits(pool_q4_1_value(gd + c * 5120, r, kk)) !=
                                f32_bits(source_q4_1_value(gate, row, col))) ++bads;
                        }
                }
            }
        std::printf("%s expert_stripes_gguf (%zu values differ)\n", bads ? "FAIL" : "ok", bads);
        if (bads) return 1;

        PackOp ed;
        ed.op = "expert_down_gguf";
        ed.tensor = "model.layer.0.mlp.down_exps_proj.weight";
        ed.experts = E;
        ed.expert_bytes = B;
        std::vector<uint8_t> dp(E * B, 0);
        pools::apply(ed, gm, 0, dp.data(), dp.size(), 5120);
        size_t badd = 0;
        const size_t nchd = B / 5120;
        for (unsigned e = 0; e < E; ++e)
            for (size_t c = 0; c < nchd; ++c) {
                const size_t rt = 4 * (c / 8) + (c % 4), cg = (c / 4) % 2;
                const size_t fc = 2 * rt + cg;
                const size_t row0 = e * hid + 32 * fc;    // ncol == 1 (ff == 256)
                for (size_t r = 0; r < 32; ++r)
                    for (size_t kk = 0; kk < 256; ++kk)
                        if (f32_bits(pool_q4_1_value(dp.data() + e * B + c * 5120, r, kk)) !=
                            f32_bits(source_q4_1_value(down, row0 + r, kk))) ++badd;
            }
        std::printf("%s expert_down_gguf (%zu values differ)\n", badd ? "FAIL" : "ok", badd);
        if (badd) return 1;
    }

    // ---- embed_row over a 3-D expert tensor: a flat row past the first expert
    {
        const unsigned cols = 256, n = 4, E = 2;
        Blocks ex = make_blocks(n * E, cols, "Q8_0", rng);
        GgufFile ge(write_gguf_tensors(dir, "test-exps.gguf",
            {{"blk.0.ffn_up_exps.weight", {cols, n, E}, 8, ex.bytes}}));
        std::vector<float> row(cols);
        ge.embed_row("model.layer.0.mlp.up_exps_proj.weight", n + 2, cols, row.data());
        size_t bad = 0;
        for (unsigned i = 0; i < cols; ++i)
            if (f32_bits(row[i]) != f32_bits(ex.values[(n + 2) * cols + i])) ++bad;
        std::printf("%s embed_row 3-D expert (row past the first expert)\n", bad ? "FAIL" : "ok");
        if (bad) return 1;
    }

    // ---- transpose from a GGUF f16 tensor (the MoE router): decode, don't reinterpret
    {
        const unsigned E = 4, hid = 64;
        std::vector<uint8_t> f16(static_cast<size_t>(E) * hid * 2);
        std::vector<float> vals(static_cast<size_t>(E) * hid);
        for (size_t i = 0; i < vals.size(); ++i) {
            const float v = -2.f + 4.f * static_cast<float>(rng() % 1000) / 1000.f;
            const uint16_t h = f32_to_f16(v);    // truncating, so the decode is what we compare
            std::memcpy(f16.data() + 2 * i, &h, 2);
            vals[i] = fp16_to_f32(h);
        }
        GgufFile gr(write_gguf_tensors(dir, "test-router.gguf",
            {{"blk.0.ffn_gate_inp.weight", {hid, E}, 1, f16}}));   // [E, hid] rows
        PackOp rt;
        rt.op = "transpose";
        rt.tensor = "model.layer.0.moe_router.weight";
        rt.rows = E;
        rt.cols = hid;
        rt.elem = 2;
        std::vector<uint8_t> out(static_cast<size_t>(E) * hid * 2, 0);
        pools::apply(rt, gr, 0, out.data(), out.size(), 5120);
        bool bad = false;
        for (unsigned r = 0; r < E && !bad; ++r)
            for (unsigned c = 0; c < hid && !bad; ++c) {
                uint16_t got;
                std::memcpy(&got, out.data() + 2 * (c * E + r), 2);
                if (f32_bits(bf16_to_f32(got)) !=
                    f32_bits(bf16_to_f32(f32_to_bf16(vals[r * hid + c])))) {
                    std::printf("FAIL transpose f16 router [%u,%u]\n", r, c);
                    bad = true;
                }
            }
        std::printf("%s transpose GGUF f16 (router)\n", bad ? "FAIL" : "ok");
        if (bad) return 1;
    }

    // ---- pack_norm: f32 -> bf16 (RNE)
    {
        std::vector<uint8_t> nb(norm.size() * 2);
        pools::pack_norm(g, "blk.0.attn_norm.weight", nb.size(), nb.data());
        for (size_t i = 0; i < norm.size(); ++i) {
            uint32_t u;
            std::memcpy(&u, &norm[i], 4);
            const uint16_t want = static_cast<uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
            uint16_t got;
            std::memcpy(&got, nb.data() + 2 * i, 2);
            if (got != want) {
                std::printf("FAIL pack_norm[%zu]: got %04x want %04x\n", i, got, want);
                return 1;
            }
        }
        std::printf("ok    pack_norm f32 -> bf16 (RNE)\n");
    }

    // ---- lmhead_q8 from a raw GGUF Q8_0 head: 128-row bands, K tiles, then four 32-row
    // quarters; each fp16 scale widened exactly and rounded to bf16, codes copied. The second
    // pass gives every block a subnormal scale (exponent 0, mantissa 1..1023), which a pure
    // Q8_0 head quantized from BF16 has wherever a block's amax < 127 * 2^-14.
    {
        const Blocks head = make_blocks(256, 512, "Q8_0", rng);
        Blocks sub = head;
        for (size_t b = 0; b < sub.bytes.size() / 34; ++b) {
            const uint16_t h = static_cast<uint16_t>(1 + (b * 37) % 0x3FF);
            std::memcpy(sub.bytes.data() + b * 34, &h, 2);
        }
        PackOp h;
        h.op = "lmhead_q8";
        h.tensor = "model.embed_tokens.weight";
        h.in_dim = 512;
        h.chunk_bytes = 8704;
        for (const auto& [label, src] : {std::pair<const char*, const Blocks*>{"normal", &head},
                                         std::pair<const char*, const Blocks*>{"subnormal", &sub}}) {
            GgufFile q8(write_gguf_tensors(dir, std::string("test-q8-head-") + label + ".gguf",
                                           {{"token_embd.weight", {512, 256}, 8, src->bytes}}));
            std::vector<uint8_t> hp(16 * 8704, 0);
            pools::apply(h, q8, 0, hp.data(), hp.size(), 6144);
            size_t bad_scale = 0, bad_code = 0;
            for (size_t band = 0; band < 2; ++band)
                for (size_t kt = 0; kt < 2; ++kt)
                    for (size_t quarter = 0; quarter < 4; ++quarter)
                        for (size_t row = 0; row < 32; ++row)
                            for (size_t kb = 0; kb < 8; ++kb) {
                                const size_t logical = band * 128 + quarter * 32 + row;
                                const uint8_t* raw = src->bytes.data() + (logical * 16 + kt * 8 + kb) * 34;
                                uint16_t half;
                                std::memcpy(&half, raw, 2);
                                const float value = fp16_to_f32(half);
                                uint32_t bits;
                                std::memcpy(&bits, &value, 4);
                                const uint16_t want = static_cast<uint16_t>((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
                                const uint8_t* chunk = hp.data() + ((band * 2 + kt) * 4 + quarter) * 8704;
                                uint16_t got;
                                std::memcpy(&got, chunk + 2 * (kb * 32 + row), 2);
                                bad_scale += got != want;
                                for (size_t k = 0; k < 32; ++k) {
                                    const size_t off = (row / 16) * 4096 + kb * 512 + k * 16 + row % 16;
                                    bad_code += chunk[512 + off] != raw[2 + k];
                                }
                            }
            std::printf("%s lmhead_q8 GGUF Q8_0, %s fp16 scales (%zu scales, %zu codes differ)\n",
                        bad_scale || bad_code ? "FAIL" : "ok   ", label, bad_scale, bad_code);
            if (bad_scale || bad_code) return 1;
        }
    }

    // ---- pack_norm with div_running's product (Gemma 4's layer_output_scale). The same case,
    // and the same expected bits, as test_gemma4.py's test_put_div_running_matches_pools_cpp.
    {
        const std::vector<uint16_t> norm16 = {0x3F80, 0x3FC0, 0xC030, 0x3DCD, 0x4049, 0x3F81};
        const std::vector<uint16_t> want = {0x4099, 0x40E5, 0xC152, 0x3EF4, 0x4170, 0x409A};
        MemFile mf;
        mf.t["norm"] = bytes_of(norm16);
        mf.t["scale.0"] = bytes_of(std::vector<uint16_t>{0x3F33});   // bf16
        mf.t["scale.1"] = bytes_of(std::vector<float>{0.3f});         // f32
        double div = 1.0;
        for (int j = 0; j < 2; ++j) div *= pools::scalar_value(mf, "scale." + std::to_string(j));
        std::vector<uint16_t> got(norm16.size());
        pools::pack_norm(mf, "norm", got.size() * 2, reinterpret_cast<uint8_t*>(got.data()), div);
        if (got != want) {
            for (size_t i = 0; i < got.size(); ++i)
                std::printf("FAIL pack_norm div[%zu]: got %04x want %04x\n", i, got[i], want[i]);
            return 1;
        }
        std::printf("ok    pack_norm divided by the running scale product\n");
    }

    // ---- embed_row (Q8_0, bit-exact)
    {
        std::vector<float> row(256);
        g.embed_row("token_embd.weight", 7, 256, row.data());
        for (size_t i = 0; i < 256; ++i) {
            if (f32_bits(row[i]) != f32_bits(emb.values[7 * 256 + i])) {
                std::printf("FAIL embed_row(Q8_0)[%zu]: got %.9g want %.9g\n", i, row[i], emb.values[7 * 256 + i]);
                return 1;
            }
        }
        std::printf("ok    embed_row Q8_0 (bit-exact)\n");
    }

    // ---- embed_row (Q6_K, the qh-advance regression: every super-block half)
    {
        BlocksQ6K q6 = make_blocks_q6k(4, 512, rng);      // 4 rows x 2 super-blocks
        GgufFile g6(write_gguf_q6k(dir, q6));
        std::vector<float> row(512);
        bool bad6 = false;
        for (unsigned r = 0; r < 4 && !bad6; ++r) {
            g6.embed_row("blk.0.attn_q.weight", r, 512, row.data());
            for (size_t i = 0; i < 512; ++i)
                if (f32_bits(row[i]) != f32_bits(q6.values[static_cast<size_t>(r) * 512 + i])) {
                    std::printf("FAIL embed_row(Q6_K)[r%u i%zu]: got %.9g want %.9g\n", r, i, row[i],
                                q6.values[static_cast<size_t>(r) * 512 + i]);
                    bad6 = true;
                    break;
                }
        }
        std::printf("%s embed_row Q6_K (both super-block halves bit-exact)\n", bad6 ? "FAIL" : "ok");
        if (bad6) return 1;
    }

    // ---- OPEN-LMHEAD-Q6: q35_lmhead6 keeps every Q6_K value, read back the way lm_head_q8.h's q6 tile reads it
    {
        constexpr unsigned S = 2, P = 2, rows = 128 * S * 2, cols = 512, nk = cols / 256;
        BlocksQ6K q6 = make_blocks_q6k(rows, cols, rng);
        for (unsigned r = 0; r < rows; ++r)                 // random sub-block scales (the synthetic ones are all 1)
            for (unsigned sb = 0; sb < nk; ++sb) {
                uint8_t* blk = q6.bytes.data() + (static_cast<size_t>(r) * nk + sb) * 210;
                for (unsigned g = 0; g < 16; ++g) {
                    int sc = static_cast<int>(rng() % 41) - 20;
                    if (!sc) sc = 7;
                    blk[192 + g] = static_cast<uint8_t>(static_cast<int8_t>(sc));
                    for (unsigned v = 16 * g; v < 16 * g + 16; ++v) {
                        float& y = q6.values[static_cast<size_t>(r) * cols + sb * 256 + v];
                        y *= static_cast<float>(sc);         // 0.5 * (code - 32) -> 0.5 * sc * (code - 32), exact
                    }
                }
            }
        GgufFile g6(write_gguf_q6k(dir, q6));
        PackOp op;
        op.op = "q35_lmhead6";
        op.tensor = "blk.0.attn_q.weight";
        op.in_dim = cols;
        op.groups = S;
        op.width = P;
        const size_t nch = rows / 32 * nk, per_band = 4 * nk;
        std::vector<uint8_t> pool(nch * 6784, 0xAB);
        pools::apply(op, g6, 0, pool.data(), pool.size(), 6784);
        size_t bad = 0;
        for (size_t k = 0; k < nch && !bad; ++k) {
            const size_t set = k / (S * per_band), in_set = k % (S * per_band);
            const size_t g = in_set / (S * P), j = in_set % (S * P) / P, i = in_set % P;
            const size_t band = set * S + j, cb = g * P + i, rb = 4 * band + cb % 4, kt = cb / 4;
            const uint8_t* c = pool.data() + k * 6784;
            for (unsigned r = 0; r < 32; ++r) {
                float d;
                std::memcpy(&d, c + 4 * r, 4);
                for (unsigned v = 0; v < 256; ++v) {
                    const unsigned half = r / 16, ci = (v % 8) * 16 + r % 16, blk = (v / 32) * 4 + (v % 32) / 8;
                    const uint8_t lo = c[640 + half * 2048 + blk * 64 + ci % 64];
                    const unsigned code = (ci < 64 ? lo & 15u : lo >> 4) |
                                          (((c[4736 + half * 1024 + blk * 32 + ci % 32] >> (2 * (ci / 32))) & 3u) << 4);
                    const float sc = static_cast<float>(static_cast<int8_t>(c[128 + (v / 16) * 32 + r]));
                    const float got = d * sc * static_cast<float>(static_cast<int>(code) - 32);
                    const float want = q6.values[(32 * rb + r) * static_cast<size_t>(cols) + kt * 256 + v];
                    if (f32_bits(got) != f32_bits(want) && !bad++)
                        std::printf("FAIL q35_lmhead6 chunk %zu row %u k %u: got %.9g want %.9g\n", k, r, v, got, want);
                }
            }
        }
        std::printf("%s q35_lmhead6 (split order, every Q6_K value bit-exact)\n", bad ? "FAIL" : "ok");
        if (bad) return 1;
    }

    // ---- OPEN-GGUF-GRANITE-FOLD: the folds granite-4.2-3b's metadata asks for, and their refusals
    {
        const auto folds = [&](const std::vector<std::pair<std::string, float>>& s) {
            GgufFile gm(write_granite_meta(dir, s));
            return granite_folds(gm);
        };
        const auto f42 = folds({{"attention.scale", 0.015625f}, {"embedding_scale", 1.f}, {"residual_scale", 1.f},
                                {"logit_scale", 1.f}});
        if (f42.size() != 1 || f42.count("attn_q.weight") != 1 || f42.at("attn_q.weight") != 0.125f) {
            std::printf("FAIL granite_folds(granite-4.2-3b): want only attn_q.weight x0.125\n");
            return 1;
        }
        const auto all = folds({{"attention.scale", 0.25f}, {"embedding_scale", 4.f}, {"residual_scale", 0.5f},
                                {"logit_scale", 2.f}});
        if (all.size() != 5 || all.at("attn_q.weight") != 2.f || all.at("attn_output.weight") != 0.5f ||
            all.at("ffn_down.weight") != 0.5f || all.at("token_embd.weight") != 4.f || all.at("output.weight") != 0.5f) {
            std::printf("FAIL granite_folds: the four multipliers map to the wrong tensors\n");
            return 1;
        }
        int refused = 0;
        for (const auto& s : std::vector<std::vector<std::pair<std::string, float>>>{
                 {{"attention.scale", 0.02f}}, {{"residual_scale", 1.f}}, {{"attention.scale", 0.015625f}, {"logit_scale", 3.f}}}) {
            try {
                folds(s);
            } catch (const std::runtime_error&) {
                ++refused;
            }
        }
        if (refused != 3) {
            std::printf("FAIL granite_folds refused %d of 3 (not a power of two, no attention scale, logit 1/3)\n", refused);
            return 1;
        }
        std::printf("ok    granite_folds (4.2-3b: attn_q x0.125; non-power-of-two and missing refused)\n");
    }

    // ---- the fold in std_perm_gguf and embed_row: values scale exactly, codes do not move
    {
        GgufFile gf(write_gguf(dir, mm, emb, norm));
        PackOp fo;
        fo.op = "std_perm_gguf";
        fo.tensor = "model.layers.0.self_attn.q_proj.weight";
        fo.in_dim = 512;
        fo.nch = 64ull * 512 / (32 * 256);
        std::vector<uint8_t> p0(static_cast<size_t>(fo.nch) * 6144, 0), pf(p0.size(), 0);
        pools::apply(fo, gf, 0, p0.data(), p0.size(), 6144);
        gf.set_fold("attn_q.weight", 0.125f);
        gf.set_fold("token_embd.weight", 0.5f);
        pools::apply(fo, gf, 0, pf.data(), pf.size(), 6144);
        size_t badf = 0;
        for (size_t c = 0; c < fo.nch; ++c) {
            const uint8_t* a = p0.data() + c * 6144;
            const uint8_t* b = pf.data() + c * 6144;
            for (size_t i = 0; i < 2048; i += 4) {
                float x, y;
                std::memcpy(&x, a + i, 4);
                std::memcpy(&y, b + i, 4);
                badf += f32_bits(y) != f32_bits(x * 0.125f);
            }
            badf += std::memcmp(a + 2048, b + 2048, 6144 - 2048) != 0;
        }
        std::vector<float> row(256);
        gf.embed_row("token_embd.weight", 7, 256, row.data());
        for (size_t i = 0; i < 256; ++i) badf += f32_bits(row[i]) != f32_bits(emb.values[7 * 256 + i] * 0.5f);
        bool refused = false;
        try {
            gf.set_fold("attn_q.weight", 0.3f);
        } catch (const std::runtime_error&) {
            refused = true;
        }
        std::printf("%s fold: pool scales x0.125 and embed_row x0.5 bit-exact, codes unchanged, 0.3 refused (%zu differ)\n",
                    badf || !refused ? "FAIL" : "ok   ", badf);
        if (badf || !refused) return 1;
    }

    {
        // Traces: OPEN-GGUF-Q8-ROLES-QWEN36MOE. q35_q8 keeps a Q8_0 tensor's codes in q8_perm's half-tiles:
        // half-tile c of a 64-row band holds rows 16 (c % 4) .. +16 of k-tile c / 4, bf16 scales [kb * 16 + r]
        // then int8 codes [k * 16 + r] (spec.md, the q8 band law).
        const unsigned rows = 64, cols = 512, ncol = cols / 256;
        const Blocks w = make_blocks(rows, cols, "Q8_0", rng);
        GgufFile gq(write_gguf_tensors(dir, "q8-roles.gguf",
                                       {{"blk.0.attn_qkv.weight", {cols, rows}, static_cast<uint32_t>(GgufFile::Type::Q8_0), w.bytes}}));
        PackOp op;
        op.op = "q35_q8";
        op.tensor = "blk.{l}.attn_qkv.weight";
        op.nch = 2 * rows / 32 * ncol;
        op.in_dim = cols;
        std::vector<uint8_t> pool(static_cast<size_t>(op.nch) * 5120, 0xff);
        pools::apply(op, gq, 0, pool.data(), pool.size(), 5120);
        auto rne = [](float f) {
            uint32_t u;
            std::memcpy(&u, &f, 4);
            return static_cast<uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
        };
        size_t bad = 0;
        for (size_t c = 0; c < op.nch; ++c) {
            const size_t band = c / (4 * ncol), cc = c % (4 * ncol), kt = cc / 4;
            const uint8_t* h = pool.data() + c * 5120;
            for (unsigned r = 0; r < 16; ++r) {
                const size_t row = 64 * band + 16 * (cc % 4) + r;
                for (unsigned kb = 0; kb < 8; ++kb) {
                    const uint8_t* blk = w.bytes.data() + (row * (cols / 32) + kt * 8 + kb) * 34;
                    uint16_t d16, s;
                    std::memcpy(&d16, blk, 2);
                    std::memcpy(&s, h + 2 * (kb * 16 + r), 2);
                    bad += s != rne(fp16_to_f32(d16));
                    for (unsigned i = 0; i < 32; ++i) bad += h[256 + (kb * 32 + i) * 16 + r] != blk[2 + i];
                }
            }
        }
        std::printf("%s q35_q8: Q8_0 codes and bf16 scales in q8_perm half-tiles (%zu differ)\n", bad ? "FAIL" : "ok   ", bad);
        if (bad) return 1;
    }

    std::printf("GGUF-PACK OK\n");
    return 0;
}
