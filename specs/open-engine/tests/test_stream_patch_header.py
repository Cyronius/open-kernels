"""The host patch header must compile without XRT and retain KV-slice offsets."""
from pathlib import Path
import subprocess
from native_compiler import executable_path, find_cxx

import pytest


def test_row_patches_compile_and_preserve_slice_geometry(tmp_path):
    compiler = find_cxx()
    if compiler is None:
        pytest.skip('requires a GNU-style C++ compiler')
    source = tmp_path / 'rows.cpp'
    source.write_text(r'''
#include "stream_patch.hpp"
#include <cassert>
int main() {
    using namespace stream_patch;
    AttnGeometry g;
    g.kv_row = 256;
    g.ptab_row = 64;
    std::vector<AttnRowPatch> table = {
        {0, 0, 0, 2, 0, 16},
        {1, 1, 0x80000000u, 0},
        {2, 2, 0, 1},
        {3, 3, 0, 2, 128},
    };
    uint32_t words[4] = {};
    attn_rows_apply(words, table, 7, g);
    assert(words[0] == 9 * 64);
    assert(words[1] == (0x80000000u | 7 * 256));
    assert(words[2] == 8 * 64);
    assert(words[3] == 128);
    g.kvslice = 4;
    attn_rows_apply(words, table, 7, g);
    assert(words[0] == 12 * 16);
    assert(words[1] == (0x80000000u | 7 * 256));
    assert(words[2] == 8 * 64);
    assert(words[3] == 128);
}
''')
    harness = Path(__file__).resolve().parents[3] / 'open_kernels/harness'
    binary = executable_path(tmp_path / 'rows')
    subprocess.run([compiler, '-std=c++17', '-I', str(harness), str(source),
                    '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)

def test_bert_key_length_patch_table(tmp_path):
    compiler = find_cxx()
    if compiler is None:
        pytest.skip("requires a GNU-style C++ compiler")
    source = tmp_path / "bertlength.cpp"
    source.write_text(r"""
#include "stream_patch.hpp"
#include <cassert>

int main() {
    using stream_patch::bert_length_table;
    constexpr uint32_t marker = 381;
    std::vector<uint32_t> words = {0, 0, 32, 0};
    for (uint32_t col = 0; col < 8; ++col) {
        for (uint32_t row = 2; row < 6; ++row) {
            const auto address = (col << 25) | (row << 20) | 0x8000;
            words.insert(words.end(), {0, 0, address, 0, marker, 24});
        }
    }
    words[3] = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
    const auto table = bert_length_table(words, marker);
    for (size_t tile = 0; tile < table.size(); ++tile) {
        assert(table[tile] == 8 + 6 * tile);
        assert(words[table[tile]] == marker);
    }
    const auto reject = [&](const std::vector<uint32_t>& data) {
        try {
            bert_length_table(data, marker);
        } catch (const std::runtime_error&) {
            return;
        }
        assert(false && "invalid transaction accepted");
    };
    auto malformed = words;
    malformed[12] = malformed[6];  // duplicate compute tile
    reject(malformed);
    malformed = words;
    malformed.pop_back();
    malformed[3] = static_cast<uint32_t>(malformed.size() * sizeof(uint32_t));
    reject(malformed);  // incomplete final WRITE32
}
""", encoding="utf-8")
    harness = Path(__file__).resolve().parents[3] / "open_kernels/harness"
    binary = executable_path(tmp_path / "bertlength")
    subprocess.run([compiler, "-std=c++17", "-I", str(harness), str(source),
                    "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
