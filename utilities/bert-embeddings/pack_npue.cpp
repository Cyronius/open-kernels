// SPDX-License-Identifier: MIT
#include "npue_pack.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "usage: pack-npue {bert|nomic|gte} CHECKPOINT SOURCE_REPO POOL SEQ TILE_N OUT\n";
        return 2;
    }
    try {
        const std::string arch = argv[1], dir = argv[2], repo = argv[3], pool = argv[4], out = argv[7];
        const auto seq = std::stoll(argv[5]);
        const auto tile_n = std::stoll(argv[6]);
        const auto layout = npue::gemm_b_layout(64, tile_n);
        auto log = +[](const std::string& message) { std::cout << message << '\n'; };
        if (arch == "bert") {
            npue::prepare_model(dir + "/model.safetensors", dir + "/vocab.txt", dir + "/config.json",
                                pool, repo, out, "", layout.json, layout.hash, 64, tile_n, seq, log);
        } else if (arch == "nomic") {
            npue::prepare_model_nomic(dir, pool, repo, out, layout.json, layout.hash, 64, tile_n, seq, log);
        } else if (arch == "gte") {
            npue::prepare_model_gte(dir, pool, repo, out, layout.json, layout.hash, 64, tile_n, seq, log);
        } else {
            throw std::invalid_argument("unknown embedding architecture");
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
