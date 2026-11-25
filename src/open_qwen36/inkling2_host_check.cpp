/// Token-ID driver for comparing native execution against model oracles.
#include "open_qwen36/inkling2_host.hpp"
#ifdef INKLING2_NPU_CHECK
#include "open_qwen36/inkling2_npu.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>

int main(int argc, char** argv) {
    std::unique_ptr<open_qwen36::Inkling2ExpertProjector> projector;
#ifdef INKLING2_NPU_CHECK
    if (argc < 5 || argc > 8) {
        std::cerr << "inkling2_npu_check BUNDLE MODEL.gguf IDS.txt LOGITS.f32 [GREEDY [CHUNK [RESIDUAL_PREFIX]]]\n";
        return 2;
    }
#else
    if (argc < 4 || argc > 7) {
        std::cerr << "inkling2_host_check MODEL.gguf IDS.txt LOGITS.f32 [GREEDY [CHUNK [RESIDUAL_PREFIX]]]\n";
        return 2;
    }
#endif
    try {
#ifdef INKLING2_NPU_CHECK
        projector = open_qwen36::make_inkling2_npu_experts(argv[1]);
        --argc;
        ++argv;
#endif
        std::ifstream input(argv[2]);
        std::vector<int> ids;
        for (int id; input >> id;) ids.push_back(id);
        if (!input.eof() || ids.empty()) throw std::runtime_error("expected whitespace-separated token IDs");
        const int greedy = argc > 4 ? std::stoi(argv[4]) : 0;
        const int chunksize = argc > 5 ? std::stoi(argv[5]) : static_cast<int>(ids.size());
        if (greedy < 0 || chunksize <= 0) throw std::runtime_error("invalid greedy count or chunk size");

        open_qwen36::Inkling2Host model(argv[1], ids.size() + static_cast<size_t>(std::max(0, greedy - 1)),
                                       std::move(projector));
        auto save = [](const std::string& path, const std::vector<float>& data) {
            std::ofstream out(path, std::ios::binary);
            out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
            if (!out) throw std::runtime_error("could not write " + path);
        };
        open_qwen36::Inkling2Host::VisitObserver observer;
        if (argc > 6) {
            observer = [&](size_t visit, size_t, const std::vector<float>& residual) {
                save(std::string(argv[6]) + std::to_string(model.position()) + "_" +
                     std::to_string(visit) + ".f32", residual);
            };
        }
        auto start = std::chrono::steady_clock::now();
        std::vector<float> logits;
        for (size_t i = 0; i < ids.size();) {
            const size_t end = std::min(i + static_cast<size_t>(chunksize), ids.size());
            const std::vector<int> chunk(ids.begin() + i, ids.begin() + end);
            logits = model.append(chunk, end == ids.size(), observer);
            i = end;
        }
        save(argv[3], logits);
        std::cerr << "prefill " << model.position() << " tokens in "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s\n";
        start = std::chrono::steady_clock::now();
        for (int i = 0; i < greedy; ++i) {
            const auto id = static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
            std::cout << id << std::endl;
            if (i + 1 < greedy) logits = model.append({id});
        }
        if (greedy > 1)
            std::cerr << "decode " << greedy - 1 << " tokens in "
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s\n";
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
