// Compare native proposals with captures from the independent Unsloth DSpark graph.
#include "open_qwen36/gguf_file.hpp"
#include "open_qwen36/inkling2_dspark.hpp"
#include "open_qwen36/inkling2_host.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {

using open_qwen36::Inkling2Dspark;
using open_qwen36::Inkling2Host;
constexpr size_t vocab = Inkling2Host::vocabulary_size;
constexpr size_t features = Inkling2Host::draft_feature_width;
constexpr size_t proposal = Inkling2Dspark::max_proposal;

std::vector<float> read_capture(const std::string& path, size_t floats) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const auto bytes = static_cast<std::streamsize>(floats * sizeof(float));
    std::vector<float> result(floats);
    if (file.tellg() != bytes || !file.seekg(0) || !file.read(reinterpret_cast<char*>(result.data()), bytes))
        throw std::runtime_error("cannot read capture " + path);
    return result;
}

// Once an argmax diverges, the next Markov step receives a different embedding.
// Record every row and check the entire greedy trajectory.
bool compare(const std::vector<int>& ids, const std::vector<float>& actual, const std::vector<float>& expected) {
    if (actual.size() != expected.size() || ids.size() != proposal)
        throw std::runtime_error("invalid proposal shape");
    bool ok = true;
    for (size_t row = 0; row < proposal; ++row) {
        const float* ref = expected.data() + row * vocab;
        const float* out = actual.data() + row * vocab;
        const int token = static_cast<int>(std::max_element(ref, ref + vocab) - ref);
        double error = 0, power = 0;
        for (size_t col = 0; col < vocab; ++col) {
            const double diff = static_cast<double>(out[col]) - ref[col];
            error += diff * diff;
            power += static_cast<double>(ref[col]) * ref[col];
        }
        const double rel_l2 = std::sqrt(error / power);
        std::cout << "  " << (row + 1) << " id=" << ids[row] << " oracle=" << token << " rel_l2=" << rel_l2 << '\n';
        ok &= ids[row] == token && rel_l2 < 0.05;
    }
    return ok;
}

void check_window(const char* draft, const open_qwen36::GgufFile& target, const std::vector<float>& captured) {
    constexpr size_t start = 1023, end = 1031, capacity = end + 16;
    std::vector<float> input((end + 8) * features);
    const size_t source_rows = captured.size() / features;
    for (size_t row = 0; row < end + 8; ++row)
        std::copy_n(captured.begin() + row % source_rows * features, features, input.begin() + row * features);

    Inkling2Dspark direct(draft, target, capacity), staged(draft, target, capacity);
    staged.append(input, start);
    const std::vector<float> tail(input.begin() + start * features, input.end());
    input.resize(end * features);
    direct.append(input, end);
    staged.propose(152717, proposal);
    staged.append(tail, 0);
    if (staged.position() != start) throw std::runtime_error("draft changed the prefix at the sliding boundary");
    staged.append(tail, end - start);
    std::vector<float> expected, actual;
    const auto expected_ids = direct.propose(68014, proposal, &expected);
    const auto actual_ids = staged.propose(68014, proposal, &actual);
    if (staged.position() != end || actual != expected || actual_ids != expected_ids)
        throw std::runtime_error("staged draft cache differed from the direct prefix after the sliding boundary");
    std::cout << "draft window: exact logits and IDs after accepting " << end - start
              << " of " << tail.size() / features << " feature rows\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4 && (argc != 5 || std::string(argv[4]) != "--window")) {
        std::cerr << "usage: inkling2_dspark_check ORACLE-TARGET.gguf DRAFT.gguf CAPTURE-STEM [--window]\n";
        return 2;
    }
    try {
        const open_qwen36::GgufFile target(argv[1]);
        Inkling2Dspark model(argv[2], target, 192);
        const std::string stem = argv[3];
        auto prefill = read_capture(stem + "-prefill.f32", 5 * features);
        model.append(prefill, 5);
        bool ok = true;
        for (const auto& step : std::array<std::array<size_t, 3>, 3>{{{5, 152717, 3}, {8, 32, 1}, {9, 13, 1}}}) {
            const auto [position, anchor, commit] = step;
            const auto path = stem + "-" + std::to_string(position);
            if (model.position() != position) throw std::runtime_error("wrong position before draft");
            std::vector<float> logits;
            const auto start = std::chrono::steady_clock::now();
            const auto ids = model.propose(static_cast<int>(anchor), proposal, &logits);
            std::cout << "position=" << position << " seconds="
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << '\n';
            ok &= compare(ids, logits, read_capture(path + "-draft-logits.f32", proposal * vocab));
            if (model.position() != position) throw std::runtime_error("draft changed committed position");
            auto verified = read_capture(path + "-verify-features.f32", Inkling2Dspark::block_size * features);
            model.append(verified, 0);
            if (model.position() != position) throw std::runtime_error("empty commit changed position");
            model.append(verified, commit);
        }
        model.reset();
        model.append(prefill, 5);
        const auto replay = model.propose(152717, 2);
        if (model.position() != 5 || replay != std::vector<int>{13, 68014})
            throw std::runtime_error("reset and replay changed first proposals");
        if (argc == 5) check_window(argv[2], target, prefill);
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
