// Model-backed checks of tentative logits, DSpark features, and the committed KV and convolution state.
#include "open_qwen36/inkling2_host.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using open_qwen36::Inkling2Host;
constexpr size_t vocab = Inkling2Host::vocabulary_size;
constexpr size_t feature_width = Inkling2Host::draft_feature_width;
constexpr size_t hidden = Inkling2Host::embedding_width;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Operation>
void rejects(Operation operation, const char* message) {
    try {
        operation();
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error(message);
}

void exercise(Inkling2Host& model, Inkling2Host& replay, const std::vector<int>& proposals,
              size_t accepted) {
    const size_t position = model.position();
    check(replay.position() == position, "wrong replay prefix position");
    const std::vector<int> keep(proposals.begin(), proposals.begin() + accepted);
    const std::vector<int> follow{441, 19, 17, 136, 697};
    std::vector<float> expected_features;
    constexpr std::array<size_t, 3> source_visits{4, 22, 40};
    std::array<std::vector<float>, source_visits.size()> observed;
    const auto observe = [&](size_t visit, size_t, const std::vector<float>& residual) {
        const auto found = std::find(source_visits.begin(), source_visits.end(), visit);
        if (found != source_visits.end()) observed[found - source_visits.begin()] = residual;
    };

    const auto last = keep.empty() ? std::vector<float>{} : replay.append(keep, true, observe, &expected_features);
    const auto after = replay.append(follow);

    std::vector<float> features;
    const auto tentative = model.verify(proposals, &features);
    check(tentative.size() == proposals.size() * vocab, "unexpected verify logits size");
    check(features.size() == proposals.size() * feature_width, "unexpected DSpark feature shape");
    check(std::all_of(features.begin(), features.end(), [](float value) { return std::isfinite(value); }),
          "nonfinite DSpark features");
    check(expected_features.size() == accepted * feature_width, "unexpected accepted DSpark feature shape");
    check(std::equal(expected_features.begin(), expected_features.end(), features.begin()),
          "DSpark features differ from direct replay");
    for (size_t tap = 0; tap < observed.size(); ++tap) {
        check(observed[tap].size() == accepted * hidden, "missing pre-visit residual");
        for (size_t row = 0; row < accepted; ++row) {
            const auto begin = observed[tap].begin() + row * hidden;
            check(std::equal(begin, begin + hidden, features.begin() + row * feature_width + tap * hidden),
                  "DSpark pre-visit feature has the wrong values or order");
        }
    }
    check(model.position() == position, "verify wrote live state");
    rejects([&] { model.append({441}, false); }, "append discarded a pending verify pass");
    rejects([&] { model.verify({441}); }, "verify replaced a pending pass");
    rejects([&] { model.accept(proposals.size() + 1); }, "accept committed more than was evaluated");
    check(model.position() == position, "invalid operations changed position");
    if (accepted)
        check(std::equal(last.begin(), last.end(), tentative.begin() + (accepted - 1) * vocab),
              "verify row depends on later draft tokens");
    model.accept(accepted);
    check(model.position() == position + accepted, "verify prefix position mismatch");
    check(model.append(follow) == after, "rejected proposal affected subsequent logits");
    rejects([&] { model.accept(0); }, "accept worked without a pending pass");
    std::cout << "verified logits, DSpark features and " << accepted << '/' << proposals.size()
              << " accepted after " << position << " tokens" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::cerr << "usage: inkling2_verify_check MODEL.gguf [WINDOW-PREFIX-IDS.txt]\n";
        return 2;
    }
    try {
        const std::vector<int> prospects{50, 747, 350, 42, 220, 13, 68014, 161721};
        check(prospects.size() == Inkling2Host::verify_limit, "fixture must cover the full verify width");
        constexpr size_t prefix_size = 1023;
        const size_t capacity = argc == 3 ? prefix_size + 128 : 128;
        Inkling2Host model(argv[1], capacity), replay(argv[1], capacity);
        if (argc == 3) {
            std::ifstream input(argv[2]);
            std::vector<int> prefix;
            for (int id; prefix.size() < prefix_size && input >> id;) prefix.push_back(id);
            check(prefix.size() == prefix_size, "expected at least 1023 prefix token IDs");
            for (auto* route : {&model, &replay}) {
                constexpr size_t chunk = 256;
                for (size_t at = 0; at < prefix.size(); at += chunk) {
                    const auto end = prefix.begin() + std::min(at + chunk, prefix.size());
                    route->append(std::vector<int>(prefix.begin() + at, end), false);
                    std::cout << (route == &model ? "tentative" : "direct") << " route prefilled "
                              << route->position() << " tokens" << std::endl;
                }
            }
            // The first partial commit crosses the sliding-window boundary.
            for (size_t accepted : {2, 0, 7, 8}) exercise(model, replay, prospects, accepted);
        } else {
            for (size_t accepted = 0; accepted <= prospects.size(); ++accepted)
                exercise(model, replay, prospects, accepted);
        }
        model.verify(prospects);
        model.reset();
        check(model.position() == 0, "reset did not clear verify state");
        rejects([&] { model.accept(1); }, "reset left a verify pass pending");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
