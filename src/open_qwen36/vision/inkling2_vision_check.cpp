/// Raw RGB driver for comparing the native vision tower against the pinned GGUF oracle.
#include "open_qwen36/vision/inkling2_vision.hpp"
#ifdef INKLING2_VISION_NPU_CHECK
#include "open_qwen36/vision/inkling2_vision_npu.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

int main(int argc, char** argv) {
    std::unique_ptr<open_qwen36::vision::Inkling2VisionBackend> backend;
#ifdef INKLING2_VISION_NPU_CHECK
    if (argc != 7) {
        std::cerr << "inkling2_vision_npu_check BUNDLE MMPROJ.gguf WIDTH HEIGHT RGB8 EMBEDDINGS.f32\n";
        return 2;
    }
#else
    if (argc != 6) {
        std::cerr << "inkling2_vision_check MMPROJ.gguf WIDTH HEIGHT RGB8 EMBEDDINGS.f32\n";
        return 2;
    }
#endif
    try {
#ifdef INKLING2_VISION_NPU_CHECK
        backend = open_qwen36::vision::make_inkling2_vision_npu(argv[1], 60000);
        if (!backend) throw std::runtime_error("bundle does not provide Inkling 2 vision streams");
        ++argv;
#endif
        const int width = std::stoi(argv[2]), height = std::stoi(argv[3]);
        std::ifstream image(argv[4], std::ios::binary);
        if (!image) throw std::runtime_error("could not open input RGB image");
        const std::vector<uint8_t> rgb{std::istreambuf_iterator<char>(image), std::istreambuf_iterator<char>()};
        const open_qwen36::vision::Inkling2Vision encoder(argv[1], std::move(backend));
        const auto start = std::chrono::steady_clock::now();
        const auto embeddings = encoder.encode(rgb, width, height);
        if (!std::all_of(embeddings.begin(), embeddings.end(), [](float value) { return std::isfinite(value); }))
            throw std::runtime_error("nonfinite vision embeddings");
        const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::ofstream output(argv[5], std::ios::binary);
        output.write(reinterpret_cast<const char*>(embeddings.data()), embeddings.size() * sizeof(float));
        if (!output) throw std::runtime_error("could not write embeddings");
        std::cerr << width << 'x' << height << " vision encode in " << seconds << " s\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
