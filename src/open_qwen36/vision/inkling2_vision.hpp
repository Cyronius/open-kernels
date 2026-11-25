#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace open_qwen36::vision {

struct Linear;

class Inkling2VisionBackend {
public:
    virtual ~Inkling2VisionBackend() = default;
    virtual void project(const float* input, int rows, const Linear& weights, float* output) = 0;
    // Q, K and V are contiguous within each token. The rotation uses adjacent pairs.
    virtual void attend(const float* qkv, int rows, const std::vector<float>& cos,
                        const std::vector<float>& sin, float* output) = 0;
};

/// Inkling 2 Mini vision tower and projector from a mainline BF16 mmproj GGUF.
class Inkling2Vision {
public:
    explicit Inkling2Vision(const std::string& mmproj, std::unique_ptr<Inkling2VisionBackend> backend = {});
    ~Inkling2Vision();

    /// Input is interleaved RGB8, with both edges from 40 through 4096 pixels.
    /// Returns token-major text embeddings, one row per 40-by-40 padded tile.
    std::vector<float> encode(const std::vector<uint8_t>& rgb, int width, int height) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace open_qwen36::vision
