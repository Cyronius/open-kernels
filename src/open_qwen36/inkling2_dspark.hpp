#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace open_qwen36 {

class GgufFile;

/// The Inkling 2 Mini DSpark draft graph. The target GGUF supplies its embedding and output head.
class Inkling2Dspark {
public:
    static constexpr size_t block_size = 8;
    static constexpr size_t max_proposal = block_size - 1;

    // The target mapping must outlive the drafter.
    Inkling2Dspark(const std::string& draft_gguf, const GgufFile& target, size_t context);
    ~Inkling2Dspark();
    Inkling2Dspark(const Inkling2Dspark&) = delete;
    Inkling2Dspark& operator=(const Inkling2Dspark&) = delete;

    /// Append the accepted prefix of token-major target features.
    void append(const std::vector<float>& features, size_t accepted);
    /// Run an anchor and masked block without modifying committed state. Return the proposed IDs after the anchor.
    std::vector<int> propose(int anchor, size_t count, std::vector<float>* logits = nullptr) const;
    void reset();
    size_t position() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace open_qwen36
