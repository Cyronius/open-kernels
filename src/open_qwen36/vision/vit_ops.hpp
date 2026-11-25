#pragma once

#include <vector>

namespace open_qwen36::vision {

struct Linear;
struct VitConfig;

namespace ops {

float dot(const float* a, const float* b, int size);
void project(const float* x, int rows, const Linear& weights, float* y, int stride = 0);
void normalize(const float* x, int rows, int width, const float* weight, const float* bias, float eps, float* y);
// Bidirectional attention with adjacent rotary pairs and FP16 Q, K and V inputs.
void full_attention(const VitConfig& cfg, const float* qkv, int rows,
                    const std::vector<float>& cos, const std::vector<float>& sin, float* y);
void gelu(std::vector<float>& rows);

}  // namespace ops
}  // namespace open_qwen36::vision
