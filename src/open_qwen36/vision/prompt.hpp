/// \file prompt.hpp
/// \brief A prompt with images (OPEN-VISION-BLOCK): the processor's patches, the vision tower, and the
/// prefill that puts its rows at their M-RoPE positions. Shared by open_qwen36_cli and phlegm's bridge.
///
/// The caller resizes each image (Qwen2-VL's smart_resize: both sides multiples of patch * merge, the pixel
/// count inside the processor's bounds) and expands the image's one placeholder token into
/// grid_h * grid_w / merge^2 image tokens. This file normalises, cuts the patches in the processor's order,
/// runs the tower and prefills.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "open_qwen36/core.hpp"
#include "vit.hpp"

namespace open_qwen36::vision {

/// One image after the resize: width x height RGB8, row-major, both sides multiples of patch * merge.
struct RgbImage {
    int width = 0, height = 0;
    std::vector<uint8_t> rgb;
};

/// One image as the tower takes it: [grid_h * grid_w, patch_dim] f32 in Qwen2VLImageProcessor's order
/// (merge block major, then the merge^2 patches, each patch as channel, temporal copy, row, column).
struct PromptImage {
    int grid_h = 0, grid_w = 0;
    std::vector<float> patches;
    size_t tokens(int merge) const { return static_cast<size_t>(grid_h) * grid_w / (merge * merge); }
};

/// Normalise ((x / 255 - mean) / std per channel) and cut into patches; a still image is repeated over the
/// temporal patch, as the processor does.
PromptImage patchify(const VitConfig& cfg, const RgbImage& im, const float mean[3], const float stdv[3]);

/// The processor's resize target for an image of w x h (Qwen2-VL smart_resize with factor patch * merge):
/// both sides multiples of the factor, the pixel count in [min_pixels, max_pixels], aspect kept.
void smart_resize(int w, int h, int factor, long long min_pixels, long long max_pixels, int* out_w, int* out_h);

struct ImagePrefillStats {
    double vit_ms = 0, prefill_ms = 0;
    size_t image_tokens = 0;
};

/// Prefill `ids` from the core's position. Every image_token_id run in ids belongs to the next image of
/// `images` and must hold exactly its tokens(). block = the MoE layer-major route
/// (Core::step_gemm_prompt_mrope); else one step a row. The M-RoPE counter continues from where it is.
ImagePrefillStats prefill_images(Core& core, const VitConfig& cfg, const VitWeights& w, const std::vector<int>& ids,
                                 const std::vector<PromptImage>& images, bool block, bool want_logits);

}  // namespace open_qwen36::vision
