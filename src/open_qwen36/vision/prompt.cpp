// A prompt with images (OPEN-VISION-BLOCK): see prompt.hpp.
#include "prompt.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace open_qwen36::vision {

PromptImage patchify(const VitConfig& cfg, const RgbImage& im, const float mean[3], const float stdv[3]) {
    const int P = cfg.patch, M = cfg.merge, TP = cfg.temporal, C = cfg.channels;
    if (C != 3) throw std::runtime_error("open_qwen36: patchify takes RGB images");
    if (im.width <= 0 || im.height <= 0 || im.width % (P * M) || im.height % (P * M))
        throw std::runtime_error("open_qwen36: an image's sides must be positive multiples of " + std::to_string(P * M));
    if (im.rgb.size() != static_cast<size_t>(im.width) * im.height * 3)
        throw std::runtime_error("open_qwen36: the image holds " + std::to_string(im.rgb.size()) + " bytes, not w * h * 3");
    PromptImage out;
    out.grid_h = im.height / P;
    out.grid_w = im.width / P;
    const size_t pd = static_cast<size_t>(cfg.patch_dim());
    out.patches.resize(static_cast<size_t>(out.grid_h) * out.grid_w * pd);
    float scale[3], shift[3];
    for (int c = 0; c < 3; ++c) {
        scale[c] = 1.0f / (255.0f * stdv[c]);
        shift[c] = -mean[c] / stdv[c];
    }
    // Qwen2VLImageProcessor: (gt, tp, C, gh/M, M, P, gw/M, M, P) -> (gt, gh/M, gw/M, M, M, C, tp, P, P)
    size_t idx = 0;
    for (int bh = 0; bh < out.grid_h / M; ++bh)
        for (int bw = 0; bw < out.grid_w / M; ++bw)
            for (int mh = 0; mh < M; ++mh)
                for (int mw = 0; mw < M; ++mw) {
                    const int py0 = (bh * M + mh) * P, px0 = (bw * M + mw) * P;
                    float* dst = out.patches.data() + (idx++) * pd;
                    for (int c = 0; c < C; ++c)
                        for (int t = 0; t < TP; ++t)
                            for (int y = 0; y < P; ++y) {
                                const uint8_t* row = im.rgb.data() + (static_cast<size_t>(py0 + y) * im.width + px0) * 3;
                                for (int x = 0; x < P; ++x) *dst++ = static_cast<float>(row[3 * x + c]) * scale[c] + shift[c];
                            }
                }
    return out;
}

void smart_resize(int w, int h, int factor, long long min_pixels, long long max_pixels, int* out_w, int* out_h) {
    if (w <= 0 || h <= 0 || factor <= 0) throw std::runtime_error("open_qwen36: smart_resize of an empty image");
    if (std::max(w, h) / std::min(w, h) > 200) throw std::runtime_error("open_qwen36: image aspect ratio over 200");
    auto round_to = [&](double v) { return std::max(factor, static_cast<int>(std::lround(v / factor)) * factor); };
    int hb = round_to(h), wb = round_to(w);
    if (static_cast<long long>(hb) * wb > max_pixels) {
        const double beta = std::sqrt(static_cast<double>(h) * w / static_cast<double>(max_pixels));
        hb = std::max(factor, static_cast<int>(std::floor(h / beta / factor)) * factor);
        wb = std::max(factor, static_cast<int>(std::floor(w / beta / factor)) * factor);
    } else if (static_cast<long long>(hb) * wb < min_pixels) {
        const double beta = std::sqrt(static_cast<double>(min_pixels) / (static_cast<double>(h) * w));
        hb = static_cast<int>(std::ceil(h * beta / factor)) * factor;
        wb = static_cast<int>(std::ceil(w * beta / factor)) * factor;
    }
    *out_w = wb;
    *out_h = hb;
}

ImagePrefillStats prefill_images(Core& core, const VitConfig& cfg, const VitWeights& w, const std::vector<int>& ids,
                                 const std::vector<PromptImage>& images, bool block, bool want_logits) {
    ImagePrefillStats st;
    if (!core.has_mrope() || core.image_token_id() < 0)
        throw std::runtime_error("open_qwen36: this model has no image_token_id / mrope_section");
    if (ids.empty()) throw std::runtime_error("open_qwen36: an empty prompt");
    const int img_tok = core.image_token_id();
    const size_t hidden = core.manifest().hidden;
    // the image runs: each must be one whole image
    {
        size_t img = 0;
        for (size_t i = 0; i < ids.size();) {
            if (ids[i] != img_tok) { ++i; continue; }
            size_t j = i;
            while (j < ids.size() && ids[j] == img_tok) ++j;
            if (img >= images.size()) throw std::runtime_error("open_qwen36: more image runs in the prompt than images");
            if (j - i != images[img].tokens(cfg.merge))
                throw std::runtime_error("open_qwen36: image " + std::to_string(img) + " has " + std::to_string(j - i) +
                                         " tokens in the prompt, its grid gives " +
                                         std::to_string(images[img].tokens(cfg.merge)));
            ++img;
            i = j;
        }
        if (img != images.size()) throw std::runtime_error("open_qwen36: fewer image runs in the prompt than images");
    }
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::vector<float>> embs;
    for (const auto& im : images) {
        if (im.patches.size() != static_cast<size_t>(im.grid_h) * im.grid_w * cfg.patch_dim())
            throw std::runtime_error("open_qwen36: an image's patches do not match its grid");
        if (!cfg.deepstack.empty()) throw std::runtime_error("open_qwen36: deepstack towers are not wired to this prefill");
        embs.push_back(vit_forward(cfg, w, im.patches.data(), im.grid_h, im.grid_w));
        if (embs.back().size() != im.tokens(cfg.merge) * hidden)
            throw std::runtime_error("open_qwen36: the vision tower's width does not match the model's hidden size");
        st.image_tokens += im.tokens(cfg.merge);
    }
    st.vit_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    // positions: text (c, c, c) then c + 1; image token j (c, c + j / gw, c + j % gw), then c + max(gh, gw)
    t0 = std::chrono::steady_clock::now();
    int64_t c = core.mrope_active() ? core.mrope_pos() : static_cast<int64_t>(core.position());
    std::vector<int64_t> mpos(3 * ids.size());
    std::vector<const float*> rows(ids.size(), nullptr);
    std::vector<int64_t> adv(ids.size(), 0);   // an image's last token: how far the counter jumps after it
    size_t img = 0, j = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] != img_tok) {
            mpos[3 * i] = mpos[3 * i + 1] = mpos[3 * i + 2] = c++;
            continue;
        }
        const int64_t gh = images[img].grid_h / cfg.merge, gw = images[img].grid_w / cfg.merge;
        mpos[3 * i] = c;
        mpos[3 * i + 1] = c + static_cast<int64_t>(j) / gw;
        mpos[3 * i + 2] = c + static_cast<int64_t>(j) % gw;
        rows[i] = embs[img].data() + j * hidden;
        if (++j == static_cast<size_t>(gh * gw)) {
            adv[i] = std::max(gh, gw);
            c += std::max(gh, gw);
            ++img;
            j = 0;
        }
    }
    if (block) {
        core.step_gemm_prompt_mrope(ids, rows, mpos, c, want_logits);
    } else {
        core.mrope_begin();
        for (size_t i = 0; i < ids.size(); ++i) {
            const bool last = want_logits && i + 1 == ids.size();
            if (rows[i]) {
                core.step_embed(rows[i], last, &mpos[3 * i]);   // does not advance the counter
                core.mrope_advance(adv[i]);
            } else {
                core.step(ids[i], last);                        // (c, c, c), then c + 1
            }
        }
        if (core.mrope_pos() != c) throw std::runtime_error("open_qwen36: the M-RoPE counter lost its place");
    }
    st.prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return st;
}

}  // namespace open_qwen36::vision
