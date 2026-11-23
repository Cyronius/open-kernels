// SPDX-License-Identifier: MIT
#include "npue_encoder.hpp"
#include "bert_attention.hpp"
#include "bert_attention_files.hpp"
#include "embedding_width.hpp"
#include "gte_sparse.hpp"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <io.h>

namespace {
using Json = nlohmann::json;

bool is_nomic_v15(const npue::enc::Embedder& model) {
    return model.source_repo() == "nomic-ai/nomic-embed-text-v1.5";
}

void reply(std::FILE* out, const Json& response) {
    const auto record = response.dump(-1, ' ', true, Json::error_handler_t::replace);
    if (std::fputs(record.c_str(), out) == EOF || std::fputc('\n', out) == EOF ||
        std::fflush(out) == EOF)
        throw std::runtime_error("write to the embedding client failed");
}

Json embed(npue::enc::Embedder& model, const Json& request,
           const std::optional<npue::enc::GteSparseHead>& sparse_head) {
    const bool text_input = request.contains("texts");
    if (!request.is_object() || text_input == request.contains("token_ids"))
        throw std::invalid_argument("provide either texts or token_ids");
    const auto& inputs = request.at(text_input ? "texts" : "token_ids");
    if (!inputs.is_array() || inputs.empty())
        throw std::invalid_argument("BERT embeddings require a nonempty array of inputs");
    const auto prompt = request.value("prompt_name", std::string{});
    const bool length_only = request.value("length_only", false);
    if (length_only && !text_input)
        throw std::invalid_argument("length_only requires text inputs");
    const bool return_sparse = request.value("return_sparse", false);
    if (return_sparse && !sparse_head)
        throw std::invalid_argument("this embedding pack does not provide GTE sparse output");
    std::vector<Json> sparse;
    npue::enc::EmbedService::InspectRows inspect;
    if (return_sparse) {
        sparse.resize(inputs.size());
        inspect = [&](size_t index, std::span<const float> hidden, std::span<const int32_t> ids) {
            auto scores = sparse_head->score(hidden, ids);
            std::vector<int32_t> indices;
            std::vector<float> values;
            indices.reserve(scores.size());
            values.reserve(scores.size());
            for (const auto& [id, score] : scores) {
                indices.push_back(id);
                values.push_back(score);
            }
            sparse[index] = {{"indices", std::move(indices)}, {"values", std::move(values)}};
        };
    }
    const auto width = static_cast<size_t>(model.hidden());
    const bool nomic = width == 768 && is_nomic_v15(model);
    const bool gte = width == 768 && model.source_repo() == "Alibaba-NLP/gte-multilingual-base";
    auto dimensions = width;
    if (request.contains("dimensions")) {
        const auto& requested = request.at("dimensions");
        const bool nomic_width = nomic && (requested == 64 || requested == 128 ||
                                           requested == 256 || requested == 512);
        const bool gte_width = gte && requested >= 128 && requested <= width;
        if (!requested.is_number_integer() || (requested != width && !nomic_width && !gte_width)) {
            const auto supported = nomic ? "64, 128, 256, 512 or 768" :
                gte ? "128 through 768" : std::to_string(width);
            throw std::invalid_argument("supported dimensions are " + supported);
        }
        dimensions = requested.get<size_t>();
    }

    int64_t tokens = 0;
    std::vector<float> values;
    if (text_input) {
        const auto names = model.prompt_names();
        if (prompt.empty() && !names.empty())
            throw std::invalid_argument("prompt_name is required. Choose from " + npue::enc::join_names(names));
        if (!prompt.empty() && std::find(names.begin(), names.end(), prompt) == names.end())
            throw std::invalid_argument("unsupported prompt_name '" + prompt + "'");
        const auto texts = inputs.get<std::vector<std::string>>();
        if (length_only) return {{"ok", true}, {"fits", model.fits_texts(texts, prompt)}};
        values = model.embed(texts, prompt, &tokens, inspect);
    } else {
        if (!prompt.empty())
            throw std::invalid_argument("include the task prefix in token_ids when tokenizing");
        std::vector<std::vector<int32_t>> token_ids;
        token_ids.reserve(inputs.size());
        for (const auto& row : inputs) {
            if (!row.is_array())
                throw std::invalid_argument("token_ids must contain token sequences");
            auto& ids = token_ids.emplace_back();
            ids.reserve(row.size());
            for (const auto& token : row) {
                if (!token.is_number_integer() || token < 0 ||
                    token > (std::numeric_limits<int32_t>::max)())
                    throw std::invalid_argument("token IDs must be nonnegative int32 values");
                ids.push_back(token.get<int32_t>());
            }
        }
        values = model.embed_token_ids(token_ids, &tokens, inspect);
    }
    if (values.size() != inputs.size() * width ||
        !std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); }))
        throw std::runtime_error("encoder returned an invalid embedding");
    auto vectors = Json::array();
    for (size_t offset = 0; offset < values.size(); offset += width) {
        const float* row = values.data() + offset;
        if (dimensions == width) {
            vectors.push_back(std::vector<float>(row, row + width));
            continue;
        }
        vectors.push_back(nomic ? npue::enc::nomic_matryoshka({row, width}, dimensions) :
                                 npue::enc::normalize_embedding_prefix({row, width}, dimensions));
    }
    Json output = {{"ok", true}, {"tokens", tokens}, {"vectors", std::move(vectors)}};
    if (return_sparse) output["sparse_vectors"] = std::move(sparse);
    return output;
}
}

int main(int argc, char** argv) {
    // NPUE uses printf for diagnostics. Redirect the original stream, and use
    // the saved pipe for JSON records so the client sees no startup text.
    const int protocol_fd = _dup(_fileno(stdout));
    if (protocol_fd == -1) return 1;
    std::unique_ptr<std::FILE, decltype(&std::fclose)> pipe(_fdopen(protocol_fd, "w"), std::fclose);
    if (!pipe) {
        _close(protocol_fd);
        return 1;
    }
    if (_dup2(_fileno(stderr), _fileno(stdout)) != 0) return 1;
    auto* protocol = pipe.get();
    try {
        if (argc != 3 && argc != 4)
            throw std::invalid_argument("usage: npue-worker PACK DESIGN [ATTENTION-DIRECTORY]");
        std::optional<npue::npu::BertAttentionFiles> files;
        if (argc == 4) files = npue::npu::load_bert_attention(argv[3]);
        std::unique_ptr<npue::npu::BertAttention> attention;
        npue::enc::Embedder model({argv[1], argv[2], 12, 1});
        std::optional<npue::enc::GteSparseHead> sparse;
        const auto& packed = model.packed_model();
        if (model.source_repo() == "Alibaba-NLP/gte-multilingual-base" &&
            (packed.has("sparse.weight") || packed.has("sparse.bias")))
            sparse.emplace(packed, model.unigram_special_ids());
        if (is_nomic_v15(model)) model.enable_nomic_dynamic_rope();
        if (files) {
            if (files->sequence != model.seq() || files->heads != npue::enc::g_heads ||
                files->head_width != npue::enc::g_head_dim)
                throw std::runtime_error("BERT attention geometry disagrees with the encoder");
            attention = std::make_unique<npue::npu::BertAttention>(
                files->image, files->instructions, model.seq(), files->heads, files->head_width,
                npue::enc::bf16_fill, npue::enc::bf16_read);
            model.set_attention_probe(std::ref(*attention));
            files.reset();
        }
        reply(protocol, {{"ok", true}, {"protocol", 1}, {"dimensions", model.hidden()},
                         {"context", model.seq()}, {"npu_attention", bool(attention)},
                         {"prompt_names", model.prompt_names()}, {"sparse", bool(sparse)},
                         {"length_probe", true}});
        for (std::string line; std::getline(std::cin, line);) {
            try {
                reply(protocol, embed(model, Json::parse(line), sparse));
            } catch (const npue::InputTooLong& error) {
                const auto message = "input " + std::to_string(error.index) + " is " +
                    std::to_string(error.n_tokens) + " tokens. The maximum is " +
                    std::to_string(error.limit) + " tokens.";
                reply(protocol, {{"ok", false}, {"invalid", true}, {"error", message}});
            } catch (const Json::exception& error) {
                reply(protocol, {{"ok", false}, {"invalid", true}, {"error", error.what()}});
            } catch (const std::invalid_argument& error) {
                reply(protocol, {{"ok", false}, {"invalid", true}, {"error", error.what()}});
            }
        }
    } catch (const std::exception& error) {
        try {
            reply(protocol, {{"ok", false}, {"error", error.what()}});
        } catch (const std::exception&) {
            std::fprintf(stderr, "%s\n", error.what());
        }
        return 1;
    }
    return 0;
}
