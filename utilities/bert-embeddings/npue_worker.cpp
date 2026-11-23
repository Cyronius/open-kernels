// SPDX-License-Identifier: MIT
#include "npue_encoder.hpp"
#include "bert_attention.hpp"
#include "nomic_matryoshka.hpp"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
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

Json embed(npue::enc::Embedder& model, const Json& request) {
    if (!request.is_object() || request.contains("token_ids"))
        throw std::invalid_argument("BERT embeddings require text inputs");
    const auto texts = request.at("texts").get<std::vector<std::string>>();
    if (texts.empty())
        throw std::invalid_argument("BERT embeddings require at least one input");
    const auto prompt = request.value("prompt_name", std::string{});
    const auto names = model.prompt_names();
    if (prompt.empty() && !names.empty())
        throw std::invalid_argument("prompt_name is required. Choose from " + npue::enc::join_names(names));
    if (!prompt.empty() && std::find(names.begin(), names.end(), prompt) == names.end())
        throw std::invalid_argument("unsupported prompt_name '" + prompt + "'");
    const auto width = static_cast<size_t>(model.hidden());
    auto dimensions = width;
    if (request.contains("dimensions")) {
        const auto& requested = request.at("dimensions");
        const bool resizable = width == 768 && is_nomic_v15(model);
        if (!requested.is_number_integer() ||
            (requested != width &&
             !(resizable && (requested == 64 || requested == 128 ||
                             requested == 256 || requested == 512)))) {
            const auto supported = resizable ? "64, 128, 256, 512 or 768" : std::to_string(width);
            throw std::invalid_argument("supported dimensions are " + supported);
        }
        dimensions = requested.get<size_t>();
    }

    int64_t tokens = 0;
    auto values = model.embed(texts, prompt, &tokens);
    if (values.size() != texts.size() * width ||
        !std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); }))
        throw std::runtime_error("encoder returned an invalid embedding");
    auto vectors = Json::array();
    for (size_t offset = 0; offset < values.size(); offset += width) {
        const float* row = values.data() + offset;
        if (dimensions == width) {
            vectors.push_back(std::vector<float>(row, row + width));
            continue;
        }
        vectors.push_back(npue::enc::nomic_matryoshka({row, width}, dimensions));
    }
    return {{"ok", true}, {"tokens", tokens}, {"vectors", std::move(vectors)}};
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
            throw std::invalid_argument("usage: npue-worker PACK DESIGN [VERIFIED-ATTENTION-DIRECTORY]");
        std::unique_ptr<npue::npu::BertAttention> attention;
        npue::enc::Embedder model({argv[1], argv[2], 12, 1});
        if (is_nomic_v15(model)) model.enable_nomic_dynamic_rope();
        if (argc == 4) {
            attention = std::make_unique<npue::npu::BertAttention>(
                argv[3], model.seq(), npue::enc::bf16_fill, npue::enc::bf16_read);
            model.set_attention_probe(std::ref(*attention));
        }
        reply(protocol, {{"ok", true}, {"protocol", 1}, {"dimensions", model.hidden()},
                         {"context", model.seq()}, {"npu_attention", bool(attention)},
                         {"prompt_names", model.prompt_names()}});
        for (std::string line; std::getline(std::cin, line);) {
            try {
                reply(protocol, embed(model, Json::parse(line)));
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
