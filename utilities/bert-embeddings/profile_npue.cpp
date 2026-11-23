#include "npue_encoder.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include "bert_attention.hpp"

int main(int argc, char **argv) {
    try {
        if (argc < 6 || (argc - 6) % 2)
            throw std::runtime_error("usage: profile PACK DESIGN FIXTURE PREFIX OUTPUT [FIXTURE OUTPUT ...]");
        npue::File model(argv[1]);
        npue::enc::ShapeLease lease(model);
        npue::npu::Device device;
        npue::enc::StackOptions options;
        options.threads = 12;
        options.lanes = 1;
        npue::enc::Stack stack(device, model, argv[2], options);
        std::unique_ptr<npue::npu::BertAttention> attention;
        if (const auto* path = std::getenv("NPUE_BERT_ATTENTION_PROBE")) {
            if (npue::enc::g_heads != 12 || npue::enc::g_head_dim != 64)
                throw std::runtime_error("BERT NPU attention requires 12 heads of width 64");
            attention = std::make_unique<npue::npu::BertAttention>(path, npue::enc::g_seq, npue::enc::bf16_fill, npue::enc::bf16_read);
            stack.lead->attention_probe = std::ref(*attention);
        }
        npue::enc::EmbedService service{
            npue::enc::load_tokenizer(model, argv[1]),
            model.raw("embeddings.word").as<float>(),
            model.raw("embeddings.position").as<float>(),
            model.raw("embeddings.token_type").as<float>(),
            stack.lead.get(), stack.all, stack.batch};
        const auto run_case = [&](const char* input, const char* result, bool warm) {
          std::ifstream fixture(input, std::ios::binary);
          if (!fixture) throw std::runtime_error("could not open fixture");
          const std::string text{std::istreambuf_iterator<char>(fixture), {}};
          for (int rep = warm ? -1 : 0; rep < 1; ++rep) {
            for (auto *lane : stack.all) lane->reset_timers();
            if (attention) attention->io_ms = attention->dispatch_ms = 0;
            const auto start = npue::enc::now_s();
            int64_t token_count = 0;
            auto embedding = service.embed({text}, argv[4], &token_count);
            if (embedding.empty()) throw std::runtime_error("empty vector");
            if (rep < 0) continue;
            std::ofstream output(result, std::ios::binary);
            output.write(reinterpret_cast<const char*>(embedding.data()), embedding.size() * sizeof(float));
            if (!output) throw std::runtime_error("could not write embedding");
            const auto &e = *stack.lead;
            std::cout << "PROFILE tokens=" << token_count << " width=" << embedding.size()
                      << " dispatches=" << e.n_dispatch
                      << " wall_ms=" << (npue::enc::now_s() - start) * 1000
                      << " npu_dispatch_ms=" << e.t_disp * 1000
                      << " npu_sync_ms=" << (e.t_in + e.t_out) * 1000
                      << " convert_bias_ms=" << (e.t_conv + e.t_bias) * 1000
                      << " host_attention_ms=" << e.t_attn * 1000
                      << " host_softmax_ms=" << e.t_hostsm * 1000
                      << " host_norm_ms=" << e.t_hostln * 1000
                      << " host_gelu_ms=" << e.t_hostgelu * 1000
                      << " attention_io_ms=" << (attention ? attention->io_ms : 0)
                      << " attention_npu_ms=" << (attention ? attention->dispatch_ms : 0) << std::endl;
          }
        };
        run_case(argv[3], argv[5], true);
        for (int index = 6; index < argc; index += 2)
            run_case(argv[index], argv[index + 1], false);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
