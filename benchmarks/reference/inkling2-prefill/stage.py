"""Prepare an isolated engine source tree with Inkling 2 Mini prefill timings."""

import argparse
from pathlib import Path
import shutil

work = Path(__file__).resolve().parent
repo = work.parents[2]


def change(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"expected exactly one profiling anchor: {before!r}")
    return text.replace(before, after, 1)


def change_all(text, before, after):
    if before not in text:
        raise ValueError(f"missing profiling anchor: {before!r}")
    return text.replace(before, after)


def insert(text, before, line):
    return change(text, before, before + '\n' + line)


def load(src, name):
    s = (src / name).read_text(encoding='utf-8')
    s = '#include "inkling2_prefill_profile.hpp"\n' + s
    return insert(s, 'namespace open_qwen36 {\nnamespace {\n', 'namespace perf = inkling_prefill_profile;')



def instrument_host(src):
    host = load(src, 'inkling2_host.cpp')
    host = insert(host, 'const std::vector<float>& query, float eps) {', '    perf::Measure measure{perf::residual_mix};')
    host = insert(host, 'std::vector<float>* tape) const {', '        perf::Measure measure{perf::conv};')
    host = insert(host, 'void apply(Rows& x, const Rows& message, float eps) const {', '        perf::Measure measure{perf::controller};')
    host = insert(host, '        const Rows g = gain.apply(z), f = forget.apply(z);',
                  '        perf::Measure update{perf::controller_update};')
    gate = '        for (long long j = 0; j < gated; ++j) a.data[j] *= sigmoid(gate.data[j]);'
    host = change(host, '        const long long gated', '        const auto gate_start = perf::Clock::now();\n        const long long gated')
    host = change(host, gate, gate + '\n'
                  '        perf::add(perf::attention_gate, gate_start);')
    for line in ('            norm(u, eps, l.attn_norm[n]);', '        norm(f, eps, l.ffn_norm);'):
        spaces = line[:len(line) - len(line.lstrip())]
        host = change(host, line, f'{spaces}const auto norm_start = perf::Clock::now();\n{line}\n'
                      f'{spaces}perf::add(perf::layer_norm, norm_start);')
    host = change(host, '        for (size_t t = 0; t < x.count(); ++t) {\n'
                  '            const float tau = l.sliding ? 1.0f :',
                  '        const auto qkv_start = perf::Clock::now();\n'
                  '        for (size_t t = 0; t < x.count(); ++t) {\n'
                  '            const float tau = l.sliding ? 1.0f :')
    host = insert(host, '        encode_kv(v, next.value);', '        perf::add(perf::qkv_post, qkv_start);')
    host = change(host, '            a = attend(l, q, r, sinks, pos, state, next);',
                  '            perf::Measure measure{perf::attention_host};\n'
                  '            a = attend(l, q, r, sinks, pos, state, next);')
    host = insert(host, '    Rows embed(const std::vector<int>& ids) const {', '        perf::Measure measure{perf::embed};')
    host = insert(host, '    std::vector<float> logits_for(const Rows& normalized) const {',
                  '        perf::Measure measure{perf::head};')
    host = change(host, '        if (logits.empty()) logits = lm_head.apply(normalized, 0, vocab).data;',
                  '        if (logits.empty()) {\n'
                  '            perf::Measure host_head{perf::head_host};\n'
                  '            logits = lm_head.apply(normalized, 0, vocab).data;\n'
                  '        }')
    host = insert(host, '    void commit(PendingBlock& pending, size_t count) {',
                  '        perf::Measure measure{perf::kv_commit};')
    host = insert(host, '                                       const std::array<Rows*, 6>& output) const {',
                  '        perf::Measure measure{perf::dense_host_input};')
    host = change(host, '            attention_output = host_projection(l.output, a);',
                  '            perf::Measure measure{perf::dense_host_output};\n'
                  '            attention_output = host_projection(l.output, a);')
    host = insert(host, 'size_t layer, Inkling2ExpertProjector* projector, bool verify = false) const {',
                  '        perf::Measure measure{perf::moe_total};\n        const auto route_start = perf::Clock::now();')
    host = change(host, '        Rows out(x.count(), hidden);\n        if (projector',
                  '        perf::add(perf::route, route_start);\n        Rows out(x.count(), hidden);\n        if (projector')
    sync = host.index('        if (block_rows) {')
    overlap, host = host[:sync], host[sync:]
    host = insert(host, '            while (expert < experts) {', '                const auto gather_start = perf::Clock::now();')
    host = insert(host, '                if (!slots) break;', '                perf::add(perf::gather, gather_start);')
    host = insert(host, '                    fail("grouped expert gate/up projection is unavailable");',
                  '                const auto situ_start = perf::Clock::now();')
    host = change(host, '                if (!projector->project_grouped_blocks(*down.file',
                  '                perf::add(perf::situ, situ_start);\n                if (!projector->project_grouped_blocks(*down.file')
    host = insert(host, '                    fail("grouped expert down projection is unavailable");',
                  '                const auto reduce_start = perf::Clock::now();')
    host = insert(host, '                        for (size_t j = 0; j < hidden; ++j) out[route.token][j] += route.weight * value[j];\n                    }',
                  '                perf::add(perf::reduce, reduce_start);')
    host = overlap + host
    if 'projector->grouped_overlap()' in host:
        host = instrument_host_overlap(host)
    host = change(host, '    return impl_->append(ids, logits, observe, draft_features);',
                  '    perf::reset();\n'
                  '    const auto start = perf::Clock::now();\n'
                  '    const auto position = impl_->pos;\n'
                  '    auto result = impl_->append(ids, logits, observe, draft_features);\n'
                  '    perf::add(perf::block, start);\n'
                  '    perf::print(position, ids.size());\n'
                  '    return result;')
    host = change(host, '    return impl_->verify(ids, draft_features);',
                  '    perf::reset();\n'
                  '    const auto start = perf::Clock::now();\n'
                  '    const auto position = impl_->pos;\n'
                  '    auto result = impl_->verify(ids, draft_features);\n'
                  '    perf::add(perf::block, start);\n'
                  '    perf::print(position, ids.size(), true);\n'
                  '    return result;')
    return host


def instrument_host_overlap(host):
    for name, stage in (('stage_up', 'gather'), ('finish_up', 'situ'), ('finish_down', 'reduce')):
        start = host.index(f'            const auto {name} = [&](size_t g')
        end = host.index('\n            };', start)
        section = host[start:end]
        before = '                for (size_t slot'
        measured = change(section, before, '                const auto part = perf::Clock::now();\n' + before)
        finish = f'                perf::add(perf::{stage}, part);'
        if name == 'finish_down':
            measured += '\n' + finish
        else:
            after = '                projector->stage_grouped('
            measured = change(measured, after, finish + '\n' + after)
        host = host[:start] + measured + host[end:]
    return host


def instrument_npu(src):
    npu = load(src, 'inkling2_npu.cpp')
    has_overlap = 'void wait_grouped() override' in npu
    if 'size_t project_wide_old(' in npu:
        npu = change(npu, '        try {\n            wide.packed_keys.prepare(',
                     '        auto old_stage = perf::Clock::now();\n        try {\n            wide.packed_keys.prepare(')
        npu = insert(npu, '        wide.upload_query(view);', '        perf::add(perf::attention_old_prepare, old_stage);')
        npu = change(npu, '            wide.upload_cache(view, first, count);',
                     '            old_stage = perf::Clock::now();\n'
                     '            wide.upload_cache(view, first, count);\n'
                     '            perf::add(perf::attention_old_upload, old_stage);\n'
                     '            old_stage = perf::Clock::now();')
        npu = insert(npu, '                                         wide.output, Wide::output_values, "wide older-key attention");',
                     '            perf::add(large ? perf::attention_old_dispatch_16k : perf::attention_old_dispatch_4k, old_stage);\n'
                     '            perf::Measure merge{perf::attention_old_merge};')
    npu = insert(npu, 'project_attention(const Inkling2AttentionView& view) override {', '        perf::Measure measure{perf::attention};')
    if 'std::vector<float> project_near(' in npu:
        anchor = '        const auto& strips = attention.layer_strips(device_, view);'
        npu = change(npu, anchor, '        auto near_stage = perf::Clock::now();\n' + anchor)
        anchor = '            const auto* state = dispatch(attention.kernel, attention.insts,'
        npu = change(npu, anchor, '            perf::add(perf::attention_near_prepare, near_stage);\n'
                                  '            near_stage = perf::Clock::now();\n' + anchor)
        npu = insert(npu, '                                         "near attention");',
                     '            perf::add(perf::attention_near_dispatch, near_stage);\n'
                     '            near_stage = perf::Clock::now();')
        npu = insert(npu, '            Near::merge(state, out.data(), peaks.data(), masses.data(), view.rows);',
                     '            perf::add(perf::attention_near_merge, near_stage);\n'
                     '            near_stage = perf::Clock::now();')
    npu = insert(npu, 'const auto upload_keys = [&](size_t position, size_t first, size_t last, const Tier& tier) {',
                 '            perf::Measure kv{perf::attention_kv};')
    npu = insert(npu, 'const auto attend_tile = [&](size_t token, size_t rows, size_t first, size_t last, bool sink, const Tier& tier) {',
                 '            const auto query_start = perf::Clock::now();')
    npu = insert(npu, '            attn.query.sync(XCL_BO_SYNC_BO_TO_DEVICE, query_size, 0);',
                 '            perf::add(perf::attention_query, query_start);\n'
                 '            const auto dispatch_start = perf::Clock::now();')
    npu = insert(npu, '                                           kv_heads * mode.output_column, "relative attention");',
                 '            perf::add(view.window ? perf::attention_sliding_dispatch : perf::attention_full_dispatch,\n'
                 '                      dispatch_start);\n'
                 '            perf::Measure merge{perf::attention_merge};')
    npu = insert(npu, 'const Inkling2AttentionProjections& out) override {', '        perf::Measure measure{perf::dense_input};')
    npu = insert(npu, 'const float* input,\n                                  float* output) override {', '        perf::Measure measure{perf::dense_output};')
    group_signature = 'ExpertIds ids, const ExpertRows& counts, const float* input, float* output'
    split_scale = ', bool split_scale = false'
    if group_signature + split_scale in npu:
        group_signature += split_scale
    npu = insert(npu, group_signature + ') {', '        perf::Group measure;')
    npu = insert(npu, '            block_layout::pack_input(input + slot * slot_rows * cols, counts[slot], tile_m, cols,\n                                     activation + slot * a_bytes);',
                 '            perf::add(perf::activation_pack, pack_start);')
    for rows in (('slot_rows', 'grouped_rows_') if has_overlap else ('slot_rows',)):
        call = f'            block_layout::pack_input(input + slot * {rows} * cols, counts[slot], tile_m, cols,'
        at = npu.index(call)
        start = npu.rindex('            if (counts[slot]) pack_compact_expert', 0, at)
        npu = (npu[:start] + '            const auto weight_start = perf::Clock::now();\n' + npu[start:at]
               + '            perf::add(perf::weight_prepare, weight_start);\n'
               + '            const auto pack_start = perf::Clock::now();\n' + npu[at:])
    npu = change(npu, '        input_.sync(XCL_BO_SYNC_BO_TO_DEVICE, slots * a_bytes, 0);',
                 '        auto stage = perf::Clock::now();\n'
                 '        input_.sync(XCL_BO_SYNC_BO_TO_DEVICE, slots * a_bytes, 0);\n'
                 '        perf::add(perf::activation_sync, stage);\n'
                 '        stage = perf::Clock::now();')
    npu = change(npu, '        const auto* result = dispatch(format.kernel,',
                 '        perf::add(perf::instruction_sync, stage);\n        const auto* result = dispatch(format.kernel,')
    npu = npu.replace('        block_layout::unpack_grouped_output(',
                      '        perf::Measure unpack{perf::group_unpack};\n        block_layout::unpack_grouped_output(')
    npu = change(npu, '        xrt::run run(kernel);', '        auto stage = perf::Clock::now();\n        xrt::run run(kernel);')
    npu = insert(npu, '        failed_ = true;\n        run.start();',
                 '        if (perf::grouped) perf::add(perf::configure, stage);\n        stage = perf::Clock::now();')
    npu = insert(npu, '        const auto state = run.wait(timeout_);', '        if (perf::grouped) perf::add(perf::wait, stage);')
    npu = change(npu, '        output.sync(XCL_BO_SYNC_BO_FROM_DEVICE, rows * sizeof(float), 0);',
                 '        stage = perf::Clock::now();\n'
                 '        output.sync(XCL_BO_SYNC_BO_FROM_DEVICE, rows * sizeof(float), 0);\n'
                 '        if (perf::grouped) perf::add(perf::readback_sync, stage);\n'
                 '        stage = perf::Clock::now();')
    npu = insert(npu, '            fail("non-finite output from " + tensor);', '        if (perf::grouped) perf::add(perf::finite_scan, stage);')
    return instrument_npu_overlap(npu) if has_overlap else npu


def instrument_npu_overlap(npu):
    npu = insert(npu, '            block_layout::pack_input(input + slot * grouped_rows_ * cols, counts[slot], tile_m, cols,\n                                     activation + slot * a_bytes);',
                 '            perf::add(perf::activation_pack, pack_start);')
    for signature in ('const ExpertIds& group, const ExpertRows& counts, const float* input) override {',
                      'void start_grouped(bool down, size_t index) override {',
                      'void wait_grouped() override {',
                      'void read_grouped(bool down, size_t index, const ExpertRows& counts, float* output) override {'):
        npu = insert(npu, signature, '        perf::GroupInterface measure;')
    npu = insert(npu, '        launch(lanes_[(down ? 2 : 0) + index]);', '        ++perf::calls[perf::group_total];')
    for line, stage, condition in (
        ('        lane.input.sync(XCL_BO_SYNC_BO_TO_DEVICE, slots * a_bytes, 0);', 'activation_sync', 'true'),
        ('            lane.instruction.sync(XCL_BO_SYNC_BO_TO_DEVICE);', 'instruction_sync', 'true'),
        ('        const auto state = overlap_run_.wait(timeout_);', 'wait', 'lane.expert'),
        ('        const bool finite = all_finite(result, lane.floats);', 'finite_scan', 'lane.expert'),
    ):
        indent = line[:len(line) - len(line.lstrip())]
        prefix = '' if condition == 'true' else f'if ({condition}) '
        npu = change_all(npu, line, f'{indent}const auto profile_start = perf::Clock::now();\n{line}\n'
                     f'{indent}{prefix}perf::add(perf::{stage}, profile_start);')
    npu = change(npu, '        xrt::run next(*lane.kernel);',
                 '        const auto configure_start = perf::Clock::now();\n        xrt::run next(*lane.kernel);')
    npu = insert(npu, '        overlap_run_.start();',
                 '        if (lane.expert) perf::add(perf::configure, configure_start);')
    npu = change(npu, '        lane.output.sync(XCL_BO_SYNC_BO_FROM_DEVICE, lane.floats * sizeof(float), 0);',
                 '        const auto readback_start = perf::Clock::now();\n'
                 '        lane.output.sync(XCL_BO_SYNC_BO_FROM_DEVICE, lane.floats * sizeof(float), 0);')
    npu = insert(npu, '        read_fence();', '        if (lane.expert) perf::add(perf::readback_sync, readback_start);')
    return npu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--engine-root', type=Path, default=repo)
    args = parser.parse_args()
    root = args.engine_root.resolve()
    dst = args.output.resolve()
    src = root / 'src/open_qwen36'
    engine = dst / 'src/open_qwen36'

    # Resolve all anchors before creating the staged tree.
    host = instrument_host(src)
    npu = instrument_npu(src)
    dst.mkdir(parents=True)
    for p in src.rglob('*'):
        if p.is_file() and p.suffix in ('.cpp', '.hpp'):
            output = engine / p.relative_to(src)
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, output)
    shutil.copytree(root / 'src/include/nlohmann', dst / 'src/include/nlohmann')
    harness = dst / 'open_kernels/harness'
    harness.mkdir(parents=True)
    shutil.copyfile(root / 'open_kernels/harness/stream_patch.hpp', harness / 'stream_patch.hpp')
    shutil.copyfile(work / 'profile.hpp', engine / 'inkling2_prefill_profile.hpp')
    (engine / 'inkling2_host.cpp').write_text(host, encoding='utf-8', newline='\n')
    (engine / 'inkling2_npu.cpp').write_text(npu, encoding='utf-8', newline='\n')
    print(dst)


if __name__ == '__main__':
    main()
