"""Time plain decode and verify passes of 1 to N rows through the profiled Phlegm bridge (2026-11-24)."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

repo = next(p for p in Path(__file__).resolve().parents if (p / 'AGENTS.md').is_file())
sys.path.insert(0, str(repo / 'benchmarks/reference/inkling2-prefill'))
from run import bridge_header, device_lock, read_profile, sha  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('output', type=Path)
ap.add_argument('--bridge', type=Path, required=True)
ap.add_argument('--model-dir', type=Path, required=True)
ap.add_argument('--bundle', type=Path, required=True)
ap.add_argument('--ids', type=Path, required=True)
ap.add_argument('--prefix', type=int, default=512)
ap.add_argument('--cycles', type=int, default=6)
ap.add_argument('--max-rows', type=int, default=4)
ap.add_argument('--blocks', type=int, nargs='*', default=[8, 16, 32, 64], help='block sizes to time as appends')
ap.add_argument('--dspark', action='store_true', help='reset with the drafter so the bridge accepts up to 8 verify rows')
ap.add_argument('--threads', type=int, default=6)
args = ap.parse_args()
bridge, out = args.bridge.resolve(), args.output.resolve()
ids = list(map(int, args.ids.read_text(encoding='utf-8-sig').split()))
schedule = [('decode', 1)] + [('verify', k) for k in range(1, args.max_rows + 1)]
need = args.prefix + args.cycles * sum(k for _, k in schedule) + 3 * sum(args.blocks)
assert need <= len(ids), 'not enough token IDs'
env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OMP_WAIT_POLICY='PASSIVE', INKLING2_OLD_QUERY_GROUPS='0')
env['PATH'] = r'C:\Windows\System32\AMD' + os.pathsep + env['PATH']
out.mkdir(parents=True)
calls = []


def reply(process):
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(f'bridge closed its pipe ({process.poll()})')
    header = bridge_header(line)
    size = header.get('logits_f32_le_bytes', 0)
    if len(process.stdout.read(size)) != size:
        raise ValueError('truncated logits')
    return header


def send(process, command, **fields):
    process.stdin.write((json.dumps(dict(command=command, **fields)) + '\n').encode())
    process.stdin.flush()
    return reply(process)


lock = Path(env.get('PHLEGM_NPU_LOCK', Path(tempfile.gettempdir()) / 'phlegm-npu.lock'))
with device_lock(lock), (out / 'bridge.stderr.log').open('wb') as errors, subprocess.Popen(
        [bridge], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors) as process:
    try:
        assert reply(process).get('protocol') == 2
        ready = send(process, 'init', model_dir=str(args.model_dir.resolve()), kernel_dir=str(args.bundle.resolve()),
                     max_ctx=math.ceil((need + 8) / 256) * 256, prefill_route='block', prefill_mode='fast',
                     dispatch_timeout_ms=60000)
        assert ready.get('verify_width', 0) >= args.max_rows, ready
        send(process, 'reset', dspark=args.dspark)
        send(process, 'prefill', prompt=ids[:256])
        send(process, 'reset', dspark=args.dspark)
        clock = time.perf_counter()
        send(process, 'prefill', prompt=ids[:args.prefix])
        prefill = time.perf_counter() - clock
        at = args.prefix
        for cycle in range(args.cycles):
            for kind, rows in schedule:
                clock = time.perf_counter()
                if kind == 'decode':
                    send(process, 'decode_logits', token=ids[at])
                    wall, accept = time.perf_counter() - clock, 0.0
                else:
                    send(process, 'verify_logits', tokens=ids[at:at + rows])
                    wall = time.perf_counter() - clock
                    clock = time.perf_counter()
                    send(process, 'accept', n=rows)
                    accept = time.perf_counter() - clock
                calls.append(dict(kind=kind, rows=rows, position=at, wall_ms=wall * 1e3, accept_ms=accept * 1e3))
                at += rows
        for rows in args.blocks:
            for _ in range(3):
                clock = time.perf_counter()
                send(process, 'prefill_chunk', prompt=ids[at:at + rows])
                calls.append(dict(kind='block', rows=rows, position=at, wall_ms=(time.perf_counter() - clock) * 1e3,
                                  accept_ms=0.0))
                at += rows
        send(process, 'quit')
        process.wait(timeout=15)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

profile = read_profile(out / 'bridge.stderr.log')
by_position = {(row['position'], row['rows'], row.get('verify', 0)): row for row in profile}
stages = ('block_ms', 'moe_total_ms', 'wait_ms', 'wait_calls', 'group_total_calls', 'dense_host_input_ms', 'dense_host_output_ms',
          'dense_input_ms', 'dense_output_ms', 'head_ms', 'head_calls', 'attention_ms', 'attention_kv_ms',
          'attention_sliding_dispatch_ms', 'attention_full_dispatch_ms', 'attention_near_dispatch_ms',
          'controller_ms', 'residual_mix_ms', 'conv_ms', 'situ_ms', 'gather_ms', 'reduce_ms')
summary = []
for kind, rows in schedule + [('block', rows) for rows in args.blocks]:
    timed = calls if kind == 'block' else calls[len(schedule):]  # the first cycle is a warm-up
    group = [c for c in timed if (c['kind'], c['rows']) == (kind, rows)]
    traces = [by_position[(c['position'], rows, int(kind == 'verify'))] for c in group] if profile else []
    line = dict(kind=kind, rows=rows, samples=len(group),
                wall_ms=statistics.median(c['wall_ms'] for c in group),
                accept_ms=statistics.median(c['accept_ms'] for c in group))
    line.update({name: statistics.median(t[name] for t in traces) for name in stages} if traces else {n: 0.0 for n in stages})
    line['named_ms'] = sum(line[n] for n in ('moe_total_ms', 'dense_host_input_ms', 'dense_host_output_ms',
                                               'dense_input_ms', 'dense_output_ms', 'head_ms', 'attention_ms',
                                               'controller_ms', 'residual_mix_ms', 'conv_ms'))
    summary.append(line)
    print(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in line.items()}), flush=True)
meta = json.loads(bridge.with_suffix('.build.json').read_text(encoding='utf-8-sig'))
result = dict(bridge=meta['binary_sha256'], manifest=sha(args.bundle / 'manifest.json'), ids=sha(args.ids),
              prefix=args.prefix, cycles=args.cycles, dspark=args.dspark, threads=args.threads,
              prefill_seconds=prefill, calls=calls, summary=summary)
(out / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
