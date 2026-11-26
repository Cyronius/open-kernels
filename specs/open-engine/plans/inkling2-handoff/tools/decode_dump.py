"""Save prefill and decode logits of one bundle with stage times and process memory, for a bundle-to-bundle gate (2026-11-25)."""
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

import numpy as np
import psutil

repo = next(p for p in Path(__file__).resolve().parents if (p / 'AGENTS.md').is_file())
sys.path.insert(0, str(repo / 'benchmarks/reference/inkling2-prefill'))
from run import bridge_header, device_lock, read_profile, sha  # noqa: E402

VOCAB = 200058
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('output', type=Path)
ap.add_argument('--bridge', type=Path, required=True)
ap.add_argument('--model-dir', type=Path, required=True)
ap.add_argument('--bundle', type=Path, required=True)
ap.add_argument('--ids', type=Path, required=True)
ap.add_argument('--prefix', type=int, default=896)
ap.add_argument('--positions', type=int, default=256)
ap.add_argument('--threads', type=int, default=6)
args = ap.parse_args()
bridge, out = args.bridge.resolve(), args.output.resolve()
ids = list(map(int, args.ids.read_text(encoding='utf-8-sig').split()))
P, N = args.prefix, args.positions
assert P + N <= len(ids), 'not enough token IDs'
env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OMP_WAIT_POLICY='PASSIVE', INKLING2_OLD_QUERY_GROUPS='0')
env['PATH'] = r'C:\Windows\System32\AMD' + os.pathsep + env['PATH']
out.mkdir(parents=True)


def reply(process):
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(f'bridge closed its pipe ({process.poll()})')
    header = bridge_header(line)
    size = header.get('logits_f32_le_bytes', 0)
    payload = process.stdout.read(size)
    if len(payload) != size:
        raise ValueError('truncated logits')
    return header, np.frombuffer(payload, dtype='<f4')


def send(process, command, **fields):
    process.stdin.write((json.dumps(dict(command=command, **fields)) + '\n').encode())
    process.stdin.flush()
    return reply(process)


lock = Path(env.get('PHLEGM_NPU_LOCK', Path(tempfile.gettempdir()) / 'phlegm-npu.lock'))
memory, walls = {}, []
with device_lock(lock), (out / 'bridge.stderr.log').open('wb') as errors, subprocess.Popen(
        [bridge], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors) as process:
    try:
        meter = psutil.Process(process.pid)

        def sample(name):
            info = meter.memory_info()
            memory[name] = dict(private_gib=info.private / 2**30, working_set_gib=info.rss / 2**30)

        assert reply(process)[0].get('protocol') == 2
        send(process, 'init', model_dir=str(args.model_dir.resolve()), kernel_dir=str(args.bundle.resolve()),
             max_ctx=math.ceil((P + N + 8) / 256) * 256, prefill_route='block', prefill_mode='fast',
             dispatch_timeout_ms=60000)
        send(process, 'reset', dspark=False)
        sample('loaded')
        clock = time.perf_counter()
        _, prefill = send(process, 'prefill_logits', prompt=ids[:P])
        prefill_seconds = time.perf_counter() - clock
        sample('after_prefill')
        prefill.tofile(out / 'prefill_logits.f32')
        decode = np.lib.format.open_memmap(out / 'decode_logits.npy', mode='w+', dtype=np.float32, shape=(N, VOCAB))
        for i in range(N):
            clock = time.perf_counter()
            _, decode[i] = send(process, 'decode_logits', token=ids[P + i])
            walls.append((time.perf_counter() - clock) * 1e3)
            if i == 0:
                sample('after_first_decode')
        sample('end')
        decode.flush()
        send(process, 'quit')
        process.wait(timeout=15)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

profile = read_profile(out / 'bridge.stderr.log')
tokens = [row for row in profile if row['rows'] == 1 and row['position'] >= P][8:]
blocks = [row for row in profile if row['rows'] > 1]
stages = sorted(key for key in tokens[0] if key.endswith('_ms'))
meta = json.loads(bridge.with_suffix('.build.json').read_text(encoding='utf-8-sig'))
result = dict(bridge=meta['binary_sha256'], manifest=sha(args.bundle / 'manifest.json'), ids=sha(args.ids), prefix=P,
              positions=N, threads=args.threads, prefill_seconds=prefill_seconds, memory=memory,
              decode_wall_ms_median=statistics.median(walls[8:]),
              decode_stage_ms_median={key: statistics.median(row[key] for row in tokens) for key in stages},
              block_ms=[row['block_ms'] for row in blocks],
              block_weight_prepare_ms=[row.get('weight_prepare_ms', 0.0) for row in blocks])
(out / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(prefill_seconds=round(prefill_seconds, 2), decode_ms=round(result['decode_wall_ms_median'], 1),
                      block_ms=[round(v) for v in result['block_ms']],
                      weight_prepare_ms=[round(v) for v in result['block_weight_prepare_ms']],
                      memory={k: round(v['private_gib'], 2) for k, v in memory.items()})), flush=True)
