"""Check that a partly accepted verify pass leaves the same state as plain decode, without the drafter (2026-11-25)."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

repo = next(p for p in Path(__file__).resolve().parents if (p / 'AGENTS.md').is_file())
sys.path.insert(0, str(repo / 'benchmarks/reference/inkling2-prefill'))
from run import bridge_header, device_lock  # noqa: E402

VOCAB = 200058
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--bridge', type=Path, required=True)
ap.add_argument('--model-dir', type=Path, required=True)
ap.add_argument('--bundle', type=Path, required=True)
ap.add_argument('--ids', type=Path, required=True)
ap.add_argument('--prefixes', type=int, nargs='+', default=[5, 1021])
ap.add_argument('--width', type=int, default=4)
ap.add_argument('--threads', type=int, default=6)
ap.add_argument('--reference', choices=('direct', 'trial'), default='direct',
                help='trial: wide passes use block arithmetic, so compare the changed suffix with a full trial')
args = ap.parse_args()
ids = list(map(int, args.ids.read_text(encoding='utf-8-sig').split()))
env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OMP_WAIT_POLICY='PASSIVE', INKLING2_OLD_QUERY_GROUPS='0')
env['PATH'] = r'C:\Windows\System32\AMD' + os.pathsep + env['PATH']
width, follow_count = args.width, 2
cases = list(range(width + 1))


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


def run_arm(process, prefix, arm):
    send(process, 'reset', dspark=False)
    send(process, 'prefill', prompt=ids[:prefix])
    position, output = prefix, []
    for case, accepted in enumerate(cases):
        start = 2048 + case * 32
        original = ids[start:start + width]
        changed = original[:accepted] + [(token + 1009) % VOCAB for token in original[accepted:]]
        follow = ids[start + width:start + width + follow_count]
        if arm == 'direct':
            kept = [send(process, 'decode_logits', token=token)[1] for token in original[:accepted]]
        else:
            header, rows = send(process, 'verify_logits', tokens=changed if arm == 'changed_suffix' else original)
            assert header['position'] == position, header
            kept = list(rows.reshape(width, VOCAB)[:accepted])
            header, _ = send(process, 'accept', n=accepted)
            assert header['position'] == position + accepted, header
        position += accepted
        after = [send(process, 'decode_logits', token=token)[1] for token in follow]
        position += follow_count
        output.append((np.array(kept, dtype=np.float32).reshape(accepted, VOCAB), np.array(after, dtype=np.float32)))
    return output


lock = Path(env.get('PHLEGM_NPU_LOCK', Path(tempfile.gettempdir()) / 'phlegm-npu.lock'))
failures = 0
with device_lock(lock), subprocess.Popen([args.bridge.resolve()], env=env, stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as process:
    try:
        assert reply(process)[0].get('protocol') == 2
        need = max(args.prefixes) + len(cases) * (width + follow_count) + 8
        ready, _ = send(process, 'init', model_dir=str(args.model_dir.resolve()), kernel_dir=str(args.bundle.resolve()),
                        max_ctx=math.ceil(need / 256) * 256, prefill_route='block', prefill_mode='fast',
                        dispatch_timeout_ms=60000)
        assert max(ready.get('verify_width', 0), ready.get('long_verify', {}).get('max_rows', 0)) >= width, ready
        for prefix in args.prefixes:
            direct = run_arm(process, prefix, 'direct')
            base = direct if args.reference == 'direct' else run_arm(process, prefix, 'trial')
            if args.reference == 'trial':
                for accepted, (d, w) in zip(cases, zip(direct, base)):
                    rows = np.concatenate([d[0], d[1]]).astype(np.float64), np.concatenate([w[0], w[1]]).astype(np.float64)
                    lp = [r - r.max(axis=1, keepdims=True) for r in rows]
                    lp = [x - np.log(np.exp(x).sum(axis=1, keepdims=True)) for x in lp]
                    kl = (np.exp(lp[0]) * (lp[0] - lp[1])).sum(axis=1)
                    print(json.dumps(dict(prefix=prefix, accepted=accepted, block_vs_decode_top_equal=int(
                        (rows[0].argmax(1) == rows[1].argmax(1)).sum()), rows=len(rows[0]),
                        max_kl=float(kl.max()))), flush=True)
            for arm in (('trial', 'changed_suffix') if args.reference == 'direct' else ('changed_suffix',)):
                result = run_arm(process, prefix, arm)
                for accepted, (reference, candidate) in zip(cases, zip(base, result)):
                    kept = reference[0].tobytes() == candidate[0].tobytes()
                    after = reference[1].tobytes() == candidate[1].tobytes()
                    failures += not (kept and after)
                    print(json.dumps(dict(prefix=prefix, arm=arm, accepted=accepted, accepted_rows_exact=kept,
                                          next_tokens_exact=after)), flush=True)
        send(process, 'quit')
        process.wait(timeout=15)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
print(json.dumps(dict(cases=len(args.prefixes) * (2 if args.reference == 'direct' else 1) * len(cases),
                      failures=failures)))
sys.exit(1 if failures else 0)
