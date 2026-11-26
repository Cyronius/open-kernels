"""Compare two decode_dump.py results: prefill bits, decode logits, stage times and memory (2026-11-25)."""
import json
import math
from pathlib import Path
import sys

import numpy as np

a_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
a, b = (json.loads((d / 'result.json').read_text(encoding='utf-8')) for d in (a_dir, b_dir))
assert (a['ids'], a['prefix'], a['positions']) == (b['ids'], b['prefix'], b['positions'])
pa, pb = (np.fromfile(d / 'prefill_logits.f32', dtype='<f4') for d in (a_dir, b_dir))
da, db = (np.load(d / 'decode_logits.npy', mmap_mode='r') for d in (a_dir, b_dir))


def log_softmax(row):
    x = row.astype(np.float64)
    x -= x.max()
    return x - math.log(np.exp(x).sum())


ids = list(map(int, Path(sys.argv[3]).read_text(encoding='utf-8-sig').split())) if len(sys.argv) > 3 else None
kl, same, flips, nll, right = [], 0, [], [0.0, 0.0], [0, 0]
for i in range(da.shape[0]):
    la, lb = log_softmax(da[i]), log_softmax(db[i])
    kl.append(float((np.exp(la) * (la - lb)).sum()))
    top_a, top_b = int(da[i].argmax()), int(db[i].argmax())
    same += top_a == top_b
    real = ids[a['prefix'] + i + 1] if ids and a['prefix'] + i + 1 < len(ids) else None
    if real is not None:
        nll[0] -= la[real]
        nll[1] -= lb[real]
        right[0] += top_a == real
        right[1] += top_b == real
    if top_a != top_b:
        order = np.sort(da[i])[-2:]
        flips.append(dict(position=a['prefix'] + i, a=top_a, b=top_b, a_margin=float(order[1] - order[0]),
                          kl=kl[-1], a_right=top_a == real, b_right=top_b == real))
count = len(kl)
by_quarter = [sum(kl[i * count // 4:(i + 1) * count // 4]) / (count // 4) for i in range(4)]
kl.sort()
report = dict(
    real_text=dict(mean_nll=[nll[0] / count, nll[1] / count], top1=right) if ids else None, kl_by_quarter=by_quarter,
    prefill_bit_equal=bool(np.array_equal(pa, pb)), prefill_max_abs=float(np.abs(pa - pb).max()),
    decode_bit_equal=bool(np.array_equal(da, db)), positions=int(da.shape[0]), same_top=same,
    kl_mean=sum(kl) / len(kl), kl_max=kl[-1], flips=flips,
    decode_ms=[a['decode_wall_ms_median'], b['decode_wall_ms_median']],
    prefill_seconds=[a['prefill_seconds'], b['prefill_seconds']],
    first_blocks_ms=[a['block_ms'][:4], b['block_ms'][:4]],
    private_gib={k: [a['memory'][k]['private_gib'], b['memory'][k]['private_gib']] for k in a['memory']},
    stages={k: [a['decode_stage_ms_median'][k], b['decode_stage_ms_median'].get(k, 0.0)]
            for k in a['decode_stage_ms_median']
            if max(a['decode_stage_ms_median'][k], b['decode_stage_ms_median'].get(k, 0.0)) >= 1.0})
print(json.dumps(report, indent=1, default=lambda v: round(v, 6)))
