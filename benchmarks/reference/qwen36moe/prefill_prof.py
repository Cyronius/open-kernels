# Sum OFLM_OPEN_PROFILE layer rows by layer kind and stage, per prompt length, with ms per token (2026-11-25).
import json
import sys
from collections import defaultdict

d, names = sys.argv[1], sys.argv[2:]
COLS = ["wall_ms", "gemm_ms", "host_ms", "per_token_ms", "attn_ms", "dn_conv_ms", "dn_rule_ms", "mid_ms", "prenorm_ms",
        "tail_ms", "state_ms", "sync_ms", "shared_ms", "qkv_scatter_ms", "moe_run_ms", "moe_read_ms", "moe_prep_ms"]
for n in names:
    req, sums = None, defaultdict(lambda: defaultdict(float))
    for line in open(f"{d}/{n}.jsonl"):
        r = json.loads(line)
        if r["record"] == "request" and r["what"] == "gemm-prompt":
            req = r
        elif r["record"] == "layer" and r["what"] == "gemm-prompt":
            for c in COLS:
                sums[r["kind"]][c] += r[c]
    t = req["tokens"]
    print(f"== {n}: {t} tokens, total {req['total_ms'] / 1000:.1f} s ({req['total_ms'] / t:.2f} ms/token), "
          f"gemm {req['gemm_ms'] / 1000:.1f} s, host {req['host_ms'] / 1000:.1f} s, per-token {req['per_token_ms'] / 1000:.1f} s")
    for kind, s in sums.items():
        print(f"  {kind:8s} " + "  ".join(f"{c[:-3]} {s[c] / t:.3f}" for c in COLS if s[c] > 0.0005 * t))
