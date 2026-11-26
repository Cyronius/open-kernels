# KL of open_qwen36_cli logits against a llama-perplexity --kl-divergence-base file, same tokens (2026-11-23).
# `ids`: write each chunk's tokens for the CLI. `score`: KL / top-token agreement on the scored half of each chunk,
# with llama-perplexity's own formula (base log probs below -16 skipped).
import argparse
import json
from pathlib import Path

import numpy as np


def read_base(path):
    with open(path, "rb") as f:
        assert f.read(8) == b"_logits_"
        n_ctx, n_vocab, n_chunk = np.frombuffer(f.read(12), "<i4")
        tokens = np.frombuffer(f.read(4 * n_chunk * n_ctx), "<i4").reshape(n_chunk, n_ctx)
        nv = 2 * ((n_vocab + 1) // 2) + 4
        first = n_ctx // 2
        n_tok = n_ctx - 1 - first
        recs = np.memmap(path, dtype="<u2", mode="r", offset=20 + 4 * n_chunk * n_ctx,
                         shape=(n_chunk, n_tok, nv))
    return int(n_ctx), int(n_vocab), int(n_chunk), tokens, recs, first


def base_logp(rec, n_vocab):
    scale, mn = np.frombuffer(rec[:4].tobytes(), "<f4")
    return scale * rec[4:4 + n_vocab].astype(np.float64) + mn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ids", "score"])
    ap.add_argument("--base", required=True)
    ap.add_argument("--dir", required=True, help="ids: where chunk<k>.txt go; score: dump dir with chunk<k>_p<pos>.bin")
    ap.add_argument("--chunks", type=int, default=0)
    ap.add_argument("--json")
    a = ap.parse_args()
    n_ctx, n_vocab, n_chunk, tokens, recs, first = read_base(a.base)
    nc = min(a.chunks or n_chunk, n_chunk)
    d = Path(a.dir)
    if a.mode == "ids":
        d.mkdir(parents=True, exist_ok=True)
        for k in range(nc):
            (d / f"chunk{k}.txt").write_text(",".join(map(str, tokens[k])))
        print(f"{nc} chunks of {n_ctx} tokens, vocab {n_vocab}, scored positions {first}..{n_ctx - 2}")
        return
    kls, same, nll_e, nll_b = [], 0, [], []
    for k in range(nc):
        for i in range(n_ctx - 1 - first):
            pos = first + i
            x = np.fromfile(d / f"chunk{k}_p{pos}.bin", "<f4").astype(np.float64)[:n_vocab]
            lse = x.max() + np.log(np.exp(x - x.max()).sum())
            lp = x - lse
            lb = base_logp(np.asarray(recs[k, i]), n_vocab)
            keep = lb > -16
            kls.append(float((np.exp(lb[keep]) * (lb[keep] - lp[keep])).sum()))
            same += int(lb.argmax() == x.argmax())
            tok = tokens[k, pos + 1]
            nll_e.append(-lp[tok])
            nll_b.append(-lb[tok])
    kl = np.array(kls)
    res = {"positions": len(kl), "kl_mean": kl.mean(), "kl_median": float(np.median(kl)),
           "kl_p95": float(np.percentile(kl, 95)), "kl_max": kl.max(), "same_top": same / len(kl),
           "ppl_engine": float(np.exp(np.mean(nll_e))), "ppl_base": float(np.exp(np.mean(nll_b)))}
    print(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
