r"""Check an attn_fa256 run (make_test.py's o.bin) against its fp64 reference: rel_fro and worst row cosine per head.

    python compare.py <dir> [--gate 3e-2]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

T, HD = 256, 256


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--gate", type=float, default=3e-2)
    a = ap.parse_args()
    ref = np.fromfile(a.dir / "ref.bin", np.float32).astype(np.float64).reshape(T, a.heads, HD)
    got = np.fromfile(a.dir / "o.bin", np.float32).astype(np.float64).reshape(T, a.heads, HD)
    ok = bool(np.isfinite(got).all())
    worst_rf, worst_cos = 0.0, 1.0
    for h in range(a.heads):
        r, g = ref[:, h], got[:, h]
        rf = np.linalg.norm(g - r) / np.linalg.norm(r)
        cos = (g * r).sum(1) / (np.linalg.norm(g, axis=1) * np.linalg.norm(r, axis=1) + 1e-30)
        worst_rf, worst_cos = max(worst_rf, rf), min(worst_cos, cos.min())
        print(f"  head {h:2d}: rel_fro {rf:.3e}  min row cos {cos.min():.5f}  worst row {int(cos.argmin())}")
    verdict = ok and worst_rf <= a.gate
    print(f"{'PASS' if verdict else 'FAIL'}: worst rel_fro {worst_rf:.3e} (gate {a.gate:.0e}), worst row cos "
          f"{worst_cos:.5f}, finite {ok}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
