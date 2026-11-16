r"""flux2_klein_compare: a native FLUX.2 [klein] run (flux2_native_generate.py) against the
published reference (flux2_klein_reference.py) with the same prompt and noise.

    python open_kernels/flux2_klein_compare.py --native <native out> --reference <ref out>

Reports final packed latents (relative L2, cosine) and pixels (MAE in 8-bit levels,
PSNR, and with --lpips the AlexNet LPIPS, which needs torch and lpips), and writes
compare.json into the native directory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


def lpips_alex(x: np.ndarray, y: np.ndarray) -> float:
    import lpips
    import torch
    net = lpips.LPIPS(net="alex", verbose=False)
    t = [torch.from_numpy(v / 127.5 - 1).permute(2, 0, 1)[None].float() for v in (x, y)]
    with torch.no_grad():
        return float(net(t[0], t[1]).item())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--native", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--lpips", action="store_true", help="also LPIPS (AlexNet), needs torch + lpips")
    a = ap.parse_args()
    from PIL import Image

    ns = json.loads((a.native / "summary.json").read_text())
    rs = json.loads((a.reference / "summary.json").read_text())
    got = np.load(a.native / "latents.npy").view(bfloat16).astype(np.float64)
    want = np.load(a.reference / f"latents_step{rs['steps'] - 1}.npy").astype(np.float64)
    if got.shape != want.shape:
        raise SystemExit(f"latent shapes differ: {got.shape} vs {want.shape}")
    rel = float(np.linalg.norm(got - want) / np.linalg.norm(want))
    cos = float((got * want).sum() / (np.linalg.norm(got) * np.linalg.norm(want)))
    x = np.asarray(Image.open(ns["image"]).convert("RGB"), np.float64)
    y = np.asarray(Image.open(a.reference / "published.png").convert("RGB"), np.float64)
    mae = float(np.abs(x - y).mean())
    mse = float(((x - y) ** 2).mean())
    psnr = 10 * math.log10(255 ** 2 / mse) if mse else float("inf")
    out = {"native": str(a.native), "reference": str(a.reference),
           "prompt": ns["prompt"], "size": ns["size"], "steps": ns["steps"],
           "edit": ns["edit"], "latents_rel_l2": rel, "latents_cosine": cos,
           "latents_finite": bool(np.isfinite(got).all()),
           "pixel_mae_255": mae, "pixel_psnr_db": psnr}
    if a.lpips:
        out["lpips_alex"] = lpips_alex(x, y)
    (a.native / "compare.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
