"""klein_fetch: install FLUX.2 [klein] from a GGUF repo and write a `phlegm image` config, like `oflm add` (2026-11-14)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import klein_gguf as kg  # noqa: E402

PARTS = ["text_encoder/*", "vae/*", "tokenizer/*", "LICENSE.md", "model_index.json"]


def link_dir(src: Path, dst: Path) -> None:
    """A directory link without admin rights: a junction on Windows, a symlink elsewhere."""
    if dst.exists():
        return
    if os.name == "nt":
        import subprocess
        subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)], check=True,
                       capture_output=True)
    else:
        dst.symlink_to(src, target_is_directory=True)


def checkpoint_dir(base: str, out: Path, existing: Path | None) -> Path:
    """text_encoder/ vae/ tokenizer/ model_index.json, linked from `existing` or downloaded."""
    ckpt = out / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    if existing:
        for sub in ("text_encoder", "vae", "tokenizer"):
            link_dir(existing.resolve() / sub, ckpt / sub)
        for f in ("model_index.json", "LICENSE.md"):
            if (existing / f).is_file() and not (ckpt / f).exists():
                shutil.copy2(existing / f, ckpt / f)
    else:
        from huggingface_hub import snapshot_download
        snapshot_download(base, local_dir=str(ckpt), allow_patterns=PARTS)
    missing = [p for p in ("text_encoder/config.json", "vae/config.json",
                           "tokenizer/tokenizer.json", "model_index.json") if not (ckpt / p).is_file()]
    if missing:
        raise SystemExit(f"{ckpt}: missing {missing}")
    return ckpt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default="Atomic-Germ/FLUX.2-Klein-4B-OpenFLM-GGUF")
    ap.add_argument("--quant", help="GGUF file name in the repo (default: best of klein_gguf.QUANTS)")
    ap.add_argument("--gguf", type=Path, help="the DiT GGUF already on disk (skips its download)")
    ap.add_argument("--checkpoint", type=Path,
                    help="an existing diffusers checkpoint to link instead of downloading the base parts")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kernels", type=Path, required=True, help="export_dit_kernels.py output")
    ap.add_argument("--python", type=Path, default=Path(sys.executable))
    ap.add_argument("--env-script", type=Path)
    ap.add_argument("--dibs", type=Path, default=Path(shutil.which("dibs")) if shutil.which("dibs") else None)
    ap.add_argument("--output-root", type=Path)
    ap.add_argument("--config-out", type=Path, help="default <out>/phlegm-image.json")
    a = ap.parse_args()

    from huggingface_hub import hf_hub_download, list_repo_files
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    readme = Path(hf_hub_download(a.repo, "README.md")).read_text(encoding="utf-8", errors="ignore")
    base = kg.base_model_from_readme(readme)
    if not base:
        raise SystemExit(f"{a.repo}: its README names no base_model")
    if a.gguf:
        gguf = a.gguf.resolve()
    else:
        name = a.quant or kg.pick_quant(list_repo_files(a.repo))
        if any(q.lower() in name.lower() for q in kg.LOSSY):
            print(f"[WARN] {name}: OFLM measured 4-bit klein DiTs visibly grainy; prefer Q8_0", flush=True)
        print(f"DiT: {a.repo}/{name}", flush=True)
        gguf = Path(hf_hub_download(a.repo, name, local_dir=str(out)))
    print(f"checkpoint parts: {base}", flush=True)
    ckpt = checkpoint_dir(base, out, a.checkpoint)

    g = kg.Gguf(gguf)
    print(f"{gguf.name}: {g.types()} geometry {kg.geometry(g)}", flush=True)
    bfl = kg.detect_bfl_modulation(gguf, transformer_dir=(a.checkpoint / "transformer") if a.checkpoint else None,
                                   repo=base)
    modulation = {True: "bfl", False: "diffusers", None: "auto"}[bfl]
    print(f"modulation order: {modulation}", flush=True)

    model_id = base if base.startswith("black-forest-labs/") else "black-forest-labs/FLUX.2-klein-4B"
    cfg = {"backend": "flux2", "model_id": model_id, "python": str(a.python.resolve()),
           "repo_root": str(HERE.parent), "model": str(ckpt), "transformer_gguf": str(gguf),
           "gguf_modulation": modulation, "kernels": str(a.kernels.resolve()),
           "output_root": str((a.output_root or out / "outputs").resolve())}
    if a.env_script:
        cfg["env_script"] = str(a.env_script.resolve())
    if a.dibs:
        cfg["dibs"] = str(a.dibs.resolve())
    dest = a.config_out or out / "phlegm-image.json"
    dest.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {dest}\n  phlegm image --config {dest} --prompt \"a red fox in fresh snow\" --out <dir>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
