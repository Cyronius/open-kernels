"""Build and package the Inkling 2 Mini v11 text and vision kernels for NPU2."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from recipes.inkling2_routed import (FORMATS, NEAR_ATTENTION_CHUNKS, VISION_ATTENTION_ROWS, VISION_PROJECTIONS,
                                     WIDE_OLD_ATTENTION_KEYS)


HERE = Path(__file__).resolve().parent
DESIGNS = HERE / "designs"
# 36 covers a decode or verify window (1,024 keys plus up to 8 rows, chunk-aligned) without padding to 64.
ATTENTION_CHUNKS = (1, 4, 16, 32, 36, 64, 128)
SPECIALIZATION_PREFIXES = ("INKLING2_", "LMHEAD_", "AF_")


def kernel_builds(wide_old_attention=False, head_pair=False, singleton_decode=False, near_attention=False):
    for quant in FORMATS:
        for leg, width in (("up", 2304), ("down", 1152)):
            yield f"shared-{quant}-{leg}", "inkling2/shared_decode.py", dict(
                INKLING2_SHARED_FORMAT=quant, INKLING2_SHARED_K=width, INKLING2_SHARED_CHUNKS=0)
            # With singleton decode, grouped Q8 prefill reads the same packed experts.
            yield f"prefill-{quant}-{leg}", "inkling2/compact_decode.py", dict(
                INKLING2_COMPACT_FORMAT=quant, INKLING2_COMPACT_K=width,
                INKLING2_COMPACT_TOKENS=16, INKLING2_COMPACT_ROWS=4,
                INKLING2_COMPACT_SPLIT_SCALE=int(singleton_decode and quant == "q8"))
        for chunks in ATTENTION_CHUNKS:
            yield f"shared-{quant}-attention-{chunks}", "inkling2/shared_decode.py", dict(
                INKLING2_SHARED_FORMAT=quant, INKLING2_SHARED_K=0, INKLING2_SHARED_CHUNKS=chunks)

    if singleton_decode:
        for quant in FORMATS:
            for leg, width in (("up", 2304), ("down", 1152)):
                yield f"singleton-{quant}-{leg}", "inkling2/shared_decode.py", dict(
                    INKLING2_SHARED_SINGLETON=1, INKLING2_SHARED_FORMAT=quant,
                    INKLING2_SHARED_K=width, INKLING2_SHARED_CHUNKS=0)
        for chunks in ATTENTION_CHUNKS:
            yield f"singleton-attention-{chunks}", "inkling2/shared_decode.py", dict(
                INKLING2_SHARED_SINGLETON=1, INKLING2_SHARED_FORMAT="q8", INKLING2_SHARED_K=0,
                INKLING2_SHARED_CHUNKS=chunks)
        for name, width, bands in (("fused", 2304, 3), ("output", 3072, 1)):
            yield f"singleton-{name}", "inkling2/shared_decode.py", dict(
                INKLING2_SHARED_SINGLETON=1, INKLING2_SHARED_FORMAT="q8", INKLING2_SHARED_K=width,
                INKLING2_SHARED_CHUNKS=0, INKLING2_SHARED_BANDS=bands)

    for name, width, outputs in (("fused", 2304, 9216), ("query", 2304, 6912),
                                 ("kv", 2304, 1536), ("output", 3072, 2304)):
        yield f"dense-{name}", "inkling2/block_expert.py", dict(
            INKLING2_BLOCK_FORMAT="q8", INKLING2_BLOCK_K=width, INKLING2_BLOCK_N=outputs,
            INKLING2_BLOCK_ARRAY_ROWS=4, INKLING2_BLOCK_SPLIT_SCALE=1)
    yield "head", "lm_head_q8/lm_head_q8.py", dict(
        LMHEAD_K=2304, LMHEAD_N=200064, LMHEAD_CORES=8, LMHEAD_ROWS=1, LMHEAD_SCALES_F32=1)
    if head_pair:
        yield "head-pair", "lm_head_q8/lm_head_q8.py", dict(
            LMHEAD_K=2304, LMHEAD_N=200064, LMHEAD_CORES=8, LMHEAD_ROWS=2, LMHEAD_SCALES_F32=1,
            LMHEAD_OUTPUT_BANDS=1)

    for width, outputs in VISION_PROJECTIONS:
        yield f"k{width}-n{outputs}", "gemma4_mm/gemma4_mm.py", dict(
            PHLEGM_M=256, PHLEGM_K=width, PHLEGM_N=outputs)
    for rows in VISION_ATTENTION_ROWS:
        yield f"rows-{rows}", "attn_fa/attn_fa.py", dict(
            AF_MODE="attn", AF_L=rows, AF_HEADS=16, AF_KV_HEADS=16, AF_MAX_ROWS=42496,
            AF_KV_LD=4096, AF_BIDIR=1, AF_QSPLIT=1, AF_KSPLIT=1, AF_VSPLIT=1, AF_PSPLIT=1)
    if near_attention:
        yield "near-attention", "inkling2/near_attention.py", dict(INKLING2_NEAR_CHUNKS=NEAR_ATTENTION_CHUNKS)
    if wide_old_attention:
        for keys in WIDE_OLD_ATTENTION_KEYS:
            yield f"wide-old-attention-{keys}", "inkling2/wide_old_attention.py", dict(INKLING2_WIDE_KEYS=keys)


def build_kernels(build_dir: Path, wide_old_attention=False, head_pair=False, singleton_decode=False, near_attention=False):
    env = {name: value for name, value in os.environ.items() if not name.startswith(SPECIALIZATION_PREFIXES)}
    for name, design, settings in kernel_builds(wide_old_attention, head_pair, singleton_decode, near_attention):
        print(f"Building {name}", flush=True)
        stream_env = dict(env, **{key: str(value) for key, value in settings.items()})
        subprocess.run([sys.executable, HERE / "build_design.py", DESIGNS / design, build_dir / name],
                       env=stream_env, check=True)


def package(build_dir: Path, out: Path, wide_old_attention=False, head_pair=False, singleton_decode=False,
            reclaim_packed_keys=False, near_attention=False):
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}-", dir=out.parent) as temp:
        staged = Path(temp) / "bundle"
        command = [sys.executable, HERE / "recipes/inkling2_routed.py", build_dir, staged,
                   "--compact", "--shared-attention", "--grouped-blocks", build_dir,
                   "--dense", build_dir, "--head", build_dir / "head", "--vision-gemm", build_dir,
                   "--vision-attention", build_dir, "--vision-split-keys", "--vision-split-values"]
        if wide_old_attention:
            command.extend(["--wide-old-attention", build_dir])
        if reclaim_packed_keys:
            command.append("--reclaim-packed-keys")
        if near_attention:
            command.extend(["--near-attention", build_dir / "near-attention"])
        if head_pair:
            command.extend(["--head-pair", build_dir / "head-pair"])
        if singleton_decode:
            command.extend(["--singleton-decode", build_dir, "--split-prefill-q8"])
        subprocess.run(command, check=True)
        staged.rename(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new directory for the Phlegm kernel bundle")
    parser.add_argument("--build-dir", type=Path, help="compiler outputs, defaults to <out>-builds")
    parser.add_argument("--package-only", action="store_true", help="package existing v11 compiler outputs")
    parser.add_argument("--wide-old-attention", action="store_true", help="include the experimental 4K/16K older-key streams")
    parser.add_argument("--reclaim-packed-keys", action="store_true",
                        help="release duplicate raw key pages on Windows with wide older-key attention")
    parser.add_argument("--head-pair", action="store_true", help="include the two-row Q8 verification head")
    parser.add_argument("--near-attention", action="store_true",
                        help="include the 256-row near-range block attention stream")
    parser.add_argument("--singleton-decode", action=argparse.BooleanOptionalAction, default=True,
                        help="dense decode projections on the NPU with a shared Q8 prefill pool (about 1 GiB more memory)")
    args = parser.parse_args()
    if args.reclaim_packed_keys and not args.wide_old_attention:
        parser.error("--reclaim-packed-keys requires --wide-old-attention")
    out = args.out.resolve()
    build_dir = (args.build_dir or out.with_name(out.name + "-builds")).resolve()
    if out.exists():
        parser.error(f"output already exists: {out}")
    if out == build_dir or out in build_dir.parents:
        parser.error("build directory must be outside the bundle output")
    if args.package_only:
        if not build_dir.is_dir():
            parser.error(f"missing build directory: {build_dir}")
    else:
        if build_dir.exists() and (not build_dir.is_dir() or any(build_dir.iterdir())):
            parser.error(f"choose an empty build directory: {build_dir}")
        build_kernels(build_dir, args.wide_old_attention, args.head_pair, args.singleton_decode, args.near_attention)
    package(build_dir, out, args.wide_old_attention, args.head_pair, args.singleton_decode, args.reclaim_packed_keys,
            args.near_attention)
    print(f"Wrote Inkling 2 Mini v11 kernels to {out}")


if __name__ == "__main__":
    main()
