"""Build or check a runtime-length bidirectional attention image for BERT."""

from argparse import ArgumentParser
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
DESIGN = HERE / "designs" / "whisper_fa" / "attn_fa.py"
FORMAT = "bert-attention-v1"
HEADS = 12
HEAD_WIDTH = 64
GEOMETRIES = {(12, 32): (256, 512), (12, 64): (512, 2048, 8192), (16, 64): (512,)}
SEQUENCES = tuple(sorted({seq for lengths in GEOMETRIES.values() for seq in lengths}))
LENGTH_MARKER = 381
PAYLOAD = ("final.xclbin", "insts.bin")


def validate_geometry(sequence, heads, head_width):
    if sequence not in GEOMETRIES.get((heads, head_width), ()):
        raise ValueError(f"unsupported BERT attention geometry {heads} x {head_width} at {sequence}")


def artifact_manifest(path, sequence, heads=HEADS, head_width=HEAD_WIDTH):
    validate_geometry(sequence, heads, head_width)
    files = {}
    for name in PAYLOAD:
        with (path / name).open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            if not size or (name == "insts.bin" and size % 4):
                raise ValueError(f"invalid size for {path / name}")
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        files[name] = {"bytes": size, "sha256": digest}
    return {
        "format": FORMAT,
        "sequence": sequence,
        "heads": heads,
        "head_width": head_width,
        "length_marker": LENGTH_MARKER,
        "dtype": "bf16",
        "layout": "head_sequence_width",
        "attention_scale": "inverse_sqrt_head_width",
        "files": files,
    }


def check_artifact(path, sequence, heads=HEADS, head_width=HEAD_WIDTH):
    recorded = json.loads((path / "attention.json").read_text(encoding="utf-8"))
    if recorded != artifact_manifest(path, sequence, heads, head_width):
        raise ValueError(f"{path} does not match the requested BERT attention artifact")


def build(path, sequence, heads=HEADS, head_width=HEAD_WIDTH):
    validate_geometry(sequence, heads, head_width)
    if path.exists():
        raise FileExistsError(path)
    env = {
        **os.environ,
        "FA_LQ": str(sequence),
        "FA_LK": str(sequence),
        "FA_LQP": "256",
        "FA_LKP": str(head_width),
        "FA_DK": str(head_width),
        "FA_DV": str(head_width),
        "FA_NUM_HEADS": str(heads),
        "FA_HEADS_PER_UNROLL": "2",
        "FA_CASCADE_STAGES": "4",
        "FA_VALID_LEN": str(LENGTH_MARKER),
        "FA_RUNTIME_VALID_LEN": "1",
        "FA_SCALE_ONE": "0",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as work:
        compiled = Path(work) / "build"
        staged = Path(work) / "staged"
        subprocess.run([sys.executable, str(HERE / "build_design.py"), str(DESIGN), str(compiled)],
                       env=env, check=True)
        staged.mkdir()
        for name in PAYLOAD:
            shutil.copyfile(compiled / name, staged / name)
        manifest = artifact_manifest(staged, sequence, heads, head_width)
        (staged / "attention.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staged.rename(path)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sequence", required=True, type=int, choices=SEQUENCES)
    parser.add_argument("--heads", type=int, default=HEADS)
    parser.add_argument("--head-width", type=int, default=HEAD_WIDTH)
    parser.add_argument("--check", action="store_true", help="verify an existing artifact")
    args = parser.parse_args()
    if not args.check:
        build(args.out, args.sequence, args.heads, args.head_width)
    check_artifact(args.out, args.sequence, args.heads, args.head_width)
    print(f"Verified {args.out} at {args.sequence} tokens ({args.heads} x {args.head_width})")


if __name__ == "__main__":
    main()
