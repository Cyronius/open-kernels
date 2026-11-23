"""Derive a sequence-length design from an existing GEMM design's rows.

A stream compiles the matrix row count M. Changing the interpretation of
M = batch * sequence does not change its instructions. Keep the maximum M
so the bootstrap instruction, device buffer sizes and xclbin can be reused.
"""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

from check_design_sets import STREAM_OPS


def derive(source: Path, output: Path, sequence: int, batches: list[int]) -> None:
    source = source / "gemm_rtp"
    design = json.loads((source / "design.json").read_text(encoding="utf-8"))
    if sequence < 8 or sequence % 8:
        raise ValueError("sequence length must be a positive multiple of eight")
    if not batches or min(batches) < 1 or sorted(set(batches)) != batches:
        raise ValueError("batch sizes must be positive, unique and increasing")
    if sequence * batches[-1] != design["M"]:
        raise ValueError("the largest matrix row count must match the source design")
    if output.exists():
        raise FileExistsError(output)

    source_streams = design["streams"]
    streams = {(stream["op"], stream["M"]): stream for stream in source_streams}
    if len(streams) != len(source_streams):
        raise ValueError("the source design contains duplicate row counts for an operation")
    copied = {"final.xclbin": "final.xclbin", "insts.bin": "insts.bin"}
    if (source / "toolchain.json").is_file():
        copied["toolchain.json"] = "toolchain.json"
    selected = []
    for batch in batches:
        for op in STREAM_OPS:
            rows = sequence * batch
            if (op, rows) not in streams:
                raise ValueError(f"{op} has no compiled stream for {rows} rows")
            stream = streams[op, rows]
            filename = f"insts_{op}_b{stream['batch']}.bin"
            if stream["file"] != filename:
                raise ValueError(f"invalid source instruction name for {op} at {rows} rows")
            target = f"insts_{op}_b{batch}.bin"
            copied[target] = filename
            selected.append(dict(stream, batch=batch, slot=len(selected) + 1, file=target))

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        staged = Path(temp) / "design"
        bundle = staged / "gemm_rtp"
        bundle.mkdir(parents=True)
        for target, filename in copied.items():
            shutil.copyfile(source / filename, bundle / target)
        design.update(seq=sequence, batch=batches[-1], tiers=batches, streams=selected)
        (bundle / "design.json").write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")
        staged.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seq", type=int, required=True)
    parser.add_argument("--batches", type=int, nargs="+", required=True)
    args = parser.parse_args()
    derive(args.source, args.output, args.seq, args.batches)


if __name__ == "__main__":
    main()
