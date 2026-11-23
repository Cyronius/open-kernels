"""Extend a packed GGUF encoder's positions for local context experiments.

The stock packer truncates its position table. Restore the missing position
rows from the same GGUF while preserving all other packed tensor bytes.
"""

import argparse
import hashlib
from pathlib import Path
import sys

import gguf
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "npu_offload" / "gemm_rtp"))
from npue import Reader, Writer


POSITION = "embeddings.position"


def extend(packed, weights, output, length):
    if output.exists():
        raise FileExistsError(output)
    source = gguf.GGUFReader(weights)
    arch = source.fields["general.architecture"].contents()
    capacity = source.fields[f"{arch}.context_length"].contents()
    with Reader(packed) as container:
        entry = container.entries[POSITION]
        old_length, width = entry["logical_shape"]
        if old_length != container.config["max_seq_len"] or entry["dtype"] != "F32":
            raise ValueError("packed position tensor disagrees with its declared capacity")
        if not old_length < length <= capacity:
            raise ValueError(f"requested length must be above {old_length} and at most {capacity}")
        old = container.raw(POSITION).reshape(old_length, width)
        if arch == "bert" and container.config.get("position_embedding_type") is None:
            table = next(t for t in source.tensors if t.name == "position_embd.weight")
            positions = gguf.dequantize(table.data, table.tensor_type)[:length]
        elif arch == "nomic-bert" and container.config.get("position_embedding_type") == "rope":
            positions = np.zeros((length, width), dtype=np.float32)
        else:
            raise ValueError("GGUF architecture and packed position encoding are unsupported")
        if positions.shape != (length, width) or not np.array_equal(positions[:old_length], old):
            raise ValueError("GGUF position tensor disagrees with the original packed rows")
        del old

        writer = Writer({**container.config, "max_seq_len": length}, container.arch, container.flags)
        for name, item in container.entries.items():
            if name == POSITION:
                writer.add(name, positions, "F32", item["role"], positions.shape)
            else:
                writer.add(name, container.raw(name), item["dtype"], item["role"],
                           item["logical_shape"], item.get("layout"), item["padded_shape"])
        temporary = output.with_suffix(".npue.part")
        writer.write(temporary)
        temporary.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packed", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--gguf-sha256", required=True)
    args = parser.parse_args()
    with args.weights.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != args.gguf_sha256:
            raise ValueError("GGUF SHA-256 mismatch")
    extend(args.packed, args.weights, args.output, args.length)


if __name__ == "__main__":
    main()
