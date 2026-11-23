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
from npue import ARCH_GTE_NEW_ROPE_GEGLU, Reader, Writer


POSITION = "embeddings.position"
GTE_SPARSE = {
    "cls.output.weight": ("sparse.weight", (1, 768)),
    "cls.output.bias": ("sparse.bias", (1,)),
}


def check_gte_rope(source, container):
    config = container.config
    if container.arch != ARCH_GTE_NEW_ROPE_GEGLU or config["head_dim"] != 64:
        raise ValueError("expected a packed 64-wide GTE checkpoint")
    if source.fields["general.name"].contents() != "Alibaba-NLP/gte-multilingual-base":
        raise ValueError("expected the GTE embedding checkpoint")
    if source.fields["new.rope.scaling.type"].contents() != "linear":
        raise ValueError("expected linearly represented GTE rotary frequencies")
    base = source.fields["new.rope.freq_base"].contents()
    factor = source.fields["new.rope.scaling.factor"].contents()
    expected = base ** (-np.arange(32, dtype=np.float64) / 32) / factor
    stored = np.asarray(config["rope_inv_freq"])
    if stored.shape != expected.shape or not np.allclose(stored, expected, rtol=2e-6, atol=0):
        raise ValueError("GGUF rotary frequencies disagree with the original GTE pack")


def add_gte_sparse_head(writer, source, container):
    tensors = {tensor.name: tensor for tensor in source.tensors}
    for name, (key, shape) in GTE_SPARSE.items():
        tensor = tensors.get(name)
        if (tensor is None or tensor.tensor_type != gguf.GGMLQuantizationType.F32
                or tuple(reversed(tensor.shape)) != shape
                or not np.isfinite(tensor.data).all()):
            raise ValueError(f"invalid GTE sparse head tensor {name}")
        if key in container.entries:
            raise ValueError(f"{key} is already present in the NPUE pack")
        writer.add(key, tensor.data.reshape(shape), "F32", "classifier", shape)


def extend(packed, weights, output, length, *, nomic_dynamic_rope=False,
           gte_sparse_head=False):
    if output.exists():
        raise FileExistsError(output)
    source = gguf.GGUFReader(weights)
    arch = source.fields["general.architecture"].contents()
    if gte_sparse_head and arch != "new":
        raise ValueError("a GTE sparse head requires the GTE embedding GGUF")
    capacity = source.fields[f"{arch}.context_length"].contents()
    with Reader(packed) as container:
        if nomic_dynamic_rope:
            config = container.config
            if (arch != "nomic-bert" or capacity != 2048
                    or source.fields["general.name"].contents() != "nomic-embed-text-v1.5"
                    or source.fields["nomic-bert.rope.freq_base"].contents() != 1000.0
                    or config.get("source_repo") != "nomic-ai/nomic-embed-text-v1.5"
                    or config.get("arch") != "nomic_bert_rope_swiglu"
                    or config.get("head_dim") != 64 or config.get("rope_theta") != 1000.0):
                raise ValueError("dynamic RoPE extension requires the Nomic v1.5 GGUF and NPUE pack")
            capacity = 8192
        entry = container.entries[POSITION]
        old_length, width = entry["logical_shape"]
        if old_length != container.config["max_seq_len"] or entry["dtype"] != "F32":
            raise ValueError("packed position tensor disagrees with its declared capacity")
        same_length_sparse = gte_sparse_head and old_length == length
        if length > capacity or (not same_length_sparse and length <= old_length):
            raise ValueError(f"requested length must be above {old_length} and at most {capacity}")
        old = container.raw(POSITION).reshape(old_length, width)
        if arch == "bert" and container.config.get("position_embedding_type") is None:
            table = next(t for t in source.tensors if t.name == "position_embd.weight")
            positions = gguf.dequantize(table.data, table.tensor_type)[:length]
        elif arch == "nomic-bert" and container.config.get("position_embedding_type") == "rope":
            positions = np.zeros((length, width), dtype=np.float32)
        elif arch == "new" and container.config.get("position_embedding_type") == "rope":
            check_gte_rope(source, container)
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
        if gte_sparse_head:
            add_gte_sparse_head(writer, source, container)
        temporary = output.with_suffix(".npue.part")
        writer.write(temporary)
        temporary.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packed", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--nomic-dynamic-rope", action="store_true",
                        help="allow Nomic v1.5's documented 8192-token dynamic RoPE context")
    parser.add_argument("--gte-sparse-head", action="store_true",
                        help="preserve the GTE classifier alongside its dense encoder")
    parser.add_argument("--gguf-sha256", required=True)
    args = parser.parse_args()
    with args.weights.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != args.gguf_sha256:
            raise ValueError("GGUF SHA-256 mismatch")
    extend(args.packed, args.weights, args.output, args.length,
           nomic_dynamic_rope=args.nomic_dynamic_rope, gte_sparse_head=args.gte_sparse_head)


if __name__ == "__main__":
    main()
