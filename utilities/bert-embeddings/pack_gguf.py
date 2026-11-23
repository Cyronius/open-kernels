"""Pack a supported Q8_0 embedding GGUF using the pinned CPU NPUE packer."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import gguf
import numpy as np
from safetensors.numpy import save_file

from extend_npue import Reader, extend
from stage_gte_gguf import stage as stage_gte


MODELS = {
    "all-MiniLM-L6-v2": ("bert", "sentence-transformers/all-MiniLM-L6-v2", 256),
    "bge-small-en-v1.5": ("bert", "BAAI/bge-small-en-v1.5", 512),
    "bge-base-en-v1.5": ("bert", "BAAI/bge-base-en-v1.5", 512),
    "bge-large-en-v1.5": ("bert", "BAAI/bge-large-en-v1.5", 512),
    "nomic-embed-text-v1.5": ("nomic-bert", "nomic-ai/nomic-embed-text-v1.5", 8192),
    "Alibaba-NLP/gte-multilingual-base": ("new", "Alibaba-NLP/gte-multilingual-base", 8192),
}

BERT_TOP = {
    "token_embd.weight": "embeddings.word_embeddings.weight",
    "position_embd.weight": "embeddings.position_embeddings.weight",
    "token_types.weight": "embeddings.token_type_embeddings.weight",
    "token_embd_norm.weight": "embeddings.LayerNorm.weight",
    "token_embd_norm.bias": "embeddings.LayerNorm.bias",
}
NOMIC_TOP = {
    "token_embd.weight": "embeddings.word_embeddings.weight",
    "token_types.weight": "embeddings.token_type_embeddings.weight",
    "token_embd_norm.weight": "emb_ln.weight",
    "token_embd_norm.bias": "emb_ln.bias",
}
BERT_LAYER = {
    "attn_q": "attention.self.query",
    "attn_k": "attention.self.key",
    "attn_v": "attention.self.value",
    "attn_output": "attention.output.dense",
    "attn_output_norm": "attention.output.LayerNorm",
    "ffn_up": "intermediate.dense",
    "ffn_down": "output.dense",
    "layer_output_norm": "output.LayerNorm",
}
NOMIC_LAYER = {
    "attn_qkv.weight": "attn.Wqkv.weight",
    "attn_output.weight": "attn.out_proj.weight",
    "attn_output_norm.weight": "norm1.weight",
    "attn_output_norm.bias": "norm1.bias",
    "ffn_up.weight": "mlp.fc11.weight",
    "ffn_gate.weight": "mlp.fc12.weight",
    "ffn_down.weight": "mlp.fc2.weight",
    "layer_output_norm.weight": "norm2.weight",
    "layer_output_norm.bias": "norm2.bias",
}


def field(reader, key):
    entry = reader.fields.get(key)
    if entry is None:
        raise ValueError(f"GGUF is missing {key}")
    return entry.contents()


def wordpiece_token(token):
    special = token[1:-1] if token.startswith("[") and token.endswith("]") else ""
    if special in {"PAD", "UNK", "CLS", "SEP", "MASK"} or special.startswith("unused"):
        return token
    # GGUF's WordPiece conversion marks word starts using U+2581.
    return token[1:] if token.startswith("\u2581") else "##" + token


def stage_wordpiece(reader, directory, arch, length):
    nomic = arch == "nomic-bert"
    width, layers, heads, intermediate = (
        field(reader, f"{arch}.{key}")
        for key in ("embedding_length", "block_count", "attention.head_count", "feed_forward_length")
    )
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    vocab = [wordpiece_token(token) for token in field(reader, "tokenizer.ggml.tokens")]
    if field(reader, "tokenizer.ggml.model") != "bert" or len(vocab) != tensors["token_embd.weight"].shape[1]:
        raise ValueError("GGUF vocabulary does not match the WordPiece embedding tensor")
    if any(not token or "\n" in token or "\r" in token for token in vocab):
        raise ValueError("invalid WordPiece vocabulary entry")

    mapping = dict(NOMIC_TOP if nomic else BERT_TOP)
    layer_mapping = NOMIC_LAYER if nomic else {
        f"{src}.{part}": f"{dst}.{part}"
        for src, dst in BERT_LAYER.items() for part in ("weight", "bias")
    }
    for i in range(layers):
        prefix = f"encoder.layers.{i}" if nomic else f"encoder.layer.{i}"
        mapping.update((f"blk.{i}.{src}", f"{prefix}.{dst}") for src, dst in layer_mapping.items())
    native = {}
    for key, target in mapping.items():
        tensor = tensors.get(key)
        if tensor is None:
            raise ValueError(f"GGUF is missing {key}")
        if tensor.tensor_type.name not in {"Q8_0", "F16", "F32"}:
            raise ValueError(f"unsupported tensor type for {key}")
        values = gguf.dequantize(tensor.data, tensor.tensor_type).reshape(tuple(reversed(tensor.shape)))
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite values in {key}")
        native[target] = values.astype(np.float32, copy=False)

    eps = field(reader, f"{arch}.attention.layer_norm_epsilon")
    config = {
        "model_type": "nomic_bert" if nomic else "bert", "hidden_size": width,
        "num_hidden_layers": layers, "num_attention_heads": heads, "intermediate_size": intermediate,
        "vocab_size": len(vocab), "type_vocab_size": int(tensors["token_types.weight"].shape[1]),
        "max_position_embeddings": length, "layer_norm_eps": eps,
    }
    if nomic:
        if field(reader, f"{arch}.rope.freq_base") != 1000.0 or width // heads != 64:
            raise ValueError("unexpected Nomic v1.5 rotary geometry")
        config.update(
            head_dim=64, layer_norm_epsilon=eps, rotary_emb_base=1000.0,
            rotary_emb_fraction=1.0, rotary_emb_interleaved=False, qkv_proj_bias=False,
            mlp_fc1_bias=False, mlp_fc2_bias=False, prenorm=False,
            activation_function="swiglu", hidden_act="silu",
        )
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8", newline="\n")
    save_file(native, directory / "model.safetensors")


def verify_projection(config, reader, arch):
    width = field(reader, f"{arch}.embedding_length")
    intermediate = field(reader, f"{arch}.feed_forward_length")
    gated = arch in ("nomic-bert", "new")
    operations = {
        "qkv": (width, 3 * width),
        "attn_out": (width, width),
        "ffn_up": (width, (2 if gated else 1) * intermediate),
        "ffn_down": (intermediate, width),
    }
    seen = set()
    for stream in config["streams"]:
        op = stream["op"]
        if op in operations:
            if (stream["K"], stream["N"]) != operations[op]:
                raise ValueError(f"{op} projection geometry disagrees with the GGUF")
            seen.add(op)
    if seen != operations.keys():
        raise ValueError("projection design is missing encoder streams")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_source(gguf_path, expected_hash):
    if sha256(gguf_path) != expected_hash:
        raise ValueError("GGUF SHA-256 mismatch")


def pack(gguf_path, design, packer, output, expected_hash, *, log=None):
    if output.exists():
        raise FileExistsError(output)
    require_source(gguf_path, expected_hash)
    data = gguf.GGUFReader(gguf_path)
    name = field(data, "general.name")
    if name not in MODELS:
        raise ValueError(f"unsupported embedding GGUF {name!r}")
    arch, repo, max_context = MODELS[name]
    if (field(data, "general.architecture") != arch or
            field(data, "general.file_type") != int(gguf.LlamaFileType.MOSTLY_Q8_0) or
            field(data, f"{arch}.attention.causal")):
        raise ValueError("expected the supported noncausal Q8_0 embedding GGUF")
    config = json.loads((design / "gemm_rtp" / "design.json").read_text(encoding="utf-8"))
    layout = config["b_layout"]
    length = config["seq"]
    stored_context = field(data, f"{arch}.context_length")
    if arch != "nomic-bert":
        max_context = min(max_context, stored_context)
    elif stored_context != 2048:
        raise ValueError("expected Nomic v1.5's original 2048-token GGUF context")
    if not isinstance(length, int) or not 0 < length <= max_context:
        raise ValueError(f"projection context must be at most {max_context} tokens")
    if layout["tile_k"] != 64 or layout["tile_n"] not in (32, 48):
        raise ValueError("unsupported embedding projection tile")
    verify_projection(config, data, arch)
    pool = {1: "mean", 2: "cls"}.get(field(data, f"{arch}.pooling_type"))
    if pool is None:
        raise ValueError("unsupported embedding pooling")

    with TemporaryDirectory(dir=output.parent) as directory:
        temporary = Path(directory)
        checkpoint = temporary / "checkpoint"
        if arch == "new":
            stage_gte(gguf_path, checkpoint)
        else:
            stage_wordpiece(data, checkpoint, arch, length)
        unaugmented = temporary / "packed.npue"
        packer_arch = {"bert": "bert", "nomic-bert": "nomic", "new": "gte"}[arch]
        subprocess.run([str(packer.resolve(strict=True)), packer_arch, str(checkpoint),
                        repo, pool, str(length), str(layout["tile_n"]), str(unaugmented)],
                       check=True, stdout=log)
        # Verify the pack against its selected design before making it visible.
        with Reader(unaugmented) as packed:
            if packed.config["max_seq_len"] != length or packed.config["hidden"] != field(
                data, f"{arch}.embedding_length"
            ):
                raise ValueError("packed encoder geometry disagrees with the requested design")
            if packed.entries["layer.0.attn_out"]["layout_hash"] != config["b_layout_hash"]:
                raise ValueError("packed encoder layout disagrees with the projection design")
        result = unaugmented
        if arch == "new":
            result = temporary / "with-sparse.npue"
            extend(unaugmented, gguf_path, result, length, gte_sparse_head=True)
        result.rename(output)


def cached_pack(gguf_path, design, packer, cache, expected_hash):
    require_source(gguf_path, expected_hash)
    here = Path(__file__).resolve().parent
    identity = {
        "gguf_sha256": expected_hash,
        "design_sha256": sha256(design / "gemm_rtp" / "design.json"),
        "packer_sha256": sha256(packer),
        "scripts_sha256": {
            rel: sha256(here / rel)
            for rel in ("pack_gguf.py", "stage_gte_gguf.py", "convert_gte_gguf.py",
                        "extend_npue.py", "../../npu_offload/gemm_rtp/npue.py")
        },
        "python_packages": {name: version(name) for name in ("gguf", "numpy", "safetensors")},
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    destination = cache / key
    output = destination / "model.npue"
    if not destination.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="packing-", dir=cache) as temporary:
            entry = Path(temporary) / "entry"
            entry.mkdir()
            packed = entry / "model.npue"
            pack(gguf_path, design, packer, packed, expected_hash, log=sys.stderr)
            (entry / "receipt.json").write_text(
                json.dumps({**identity, "npue_sha256": sha256(packed)}, sort_keys=True),
                encoding="utf-8",
            )
            try:
                entry.rename(destination)
            except OSError:
                # A concurrent cold run may already have published this key.
                if not destination.is_dir():
                    raise
    receipt = json.loads((destination / "receipt.json").read_text(encoding="utf-8"))
    if receipt != {**identity, "npue_sha256": sha256(output)}:
        raise ValueError(f"cached NPUE integrity check failed: {destination}")
    return output.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("design", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--cache", type=Path, help="publish and reuse a checksummed NPUE cache entry")
    parser.add_argument("--packer", type=Path, required=True)
    parser.add_argument("--gguf-sha256", required=True)
    args = parser.parse_args()
    if (args.cache is None) == (args.output is None):
        parser.error("provide either an output pathname or --cache")
    if args.cache is not None:
        output = cached_pack(args.gguf, args.design, args.packer, args.cache, args.gguf_sha256)
        print(json.dumps({"pack_path": str(output), "gguf_sha256": args.gguf_sha256}))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pack(args.gguf, args.design, args.packer, args.output, args.gguf_sha256)


if __name__ == "__main__":
    main()
