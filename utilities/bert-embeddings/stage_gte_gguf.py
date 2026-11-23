"""Stage the supported GTE embedding GGUF for the existing NPUE packer."""

from argparse import ArgumentParser
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from gguf import GGMLQuantizationType as Quant, GGUFReader, Keys, LlamaFileType, PoolingType, dequantize
import numpy as np
from safetensors.numpy import save_file

from convert_gte_gguf import (
    CONTEXT, FEED_FORWARD, HEADS, LAYERS, MODEL_ID, VOCAB_SIZE, WIDTH, tensor_mapping, tokenizer_parts,
)


def verify_metadata(reader):
    # The existing GTE packer accepts the original NTK description of these
    # same frequencies. Refuse other variants before reconstructing it.
    expected = {
        "general.architecture": "new",
        "general.name": MODEL_ID,
        "general.file_type": int(LlamaFileType.MOSTLY_Q8_0),
        "new.attention.causal": False,
        "new.context_length": CONTEXT,
        "new.embedding_length": WIDTH,
        "new.block_count": LAYERS,
        "new.feed_forward_length": FEED_FORWARD,
        "new.attention.head_count": HEADS,
        "new.attention.layer_norm_epsilon": float(np.float32(1e-12)),
        "new.rope.dimension_count": WIDTH // HEADS,
        "new.rope.freq_base": 160000.0,
        "new.rope.scaling.type": "linear",
        "new.rope.scaling.factor": float(np.float32(8 ** (2 / (WIDTH // HEADS)))),
        "new.pooling_type": int(PoolingType.CLS),
    }
    for key, value in expected.items():
        field = reader.fields.get(key)
        if field is None or field.contents() != value:
            raise ValueError(f"unsupported GTE metadata {key}, expected {value!r}")


def stage(weights, output):
    if output.exists():
        raise FileExistsError(output)
    reader = GGUFReader(weights)
    verify_metadata(reader)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    mapping = tensor_mapping()
    if tensors.keys() != {name for name, _ in mapping.values()}:
        raise ValueError("GGUF tensor names disagree with the GTE embedding checkpoint")
    tokenizer = reader.fields[Keys.Tokenizer.HF_JSON].contents()
    tokenizer_parts(tokenizer)

    native = {}
    for key, (name, shape) in mapping.items():
        tensor = tensors[name]
        expected_type = Quant.Q8_0 if len(shape) == 2 and shape[0] > 1 else Quant.F32
        if tuple(reversed(tensor.shape)) != shape or tensor.tensor_type != expected_type:
            raise ValueError(f"unexpected shape or dtype for {name}")
        values = dequantize(tensor.data, tensor.tensor_type).reshape(shape).astype(np.float16)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite values in {name}")
        native[key] = values
    config = {
        "model_type": "new",
        "architectures": ["GteModel", "GteForTokenClassification"],
        "hidden_size": WIDTH,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "intermediate_size": FEED_FORWARD,
        "max_position_embeddings": CONTEXT,
        "vocab_size": VOCAB_SIZE,
        "type_vocab_size": 1,
        "pad_token_id": 1,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-12,
        "position_embedding_type": "rope",
        "pack_qkv": True,
        "rope_theta": 20000,
        "rope_scaling": {"type": "ntk", "factor": 8.0},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        staging = Path(temporary) / "checkpoint"
        staging.mkdir()
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        (staging / "tokenizer.json").write_bytes(tokenizer.encode("utf-8"))
        pooling = staging / "1_Pooling"
        pooling.mkdir()
        (pooling / "config.json").write_text(
            json.dumps({"pooling_mode_cls_token": True, "pooling_mode_mean_tokens": False}) + "\n",
            encoding="utf-8",
        )
        save_file(native, staging / "model.safetensors")
        staging.rename(output)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("output", type=Path, help="new directory for the staged checkpoint")
    args = parser.parse_args()
    stage(args.gguf, args.output)
    print(f"Staged a cold NPUE pack from {args.gguf} at {args.output}")


if __name__ == "__main__":
    main()
