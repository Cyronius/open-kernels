"""Convert the Alibaba GTE multilingual embedding checkpoint to Q8_0 GGUF."""

from argparse import ArgumentParser
from base64 import b64decode
import json
from math import isfinite
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from gguf import (
    GGMLQuantizationType as Quant,
    GGUFReader,
    GGUFWriter,
    Keys,
    LlamaFileType,
    PoolingType,
    RopeScalingType,
    TokenType,
    quantize,
)
import numpy as np
from safetensors import safe_open


MODEL_ID = "Alibaba-NLP/gte-multilingual-base"
WIDTH = 768
LAYERS = 12
HEADS = 12
CONTEXT = 8192
VOCAB_SIZE = 250048
TOKENIZER_VOCAB_SIZE = 250002
FEED_FORWARD = 3072


def rotary_metadata(config):
    expected = {
        "hidden_size": WIDTH,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "intermediate_size": FEED_FORWARD,
        "max_position_embeddings": CONTEXT,
        "vocab_size": VOCAB_SIZE,
        "type_vocab_size": 1,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-12,
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(f"{field} must be {value} for {MODEL_ID}")
    head_width = WIDTH // HEADS
    if config.get("model_type") == "gte":
        rope = config["rope_parameters"]
        if rope["rope_type"] != "linear":
            raise ValueError("GTE rope_parameters must use linear scaling")
        theta, factor = rope["rope_theta"], rope["factor"]
    elif config.get("model_type") == "new":
        rope = config["rope_scaling"]
        if rope["type"] != "ntk":
            raise ValueError("Legacy GTE rope_scaling must use NTK scaling")
        theta = config["rope_theta"] * rope["factor"]
        factor = rope["factor"] ** (2 / head_width)
    else:
        raise ValueError("Expected a GTE embedding checkpoint")
    if not all(isfinite(v) and v > 0 for v in (theta, factor)):
        raise ValueError("Rotary theta and scaling factor must be positive and finite")
    return head_width, theta, factor


def tensor_mapping():
    mapping = {
        "new.embeddings.word_embeddings.weight": ("token_embd.weight", (VOCAB_SIZE, WIDTH)),
        "new.embeddings.token_type_embeddings.weight": ("token_types.weight", (1, WIDTH)),
        "new.embeddings.LayerNorm.weight": ("token_embd_norm.weight", (WIDTH,)),
        "new.embeddings.LayerNorm.bias": ("token_embd_norm.bias", (WIDTH,)),
        "classifier.weight": ("cls.output.weight", (1, WIDTH)),
        "classifier.bias": ("cls.output.bias", (1,)),
    }
    layers = (
        ("attention.o_proj", "attn_output", (WIDTH, WIDTH), (WIDTH,)),
        ("attention.qkv_proj", "attn_qkv", (3 * WIDTH, WIDTH), (3 * WIDTH,)),
        ("attn_ln", "attn_output_norm", (WIDTH,), (WIDTH,)),
        ("mlp.down_proj", "ffn_down", (WIDTH, FEED_FORWARD), (WIDTH,)),
        ("mlp.up_gate_proj", "ffn_up", (2 * FEED_FORWARD, WIDTH), None),
        ("mlp_ln", "layer_output_norm", (WIDTH,), (WIDTH,)),
    )
    for layer in range(LAYERS):
        for native, gguf, weight_shape, bias_shape in layers:
            source = f"new.encoder.layer.{layer}.{native}"
            target = f"blk.{layer}.{gguf}"
            mapping[f"{source}.weight"] = (f"{target}.weight", weight_shape)
            if bias_shape is not None:
                mapping[f"{source}.bias"] = (f"{target}.bias", bias_shape)
    return mapping


def tokenizer_parts(text):
    tokenizer = json.loads(text)
    model = tokenizer["model"]
    normalizer = tokenizer["normalizer"]
    if model["type"] != "Unigram" or normalizer["type"] != "Precompiled":
        raise ValueError("Expected the publisher's Unigram tokenizer")
    vocab, scores = zip(*model["vocab"], strict=True)
    if len(vocab) != TOKENIZER_VOCAB_SIZE:
        raise ValueError(f"Expected {TOKENIZER_VOCAB_SIZE} tokenizer pieces, got {len(vocab)}")
    special = {item["content"]: item["id"] for item in tokenizer["added_tokens"] if item["special"]}
    if special != {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 250001}:
        raise ValueError("Tokenizer special IDs differ from the publisher checkpoint")
    charsmap = b64decode(normalizer["precompiled_charsmap"], validate=True)
    return vocab, scores, special, charsmap


def add_tokenizer(writer, source):
    text = source.read_bytes().decode("utf-8")
    vocab, scores, special, charsmap = tokenizer_parts(text)
    padding = range(len(vocab), VOCAB_SIZE)
    types = [TokenType.NORMAL] * len(vocab) + [TokenType.UNUSED] * len(padding)
    for piece, index in special.items():
        types[index] = TokenType.UNKNOWN if piece == "<unk>" else TokenType.CONTROL
    writer.add_tokenizer_model("t5")
    writer.add_tokenizer_pre("default")
    writer.add_string(Keys.Tokenizer.HF_JSON, text)
    writer.add_token_list(vocab + tuple(f"[PAD_VOCAB_{index}]" for index in padding))
    writer.add_token_scores(scores + (0.0,) * len(padding))
    writer.add_token_types(types)
    writer.add_token_type_count(1)
    writer.add_precompiled_charsmap(charsmap)
    writer.add_bos_token_id(special["<s>"])
    writer.add_eos_token_id(special["</s>"])
    writer.add_sep_token_id(special["</s>"])
    writer.add_pad_token_id(special["<pad>"])
    writer.add_unk_token_id(special["<unk>"])
    writer.add_mask_token_id(special["<mask>"])
    writer.add_add_bos_token(True)
    writer.add_add_eos_token(True)

def convert(checkpoint, path):
    config = json.loads((checkpoint / "config.json").read_bytes())
    head_width, theta, factor = rotary_metadata(config)
    mapping = tensor_mapping()
    writer = GGUFWriter(path, "new", use_temp_file=True)
    try:
        writer.add_name(MODEL_ID)
        writer.add_source_url(f"https://huggingface.co/{MODEL_ID}")
        writer.add_file_type(LlamaFileType.MOSTLY_Q8_0)
        writer.add_quantization_version(2)
        writer.add_causal_attention(False)
        writer.add_context_length(CONTEXT)
        writer.add_embedding_length(WIDTH)
        writer.add_block_count(LAYERS)
        writer.add_feed_forward_length(FEED_FORWARD)
        writer.add_head_count(HEADS)
        writer.add_layer_norm_eps(config["layer_norm_eps"])
        writer.add_rope_freq_base(theta)
        writer.add_rope_dimension_count(head_width)
        writer.add_rope_scaling_type(RopeScalingType.LINEAR)
        writer.add_rope_scaling_factor(factor)
        writer.add_pooling_type(PoolingType.CLS)
        add_tokenizer(writer, checkpoint / "tokenizer.json")

        expected = {}
        with safe_open(checkpoint / "model.safetensors", framework="numpy") as model:
            if set(model.keys()) != mapping.keys():
                raise ValueError("Tensor names differ from the GTE embedding checkpoint")
            for source, (target, shape) in mapping.items():
                tensor = model.get_tensor(source)
                if tensor.shape != shape:
                    raise ValueError(f"{source} has shape {tensor.shape}, expected {shape}")
                if not np.isfinite(tensor).all():
                    raise ValueError(f"{source} contains non-finite weights")
                # Keep the sparse classifier and all bias and normalization values lossless.
                if tensor.ndim == 2 and tensor.shape[0] > 1:
                    writer.add_tensor(target, quantize(tensor, Quant.Q8_0), raw_dtype=Quant.Q8_0)
                    expected[target] = (shape, Quant.Q8_0)
                else:
                    writer.add_tensor(target, tensor.astype(np.float32))
                    expected[target] = (shape, Quant.F32)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()

    reader = GGUFReader(path)
    actual = {tensor.name: (tuple(reversed(tensor.shape)), tensor.tensor_type) for tensor in reader.tensors}
    if actual != expected:
        raise ValueError("GGUF readback tensor inventory disagrees with the source checkpoint")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="publisher directory with config, tokenizer and weights")
    parser.add_argument("output", type=Path, help="new GGUF file")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    with TemporaryDirectory(dir=args.output.parent) as work:
        staged = Path(work) / "converted.gguf"
        convert(args.checkpoint, staged)
        os.link(staged, args.output)
    print(f"Wrote and verified {args.output}")


if __name__ == "__main__":
    main()
