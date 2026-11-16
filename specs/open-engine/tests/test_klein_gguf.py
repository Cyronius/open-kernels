"""IMAGE-FLUX2-GGUF: a klein DiT GGUF reads back as diffusers-named HF-layout weights."""
# Traces: IMAGE-FLUX2-GGUF
import struct

import numpy as np
import pytest
from ml_dtypes import bfloat16

import klein_gguf as kg

TYPE_ID = {"F32": 0, "F16": 1, "Q4_0": 2, "Q4_1": 3, "Q8_0": 8, "BF16": 30, "Q4_K": 12}


def encode(w: np.ndarray, typ: str) -> bytes:
    flat = w.astype(np.float32).ravel()
    if typ == "F32":
        return flat.tobytes()
    if typ == "F16":
        return flat.astype(np.float16).tobytes()
    if typ == "BF16":
        return flat.astype(bfloat16).view(np.uint16).tobytes()
    blocks = flat.reshape(-1, 32)
    out = bytearray()
    for b in blocks:
        if typ == "Q8_0":
            d = np.float16(np.abs(b).max() / 127 or 1)
            out += d.tobytes() + np.round(b / np.float32(d)).astype(np.int8).tobytes()
        elif typ == "Q4_0":
            d = np.float16(np.abs(b).max() / 7 or 1)
            q = (np.clip(np.round(b / np.float32(d)), -8, 7) + 8).astype(np.uint8)
            out += d.tobytes() + (q[:16] | (q[16:] << 4)).tobytes()
        elif typ in ("Q4_1", "Q4_K"):
            m = np.float16(b.min())
            d = np.float16((b.max() - np.float32(m)) / 15 or 1)
            q = np.clip(np.round((b - np.float32(m)) / np.float32(d)), 0, 15).astype(np.uint8)
            out += d.tobytes() + m.tobytes() + (q[:16] | (q[16:] << 4)).tobytes()
    return bytes(out)


def reference(w: np.ndarray, typ: str) -> np.ndarray:
    """What the file stores, decoded independently of klein_gguf."""
    flat = w.astype(np.float32).ravel()
    if typ == "F32":
        return flat.reshape(w.shape)
    if typ == "F16":
        return flat.astype(np.float16).astype(np.float32).reshape(w.shape)
    if typ == "BF16":
        return flat.astype(bfloat16).astype(np.float32).reshape(w.shape)
    out = []
    for b in flat.reshape(-1, 32):
        if typ == "Q8_0":
            d = np.float32(np.float16(np.abs(b).max() / 127 or 1))
            out.append(np.round(b / d) * d)
        elif typ == "Q4_0":
            d = np.float32(np.float16(np.abs(b).max() / 7 or 1))
            out.append(np.clip(np.round(b / d), -8, 7) * d)
        else:
            m = np.float32(np.float16(b.min()))
            d = np.float32(np.float16((b.max() - m) / 15 or 1))
            out.append(np.clip(np.round((b - m) / d), 0, 15) * d + m)
    return np.concatenate(out).reshape(w.shape)


def write_gguf(path, tensors: dict[str, tuple[np.ndarray, str]]):
    """GGUF v3: one metadata key, tensors stored ggml-style (dims fastest-first)."""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1)
    head += s("general.architecture") + struct.pack("<I", 8) + s("flux")
    blobs, table, off = [], b"", 0
    for name, (w, typ) in tensors.items():
        data = encode(w, typ)
        dims = list(reversed(w.shape))
        table += s(name) + struct.pack("<I", len(dims)) + struct.pack("<" + "Q" * len(dims), *dims)
        table += struct.pack("<IQ", TYPE_ID[typ], off)
        pad = (-len(data)) % 32
        blobs.append(data + b"\0" * pad)
        off += len(data) + pad
    hdr = head + table
    hdr += b"\0" * ((-len(hdr)) % 32)
    path.write_bytes(hdr + b"".join(blobs))


@pytest.mark.parametrize("typ", ["F32", "F16", "BF16", "Q8_0", "Q4_0", "Q4_1"])
def test_each_quant_decodes_to_the_stored_values_in_hf_layout(tmp_path, typ):
    w = np.random.default_rng(1).standard_normal((6, 64)).astype(np.float32)
    write_gguf(tmp_path / "m.gguf", {"img_in.weight": (w, typ)})
    got = kg.GgufWeights(tmp_path / "m.gguf", bfl_modulation=False).get("x_embedder.weight")
    assert got.shape == (6, 64)
    np.testing.assert_array_equal(got, reference(w, typ))


def test_fused_qkv_splits_q_k_v_in_order_per_stream(tmp_path):
    rng = np.random.default_rng(2)
    img, txt = rng.standard_normal((96, 32)), rng.standard_normal((96, 32))
    write_gguf(tmp_path / "m.gguf", {"double_blocks.3.img_attn.qkv.weight": (img, "F32"),
                                      "double_blocks.3.txt_attn.qkv.weight": (txt, "F32")})
    gw = kg.GgufWeights(tmp_path / "m.gguf", bfl_modulation=False)
    for i, c in enumerate("qkv"):
        np.testing.assert_array_equal(gw.get(f"transformer_blocks.3.attn.to_{c}.weight"),
                                      img[32 * i:32 * (i + 1)].astype(np.float32))
        np.testing.assert_array_equal(gw.get(f"transformer_blocks.3.attn.add_{c}_proj.weight"),
                                      txt[32 * i:32 * (i + 1)].astype(np.float32))


def test_bfl_modulation_swaps_the_final_adaln_halves_and_nothing_else(tmp_path):
    rng = np.random.default_rng(3)
    mod, other = rng.standard_normal((8, 32)), rng.standard_normal((8, 32))
    write_gguf(tmp_path / "m.gguf", {kg.MOD_TENSOR: (mod, "F32"),
                                      "single_stream_modulation.lin.weight": (other, "F32")})
    bfl = kg.GgufWeights(tmp_path / "m.gguf", bfl_modulation=True)
    np.testing.assert_array_equal(bfl.get("norm_out.linear.weight"),
                                  np.concatenate([mod[4:], mod[:4]]).astype(np.float32))
    np.testing.assert_array_equal(bfl.get("single_stream_modulation.linear.weight"),
                                  other.astype(np.float32))
    plain = kg.GgufWeights(tmp_path / "m.gguf", bfl_modulation=False)
    np.testing.assert_array_equal(plain.get("norm_out.linear.weight"), mod.astype(np.float32))


def test_modulation_order_is_detected_against_the_checkpoint_tensor():
    rng = np.random.default_rng(4)
    ref = rng.standard_normal((8, 32)).astype(np.float32)
    assert kg._mirror_decision(ref, ref) is False
    assert kg._mirror_decision(np.concatenate([ref[4:], ref[:4]]), ref) is True
    assert kg._mirror_decision(rng.standard_normal((8, 32)), ref) is None


def test_single_block_and_norm_names_map(tmp_path):
    rng = np.random.default_rng(5)
    lin1, qn = rng.standard_normal((4, 32)), rng.standard_normal(32)
    write_gguf(tmp_path / "m.gguf", {"single_blocks.19.linear1.weight": (lin1, "F32"),
                                      "single_blocks.19.norm.query_norm.scale": (qn, "F32")})
    gw = kg.GgufWeights(tmp_path / "m.gguf", bfl_modulation=False)
    np.testing.assert_array_equal(gw.get("single_transformer_blocks.19.attn.to_qkv_mlp_proj.weight"),
                                  lin1.astype(np.float32))
    np.testing.assert_array_equal(gw.get("single_transformer_blocks.19.attn.norm_q.weight"),
                                  qn.astype(np.float32))
    with pytest.raises(KeyError):
        gw.get("transformer_blocks.0.attn.to_q.bias")


def test_k_quants_are_refused_by_name(tmp_path):
    write_gguf(tmp_path / "m.gguf", {"img_in.weight": (np.zeros((2, 256)), "Q4_K")})
    with pytest.raises(ValueError, match="K-quant"):
        kg.Gguf(tmp_path / "m.gguf")


def test_geometry_comes_from_tensor_shapes(tmp_path):
    t = {"img_in.weight": (np.zeros((256, 32)), "F32"), "txt_in.weight": (np.zeros((256, 96)), "F32"),
         "double_blocks.0.img_attn.norm.query_norm.scale": (np.zeros(64), "F32"),
         "double_blocks.1.img_attn.proj.weight": (np.zeros((32, 32)), "F32"),
         "single_blocks.0.linear2.weight": (np.zeros((32, 32)), "F32"),
         "single_blocks.1.linear2.weight": (np.zeros((32, 32)), "F32"),
         "single_blocks.2.linear2.weight": (np.zeros((32, 32)), "F32")}
    write_gguf(tmp_path / "m.gguf", t)
    assert kg.geometry(kg.Gguf(tmp_path / "m.gguf")) == {
        "num_layers": 2, "num_single_layers": 3, "num_attention_heads": 4,
        "attention_head_dim": 64, "joint_attention_dim": 96, "in_channels": 32}


def test_readme_base_model_and_quant_choice():
    assert kg.base_model_from_readme("---\nbase_model: black-forest-labs/FLUX.2-klein-4B\n---\n") \
        == "black-forest-labs/FLUX.2-klein-4B"
    assert kg.base_model_from_readme("---\nbase_model:\n- a/b\n---\n") == "a/b"
    assert kg.base_model_from_readme("no frontmatter") == ""
    files = ["flux-2-klein-4b-F16.gguf", "flux-2-klein-4b-Q8_0.gguf", "flux-2-klein-4b-Q4_1.gguf"]
    assert kg.pick_quant(files) == "flux-2-klein-4b-Q8_0.gguf"
