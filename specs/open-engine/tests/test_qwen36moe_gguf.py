# Traces: OPEN-FAMILY-QWEN36MOE-GGUF (canonical spec: specs/open-engine/spec.md)
"""The Qwen3.5 / 3.6 MoE GGUF-direct view: the q4_1 manifest, op for op, read from a llama.cpp
qwen35moe GGUF by the runtime's q35_* ops (upstream-patches/qwen35moe-gguf-runtime.patch)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from recipes import qwen36moe as Q
from recipes.load import default_spec
from recipes.manifest import manifest


def _view():
    m = manifest(default_spec())
    return m, m["gguf"]


def test_view_is_the_q4_1_manifest_with_gguf_pack_ops():
    m, g = _view()
    # OPEN-LMHEAD-Q6: the view adds the Q6_K head's kernels and contexts, nothing else
    q6k, q6c = {"lm_q6", "lmv_q6"}, {"lm6", "lmv6"}
    assert g["layout"] == m["layout"]
    assert {k: v for k, v in g["kernels"].items() if k not in q6k} == m["kernels"]
    assert {k: v for k, v in g["contexts"].items() if k not in q6c} == m["contexts"]
    assert g["spec_hash"] == m["spec_hash"] and g["pack"]["chunk_bytes"] == 5120 == m["pack"]["chunk_bytes"]
    for name, lt in g["layer_types"].items():
        base = m["layer_types"][name]
        assert {k: v for k, v in lt.items() if k != "pack"} == {k: v for k, v in base.items() if k != "pack"}
        for key in ("pool", "consts"):
            ops, bops = lt["pack"][key], base["pack"][key]
            # same order and offsets: the block route's weights name these ops by index
            assert [o["dst"] for o in ops] == [o["dst"] for o in bops]
            assert all("model.layer" not in o.get("tensor", "") + o.get("up", "") for o in ops)
            expert_ops = {"expert_stripes": "q35_stripes", "expert_down": "q35_down"}
            for o, b in zip(ops, bops):
                if b["op"] in expert_ops:
                    assert o["op"] == expert_ops[b["op"]]
                else:
                    assert o["op"] == "put" or o["op"].startswith("q35_")
                if "nch" in b:
                    assert o["nch"] == b["nch"] and o["in_dim"] == b["in_dim"]
                if "cap" in b and o["op"] in ("put", "q35_vec"):
                    assert o["cap"] == b["cap"]
        gb = lt.get("gemm_block", {})
        for wkey in ("weights", "shared_weights"):
            for w in gb.get(wkey, {}).values():
                src = lt["pack"]["pool" if w["from"] == "pool" else "consts"]
                assert all(src[i]["op"] == "q35_mat" for i in w["ops"])
    assert g["pack"]["embed"]["tensor"] == "token_embd.weight"
    assert g["pack"]["norm"]["tensor"] == "output_norm.weight"
    assert g["pack"]["lm_head"]["ops"][0]["op"] == "q35_lmhead"
    assert "gguf" not in g


def test_view_carries_the_q6_head():
    """OPEN-LMHEAD-Q6: the GGUF view's alternative head -- the Q6_K codes in 6784-B chunks, two cores a column --
    with its own kernels; the base (container) view has none."""
    m, g = _view()
    q6 = g["pack"]["lm_head_q6"]
    op = q6["ops"][0]
    assert op == {"op": "q35_lmhead6", "tensor": "output.weight", "dst": 0, "in_dim": 2048, "groups": 2, "width": 2}
    nch = 248320 // 32 * (2048 // 256)
    assert q6["chunk_bytes"] == 6784 and nch * 6784 <= q6["pool_bytes"] < nch * 6784 + (1 << 20)
    assert g["kernels"]["lm_q6"] == {"context": "lm6", "insts": "lm_head_q6/insts.bin", "build": "lm_head_q6"}
    assert g["contexts"]["lm6"] == "lm_head_q6/final.xclbin"
    assert "lm_head_q6" not in m["pack"] and "lm_q6" not in m["kernels"]


def test_q_gate_split_and_value_head_untile_fields():
    _, g = _view()
    s = default_spec()
    full = {o["tensor"].split(".")[-2] + str(o.get("offset", "")): o for o in g["layer_types"]["full_attention"]["pack"]["pool"] if "tensor" in o}
    assert full["attn_q0"]["period"] == 2 * s.head_dim and full["attn_q0"]["take"] == s.head_dim
    assert full["attn_q" + str(s.head_dim)]["offset"] == s.head_dim
    lin = {o["tensor"].split(".")[-2]: o for o in g["layer_types"]["linear_attention"]["pack"]["pool"] + g["layer_types"]["linear_attention"]["pack"]["consts"] if "tensor" in o}
    assert lin["attn_qkv"]["v0"] == 2 * s.lin_key_heads * s.lin_key_dim and lin["attn_qkv"]["vaxis"] == 0
    assert lin["ssm_out"]["vaxis"] == 1 and lin["ssm_out"]["v0"] == 0
    assert lin["ssm_alpha"]["vdim"] == 1 and lin["ssm_alpha"]["rows"] == s.lin_value_heads


def _q35_map(n, kheads, vperk, vdim, v0):
    """The runtime's Q35Map value-head part (pools.cpp)."""
    i = np.arange(n)
    rel = i - v0
    h = rel // vdim
    src = v0 + ((h % vperk) * kheads + h // vperk) * vdim + rel % vdim
    return np.where(i >= v0, src, i)


def test_untile_inverts_llama_cpp_v_head_reorder():
    # llama.cpp conversion/qwen.py _reorder_v_heads: grouped [K][vperk][d] -> tiled [vperk][K][d]
    kh, vperk, d = 16, 2, 128
    hf = np.arange(kh * vperk * d)
    gguf = hf.reshape(kh, vperk, d).transpose(1, 0, 2).reshape(-1)
    assert np.array_equal(gguf[_q35_map(hf.size, kh, vperk, d, 0)], hf)
    # with a q | k prefix ahead of the V rows (attn_qkv, the conv channels)
    pre = 2 * kh * 128
    rows = np.arange(pre + hf.size)
    g2 = np.concatenate([rows[:pre], pre + gguf])
    assert np.array_equal(g2[_q35_map(rows.size, kh, vperk, d, pre)], rows)


def test_no_view_for_formats_the_q35_ops_do_not_write():
    s = default_spec()
    assert Q.gguf_view(replace(s, quant={**s.quant_map, "experts": "q8"}), manifest(s)) is None
    assert Q.gguf_view(replace(s, quant="q4_1_f32"), manifest(s)) is None


# Traces: OPEN-GGUF-Q8-ROLES-QWEN36MOE
def test_q8_roles_view_streams_q8_and_routes_bf16():
    s = replace(default_spec(), quant={"attn": "q8", "linear": "q8", "linear_out": "q8"})
    m = manifest(s)
    g = m["gguf"]
    for name, lt in g["layer_types"].items():
        base = m["layer_types"][name]
        for key in ("pool", "consts"):
            for o, b in zip(lt["pack"][key], base["pack"][key]):
                if b["op"] == "q8_perm":
                    assert o["op"] == "q35_q8" and o["nch"] == b["nch"] and o["in_dim"] == b["in_dim"]
                if b["op"] == "std_perm":
                    assert o["op"] == "q35_mat"
        packed = [w for key in ("weights", "shared_weights")
                  for w in lt["gemm_block"].get(key, {}).values() if w["from"] == "pack"]
        assert packed
        for w, bw in zip(packed, [w for key in ("weights", "shared_weights")
                                  for w in base["gemm_block"].get(key, {}).values() if w["from"] == "pack"]):
            for o, b in zip(w["pack"], bw["pack"]):
                assert b["op"] == "bf16_gemm" and o["op"] == "q35_bf16"
                assert (o["nch"], o["in_dim"], o["dst"], o.get("via")) == (b["nch"], b["in_dim"], b["dst"], b.get("via"))
    q = [o for o in g["layer_types"]["full_attention"]["pack"]["pool"] if o.get("tensor", "").endswith("attn_q.weight")]
    assert [o["op"] for o in q] == ["q35_q8", "q35_q8"] and sorted(o["offset"] for o in q) == [0, s.head_dim]


# Traces: OPEN-SHARED-HOST-Q8
def test_q8_shared_expert_runs_on_the_host_in_decode():
    s = replace(default_spec(), quant={"attn": "q8", "linear": "q8", "linear_out": "q8", "shared": "q8"})
    m, base = manifest(s), manifest(replace(s, quant={"attn": "q8", "linear": "q8", "linear_out": "q8"}))
    g = m["gguf"]
    for name, lt in g["layer_types"].items():
        prog = lt["program"]
        assert [st["op"] for st in prog] == ["run", "moeroute2", "shared_q8", "run"]
        st, mx = prog[2], lt["gemm_block"]
        assert st["kernel"] == prog[3]["kernel"] and (st["a_xm"], st["a_res"]) == (mx["a_xm"], mx["a_res"])
        assert (st["up"], st["gate"], st["down"], st["gate_inp"]) == (
            "blk.{l}.ffn_up_shexp.weight", "blk.{l}.ffn_gate_shexp.weight", "blk.{l}.ffn_down_shexp.weight",
            "blk.{l}.ffn_gate_inp_shexp.weight")
        assert m["layer_types"][name]["program"][2]["up"] == "model.layer.{l}.mlp.share_up_exps_proj.weight"
        # the pool layout does not move; the block route's shared GEMMs read the q8 codes
        assert lt["pack"]["pool"] == base["gguf"]["layer_types"][name]["pack"]["pool"]
        for w in lt["gemm_block"]["shared_weights"].values():
            assert w["from"] == "pack" and all(o["op"] == "q35_bf16" and "via" not in o for o in w["pack"])
        for w in lt["gemm_block_variants"]["lean"]["shared_weights"].values():
            assert [o["split"] for o in w["pack"]][:1] == ["hi"] and all(o["op"] == "q35_mat" for o in w["pack"])
    assert [st["op"] for st in base["gguf"]["layer_types"]["full_attention"]["program"]] == ["run", "moeroute2", "run"]


def test_equal_head_counts_untile_nothing():
    s = default_spec()
    assert Q.gguf_vtile(replace(s, lin_value_heads=s.lin_key_heads), 0, 128) == {}
    with pytest.raises(Q.OpRangeError):
        Q.gguf_vtile(replace(s, lin_value_heads=24), 0, 128)
