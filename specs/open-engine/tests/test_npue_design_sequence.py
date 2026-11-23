"""Exercise the BERT exporter check that guards long-context artifacts."""
import importlib.util
import json
from pathlib import Path
import runpy
import sys

import pytest


CHECKER = (Path(__file__).resolve().parents[3] /
           "npu_offload" / "gemm_rtp" / "check_design_sets.py")
spec = importlib.util.spec_from_file_location("check_npue_design_sets", CHECKER)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


@pytest.mark.parametrize(
    ("defect", "expected_exit"),
    [(None, 0), ("seq", 1), ("rows", 1), ("stream", 1), ("file", 1), ("binding", 1)],
)
def test_design_sequence_matches_export_flags_and_stream_rows(
    tmp_path, monkeypatch, defect, expected_exit
):
    root = tmp_path / "xclbins"
    design_dir = root / "bert" / "gemm_rtp"
    design_dir.mkdir(parents=True)
    family_spec = tmp_path / "families.json"
    flags = ["--seq", "512", "--batches", "2", "--tg-depth", "2"]
    family_spec.write_text(json.dumps({
        "common": flags,
        "families": [{"name": "bert", "args": []}, {"name": "unbuilt", "args": []}],
    }), encoding="utf-8")
    design = checker.expected(flags)
    operations = ("qkv", "attn_out", "ffn_up", "ffn_down")
    if defect == "stream":
        operations = operations[:-1]
    names = [f"insts_{op}_b2.bin" for op in operations]
    streams = [
        {"op": op, "batch": 2, "M": 128 if defect == "rows" else 1024,
         "slot": slot, "file": name}
        for slot, (op, name) in enumerate(zip(operations, names), start=1)
    ]
    if defect == "binding":
        streams[-1]["slot"] = streams[0]["slot"]
    files = ["final.xclbin", "insts.bin", *names]
    if defect == "file":
        files.pop()
    for name in files:
        (design_dir / name).write_bytes(b"fixture")
    design.update(seq=64 if defect == "seq" else 512,
                  tile={"n": design.pop("tile_n")}, streams=streams)
    (design_dir / "design.json").write_text(json.dumps(design), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        str(CHECKER), "--xclbins", str(root), "--spec", str(family_spec), "--only", "bert",
    ])
    assert checker.main() == expected_exit


def test_derive_sequence_reuses_the_image_and_compiled_rows(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(CHECKER.parent))
    derive = runpy.run_path(str(CHECKER.with_name("derive_sequence.py")))["derive"]
    source = tmp_path / "source"
    bundle = source / "gemm_rtp"
    bundle.mkdir(parents=True)
    (bundle / "final.xclbin").write_bytes(b"static image")
    (bundle / "insts.bin").write_bytes(b"bootstrap")
    streams = []
    for slot, op in enumerate(("qkv", "attn_out", "ffn_up", "ffn_down"), start=1):
        filename = f"insts_{op}_b8.bin"
        (bundle / filename).write_bytes(op.encode())
        streams.append(dict(op=op, M=512, K=384, N=384, batch=8, slot=slot, file=filename))
    design = dict(M=512, seq=64, batch=8, tiers=[8], streams=streams)
    (bundle / "design.json").write_text(json.dumps(design), encoding="utf-8")

    output = tmp_path / "derived"
    with pytest.raises(ValueError, match="no compiled stream"):
        derive(source, output, 256, [1, 2])
    assert not output.exists()

    derive(source, output, 512, [1])
    result = output / "gemm_rtp"
    derived = json.loads((result / "design.json").read_text(encoding="utf-8"))
    assert (derived["M"], derived["seq"], derived["batch"], derived["tiers"]) == (512, 512, 1, [1])
    assert (result / "final.xclbin").read_bytes() == (bundle / "final.xclbin").read_bytes()
    assert (result / "insts.bin").read_bytes() == (bundle / "insts.bin").read_bytes()
    for old, new in zip(streams, derived["streams"], strict=True):
        assert new == dict(old, batch=1, file=f"insts_{old['op']}_b1.bin")
        assert (result / new["file"]).read_bytes() == (bundle / old["file"]).read_bytes()
