"""Exercise the BERT exporter check that guards long-context artifacts."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


CHECKER = (Path(__file__).resolve().parents[3] /
           "npu_offload" / "gemm_rtp" / "check_design_sets.py")
spec = importlib.util.spec_from_file_location("check_npue_design_sets", CHECKER)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


@pytest.mark.parametrize(
    ("seq", "rows", "omit_last", "expected_exit"),
    [(512, 1024, False, 0), (64, 1024, False, 1),
     (512, 128, False, 1), (512, 1024, True, 1)],
)
def test_design_sequence_matches_export_flags_and_stream_rows(
    tmp_path, monkeypatch, seq, rows, omit_last, expected_exit
):
    root = tmp_path / "xclbins"
    design_dir = root / "bert" / "gemm_rtp"
    design_dir.mkdir(parents=True)
    family_spec = tmp_path / "families.json"
    flags = ["--seq", "512", "--batches", "2", "--tg-depth", "2"]
    family_spec.write_text(json.dumps({
        "common": flags,
        "families": [{"name": "bert", "args": []}],
    }), encoding="utf-8")
    design = checker.expected(flags)
    operations = ("qkv", "attn_out", "ffn_up", "ffn_down")
    if omit_last:
        operations = operations[:-1]
    design.update(seq=seq, tile={"n": design.pop("tile_n")},
                  streams=[{"op": op, "batch": 2, "M": rows} for op in operations])
    (design_dir / "design.json").write_text(json.dumps(design), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        str(CHECKER), "--xclbins", str(root), "--spec", str(family_spec),
    ])
    assert checker.main() == expected_exit
