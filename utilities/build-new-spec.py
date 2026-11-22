"""Derive, validate, build and register a kernel spec for an existing family."""
from __future__ import annotations

import argparse
import re
from devenv import ROOT, harness_binary, toolchain
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from pathlib import Path

SPECS = ROOT / "open_kernels" / "recipes" / "specs"
PROBE = ROOT / "open_kernels" / "designs" / "attn" / "probe_dense.py"
BUILD_DESIGN = ROOT / "open_kernels" / "build_design.py"
MAKE_TEST = ROOT / "open_kernels" / "designs" / "attn" / "make_test.py"
COMPARE = ROOT / "open_kernels" / "designs" / "attn" / "compare.py"
EXPORT = ROOT / "utilities" / "export-kernels.py"
GEN_SPEC = ROOT / "utilities" / "gen-kernel-spec.py"
REGISTER = ROOT / "utilities" / "register-spec.py"
OP_GATE = ROOT / "utilities" / "op-gate.py"

def run(cmd, env=None, cwd=None, check=True, quiet=False):
    cmd = [str(c) for c in cmd]
    if not quiet:
        print("+", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, env=env or os.environ, cwd=cwd or str(ROOT))
    if check and p.returncode != 0:
        raise SystemExit(f"build-new-spec: FAILED: {' '.join(cmd)}")
    return p.returncode


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def preflight(spec_path: Path) -> tuple[list[str], list[tuple[str, dict]]]:
    """(the RECIPE's walls, the missing CATALOGUE points) in one pass.

    The recipe is the FAMILY's (`recipes/families.for_spec`), not `dense`:
    Qwen3.5 / LFM2 / the MoE each have their own, and asking `dense` about them
    raises "dense recipe given a 'qwen35' spec" -- which used to be printed as a
    catalogue problem and swallow the real validation of those families
    entirely.

    The two kinds are different: a RECIPE wall (a width that does not fit a
    core's L1) means this size is not supported by this recipe at all, and no
    gate can fix it. A missing CATALOGUE point is a claim the hardware has not
    proved yet -- utilities/op-gate.py proves it.
    """
    sys.path.insert(0, str(ROOT / "open_kernels"))
    from recipes import catalogue as C
    from recipes.families import for_spec
    from recipes.load import load_spec
    from recipes.catalogue import OpRangeError
    spec = load_spec(spec_path)
    walls = []
    with C.collecting() as missing:
        try:
            for_spec(spec).recipe(spec)
        except OpRangeError as exc:
            walls.append(str(exc))
    return walls, missing


def build_probe(py: Path, env: dict, spec_path: Path, work: Path, rows: int) -> Path:
    out = work / "probe"
    e = dict(env)
    e["ATTN_PROBE_SPEC"] = str(spec_path)
    e["ATTN_PROBE_ROWS"] = str(rows)
    run([py, BUILD_DESIGN, PROBE, out], env=e)
    return out


def attn_gate(py: Path, env: dict, spec_path: Path, work: Path, rows: int) -> bool:
    """Build the attention probe at this geometry, run it, score it."""
    out = build_probe(py, env, spec_path, work, rows)
    run([py, MAKE_TEST, "--spec", spec_path, "--rows", rows, "--out", out], env=env)
    exe = harness_binary(work, env)
    run([exe, out / "attention.cfg"], env=env, cwd=out, quiet=True)
    rc = run([py, COMPARE, "--out", out], env=env, check=False)
    return rc == 0


def attn_gate_support(spec_path: Path) -> tuple[bool, str]:
    """Whether designs/attn/probe_dense.py can gate this spec, and why not.

    The probe covers the single-row and RB 1/2/4 block paths. A q/k/v-bias
    family used to be refused here (that point needed the manual OPEN-ATTN-QKV-BIAS
    whole-layer procedure), but the probe now streams the bias fifo the production
    worker uses, so those specs gate like any other. The capability is still
    decided from the spec BEFORE exporting kernels: a refusal at step 3b used to
    arrive after a full, successful export and read as a hard failure.
    """
    sys.path.insert(0, str(ROOT / "open_kernels"))
    from recipes.families import for_spec
    from recipes.load import load_spec
    s = load_spec(spec_path)
    mod = for_spec(s)
    if not hasattr(mod, "geometry"):
        # The attention probe is the DENSE worker's gate. MoE families expose
        # recipe()/layout() instead; their attention point enters through the
        # family bring-up procedure, not this probe. Say so instead of crashing.
        return False, (f"the attention probe only gates dense-family geometries; "
                       f"family {s.family!r} needs its family procedure")
    g = mod.geometry(s)
    if g.RB not in (1, 2, 4):
        return False, f"RB={g.RB} (the probe covers only RB 1/2/4)"
    return True, ""


def run_onboarding(args, spec_path: Path, name: str, generated: bool) -> int:
    sys.path.insert(0, str(ROOT / "open_kernels"))
    from recipes.families import family_from_config_dict
    from recipes.load import load_spec

    print(f"-- spec {name} ({spec_path})", flush=True)
    written = json.loads(spec_path.read_text(encoding="utf-8"))
    if "link_to" in written:
        print(f"RC=0, spec ready (existing kernel set {written['link_to']})")
        return 0
    spec = load_spec(spec_path)
    family = family_from_config_dict({"model_type": (spec.extra or {}).get("model_type")})
    print(f"   family {spec.family}" + (f" (checkpoint family {family})" if family != spec.family and family else ""))
    walls, missing = preflight(spec_path)
    for msg in walls:
        print(f"   recipe constraint: {msg}")
    for op, kw in missing:
        print(f"   catalogue gate: {op} {kw}")

    def park(reason: str) -> int:
        print(reason, file=sys.stderr)
        if generated:
            dest = SPECS.parent / "candidates" / spec_path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copyfile(spec_path, dest)
                print(f"   candidate saved to {dest}", file=sys.stderr)
            else:
                print(f"   existing candidate kept at {dest}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("   plan: gate missing catalogue points, build kernels and register the spec")
        return int(bool(walls))
    if walls:
        return park(walls[0])

    op_gate = _load_module("_op_gate", OP_GATE)
    for op, _ in missing:
        if op != "attn" and op not in op_gate.GATE_FN:
            return park(f"NOT_AUTOMATED: {op}: {op_gate.NOT_AUTOMATED.get(op, 'no automated fixture')}")
    attn_points = {tuple(sorted(kw.items())) for op, kw in missing if op == "attn"}
    if attn_points and not args.skip_gate:
        if len(attn_points) > 1:
            return park("NOT_AUTOMATED: the attention variants must pass separate fixtures")
        supported, reason = attn_gate_support(spec_path)
        if not supported:
            return park(f"NOT_AUTOMATED: attention: {reason}")

    py, env = toolchain()
    work = (Path(args.work) if args.work else ROOT / "build" / "attn-gate" / name).resolve()
    work.mkdir(parents=True, exist_ok=True)
    if missing and not args.skip_gate:
        try:
            if any(op != "attn" for op, _ in missing):
                op_gate.gate_points(spec, spec_path, py, env, harness_binary(work, env), skip=("attn",))
            if attn_points:
                if not attn_gate(py, env, spec_path, work, args.rows):
                    return park("attention gate failed")
                op_gate.promote_point("attn", dict(next(iter(attn_points))), f"{name} (attention gate)")
        except (op_gate.GateError, subprocess.CalledProcessError) as exc:
            return park(str(exc))
        op_gate._reload_catalogue()
        walls, pending = preflight(spec_path)
        if walls or pending:
            return park(f"geometry still unvalidated: {walls or pending}")

    if not args.skip_export:
        print("-- build the kernel set", flush=True)
        if run([py, EXPORT, "--specs", spec_path.resolve(), "--skip-bert", "--skip-diffusion"],
               env=env, check=False):
            return park("kernel export failed")
    if args.skip_gate or args.skip_export:
        print("   partial run complete")
        return 0

    dest = SPECS / spec_path.name
    copied = spec_path.resolve() != dest.resolve() and not dest.exists()
    if spec_path.resolve() != dest.resolve() and dest.exists() and dest.read_bytes() != spec_path.read_bytes():
        return park(f"a different shipped spec exists at {dest}")
    if copied:
        shutil.copyfile(spec_path, dest)
    rc = run([sys.executable, REGISTER, dest], env=env, check=False)
    if rc:
        if copied:
            dest.unlink()
        return park("registration failed")
    print(f"RC=0, spec ready: {dest}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--repo", help="Hugging Face repository")
    source.add_argument("--gguf", type=Path, help="local GGUF checkpoint")
    source.add_argument("--spec", help="existing spec name or path")
    ap.add_argument("--spec-name", default="", help="name for a generated spec")
    ap.add_argument("--rows", type=int, default=17)
    ap.add_argument("--work", default="")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--skip-gate", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.spec:
        path = Path(args.spec)
        if not path.is_file():
            path = SPECS / (args.spec if args.spec.endswith(".json") else f"{args.spec}.json")
        if not path.is_file():
            ap.error(f"no such spec: {args.spec}")
        return run_onboarding(args, path, path.stem, generated=False)

    name = args.spec_name or (args.repo.rsplit("/", 1)[-1].lower() if args.repo else args.gguf.stem.lower())
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+", name) or name in (".", ".."):
        ap.error(f"invalid spec name: {name!r}")
    with TemporaryDirectory(prefix="open-kernels-spec-") as tmp:
        spec_path = Path(tmp) / f"{name}.json"
        src = ["--repo", args.repo] if args.repo else ["--gguf", args.gguf.resolve()]
        rc = run([sys.executable, GEN_SPEC, *src, "--out", spec_path], check=False)
        if rc or not spec_path.is_file():
            return rc or 1
        return run_onboarding(args, spec_path, name, generated=True)


if __name__ == "__main__":
    raise SystemExit(main())
