"""klein_gguf: a FLUX.2 [klein] DiT GGUF as diffusers-named HF-layout weights; ported from OFLM c2a9e094 (2026-11-14)."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 8: "Q8_0", 30: "BF16"}
BLOCK = {"F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "Q4_0": (32, 18), "Q4_1": (32, 20),
         "Q8_0": (32, 34)}
# Best first. OFLM measured Q8_0 at 0.9946 correlation with bf16 and Q4_1 visibly grainy.
QUANTS = ["Q8_0", "BF16", "F16", "Q5_1", "Q5_0", "Q4_1", "Q4_0"]
LOSSY = ("Q4_1", "Q4_0")
MOD_TENSOR = "final_layer.adaLN_modulation.1.weight"


class Gguf:
    """A local GGUF's metadata, tensor table and data section."""

    def __init__(self, path):
        self.path = Path(path)
        self._f = open(self.path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        if self._mm[:4] != b"GGUF":
            raise ValueError(f"{path} is not a GGUF")
        self._parse()

    def _parse(self):
        mm = self._mm
        version, ntensors, nkv = struct.unpack_from("<IQQ", mm, 4)
        if version < 2:
            raise ValueError(f"{self.path}: GGUF v{version} unsupported")
        p = 24

        def rd_str():
            nonlocal p
            (n,) = struct.unpack_from("<Q", mm, p)
            p += 8
            s = mm[p:p + n].decode("utf-8", "replace")
            p += n
            return s

        fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
               10: "<Q", 11: "<q", 12: "<d"}

        def rd_value(t):
            nonlocal p
            if t == 8:
                return rd_str()
            if t == 9:
                et, n = struct.unpack_from("<IQ", mm, p)
                p += 12
                return [rd_value(et) for _ in range(n)]
            if t not in fmt:
                raise ValueError(f"GGUF metadata type {t} not handled")
            v = struct.unpack_from(fmt[t], mm, p)[0]
            p += struct.calcsize(fmt[t])
            return v

        self.kv = {}
        for _ in range(nkv):
            k = rd_str()
            (t,) = struct.unpack_from("<I", mm, p)
            p += 4
            self.kv[k] = rd_value(t)
        self.tensors = {}
        for _ in range(ntensors):
            name = rd_str()
            (nd,) = struct.unpack_from("<I", mm, p)
            p += 4
            dims = list(struct.unpack_from("<" + "Q" * nd, mm, p))
            p += 8 * nd
            ty, off = struct.unpack_from("<IQ", mm, p)
            p += 12
            if ty not in TYPES:
                kind = "a K-quant" if 10 <= ty <= 23 else "unsupported"
                raise ValueError(f"{self.path}: {name} is type {ty} ({kind}); use a Q8_0, BF16 "
                                 "or F16 klein GGUF")
            self.tensors[name] = (TYPES[ty], dims, off)
        align = int(self.kv.get("general.alignment", 32))
        self.data_base = p + (-p) % align

    def names(self) -> set[str]:
        return set(self.tensors)

    def shape(self, name) -> list[int]:
        return list(reversed(self.tensors[name][1]))

    def types(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for typ, _, _ in self.tensors.values():
            out[typ] = out.get(typ, 0) + 1
        return out

    def array(self, name) -> np.ndarray:
        """f32 in HF layout: ggml's [in, out] fastest-first is the same bytes as row-major [out, in]."""
        typ, dims, off = self.tensors[name]
        n = int(np.prod(dims, dtype=np.int64))
        per, width = BLOCK[typ]
        start = self.data_base + off
        raw = np.frombuffer(self._mm, np.uint8, n // per * width, start)
        return decode(raw, typ, n).reshape(dims[::-1])


def decode(buf: np.ndarray, typ: str, n: int) -> np.ndarray:
    if typ == "F32":
        return buf.view("<f4")[:n].astype(np.float32)
    if typ == "F16":
        return buf.view("<f2")[:n].astype(np.float32)
    if typ == "BF16":
        return (buf.view("<u2")[:n].astype(np.uint32) << 16).view(np.float32)
    if typ == "Q8_0":
        b = buf.reshape(-1, 34)
        d = b[:, 0:2].copy().view("<f2").astype(np.float32)
        q = b[:, 2:34].copy().view(np.int8).astype(np.float32)
        return (d * q).reshape(-1)
    if typ in ("Q4_0", "Q4_1"):
        w = 18 if typ == "Q4_0" else 20
        b = buf.reshape(-1, w)
        d = b[:, 0:2].copy().view("<f2").astype(np.float32)
        q = b[:, w - 16:w]
        lo, hi = (q & 0x0F).astype(np.float32), (q >> 4).astype(np.float32)
        out = np.empty((len(b), 32), np.float32)
        if typ == "Q4_0":
            out[:, :16], out[:, 16:] = d * (lo - 8), d * (hi - 8)
        else:
            m = b[:, 2:4].copy().view("<f2").astype(np.float32)
            out[:, :16], out[:, 16:] = d * lo + m, d * hi + m
        return out.reshape(-1)
    raise ValueError(f"decode: {typ} not implemented")


_TOP = {
    "time_guidance_embed.timestep_embedder.linear_1.weight": "time_in.in_layer.weight",
    "time_guidance_embed.timestep_embedder.linear_2.weight": "time_in.out_layer.weight",
    "context_embedder.weight": "txt_in.weight",
    "x_embedder.weight": "img_in.weight",
    "proj_out.weight": "final_layer.linear.weight",
    "double_stream_modulation_img.linear.weight": "double_stream_modulation_img.lin.weight",
    "double_stream_modulation_txt.linear.weight": "double_stream_modulation_txt.lin.weight",
    "single_stream_modulation.linear.weight": "single_stream_modulation.lin.weight",
    "norm_out.linear.weight": MOD_TENSOR,
}
_DBL_QKV = {"attn.to_q.weight": (0, "img"), "attn.to_k.weight": (1, "img"),
            "attn.to_v.weight": (2, "img"), "attn.add_q_proj.weight": (0, "txt"),
            "attn.add_k_proj.weight": (1, "txt"), "attn.add_v_proj.weight": (2, "txt")}
_DBL = {"attn.to_out.0.weight": "img_attn.proj.weight",
        "attn.to_add_out.weight": "txt_attn.proj.weight",
        "ff.linear_in.weight": "img_mlp.0.weight", "ff.linear_out.weight": "img_mlp.2.weight",
        "ff_context.linear_in.weight": "txt_mlp.0.weight",
        "ff_context.linear_out.weight": "txt_mlp.2.weight",
        "attn.norm_q.weight": "img_attn.norm.query_norm.scale",
        "attn.norm_k.weight": "img_attn.norm.key_norm.scale",
        "attn.norm_added_q.weight": "txt_attn.norm.query_norm.scale",
        "attn.norm_added_k.weight": "txt_attn.norm.key_norm.scale"}
_SGL = {"attn.to_qkv_mlp_proj.weight": "linear1.weight", "attn.to_out.weight": "linear2.weight",
        "attn.norm_q.weight": "norm.query_norm.scale", "attn.norm_k.weight": "norm.key_norm.scale"}


def resolve(name: str) -> tuple[str, int | None] | None:
    """(GGUF tensor, q|k|v slice or None) for a diffusers transformer weight name."""
    if name in _TOP:
        return _TOP[name], None
    for prefix, gg in (("transformer_blocks.", "double_blocks."),
                       ("single_transformer_blocks.", "single_blocks.")):
        if name.startswith(prefix):
            b, _, tail = name[len(prefix):].partition(".")
            if gg == "double_blocks.":
                if tail in _DBL_QKV:
                    part, stream = _DBL_QKV[tail]
                    return f"{gg}{b}.{stream}_attn.qkv.weight", part
                if tail in _DBL:
                    return f"{gg}{b}.{_DBL[tail]}", None
            elif tail in _SGL:
                return f"{gg}{b}.{_SGL[tail]}", None
    return None


class GgufWeights:
    """`SafeTensors.get`-compatible view; bfl_modulation: adaLN halves are BFL's (scale, shift), as unsloth's and leejet's files are."""

    def __init__(self, path, bfl_modulation: bool):
        self.gguf = Gguf(path)
        self.bfl_modulation = bool(bfl_modulation)

    def get(self, name: str) -> np.ndarray:
        r = resolve(name)
        if r is None:
            raise KeyError(f"klein_gguf: no mapping for {name}")
        tensor, part = r
        w = self.gguf.array(tensor)
        if tensor == MOD_TENSOR and self.bfl_modulation:
            half = w.shape[0] // 2
            w = np.concatenate([w[half:], w[:half]], axis=0)
        if part is not None:
            w = np.split(w, 3, axis=0)[part]
        return np.ascontiguousarray(w, np.float32)


def geometry(g: Gguf) -> dict:
    """The transformer config keys check_checkpoint compares, read from tensor shapes."""
    def count(prefix):
        return len({n.split(".")[1] for n in g.names() if n.startswith(prefix)})
    hid, lat = g.shape("img_in.weight")
    hd = g.shape("double_blocks.0.img_attn.norm.query_norm.scale")[-1]
    return {"num_layers": count("double_blocks."), "num_single_layers": count("single_blocks."),
            "num_attention_heads": hid // hd, "attention_head_dim": hd,
            "joint_attention_dim": g.shape("txt_in.weight")[1], "in_channels": lat}


def _mirror_decision(got: np.ndarray, ref: np.ndarray) -> bool | None:
    got, ref = got.astype(np.float64), ref.astype(np.float64)
    rms = float(np.sqrt((ref ** 2).mean())) or 1.0
    half = got.shape[0] // 2
    as_is = float(np.sqrt(((got - ref) ** 2).mean())) / rms
    swapped = float(np.sqrt(((np.concatenate([got[half:], got[:half]]) - ref) ** 2).mean())) / rms
    if swapped < as_is / 10:
        return True
    if as_is < swapped / 10:
        return False
    return None


def _bf16_rows(raw: bytes, shape) -> np.ndarray:
    return (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32).reshape(shape)


def _local_reference(transformer_dir: Path) -> np.ndarray | None:
    for f in sorted(transformer_dir.glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
            meta = hdr.get("norm_out.linear.weight")
            if meta is None:
                continue
            a, b = meta["data_offsets"]
            fh.seek(8 + n + a)
            raw = fh.read(b - a)
        if meta["dtype"] == "BF16":
            return _bf16_rows(raw, meta["shape"])
        return np.frombuffer(raw, {"F32": "<f4", "F16": "<f2"}[meta["dtype"]]).astype(
            np.float32).reshape(meta["shape"])
    return None


def _range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _hub_reference(repo: str, revision: str = "main") -> np.ndarray | None:
    """norm_out.linear.weight from the base repo by HTTP range (36 MB, not the 8 GB file)."""
    url = (f"https://huggingface.co/{repo}/resolve/{revision}/transformer/"
           "diffusion_pytorch_model.safetensors")
    try:
        n = struct.unpack("<Q", _range(url, 0, 7))[0]
        meta = json.loads(_range(url, 8, 8 + n - 1))["norm_out.linear.weight"]
        a, b = meta["data_offsets"]
        if meta["dtype"] != "BF16":
            return None
        return _bf16_rows(_range(url, 8 + n + a, 8 + n + b - 1), meta["shape"])
    except Exception:  # noqa: BLE001 - the caller reports and falls back
        return None


def detect_bfl_modulation(gguf_path: Path, *, transformer_dir: Path | None = None,
                          repo: str | None = None) -> bool | None:
    """True when the GGUF's adaLN halves are BFL-ordered; None when no reference was readable."""
    ref = _local_reference(transformer_dir) if transformer_dir and transformer_dir.is_dir() else None
    if ref is None and repo:
        ref = _hub_reference(repo)
    if ref is None:
        return None
    return _mirror_decision(Gguf(gguf_path).array(MOD_TENSOR), ref)


def file_identity(path: Path) -> str:
    """Name, size and three 1 MB samples: enough to refuse a packed cache from another file."""
    h = hashlib.sha256()
    n = path.stat().st_size
    h.update(f"{path.name}:{n}".encode())
    with open(path, "rb") as f:
        for at in (0, max(0, n // 2 - (1 << 19)), max(0, n - (1 << 20))):
            f.seek(at)
            h.update(f.read(1 << 20))
    return h.hexdigest()


def base_model_from_readme(text: str) -> str:
    """The frontmatter's `base_model` (a string, a JSON list or a YAML list), or ""."""
    if not text.startswith("---"):
        return ""
    end = text.find("---", 3)
    if end == -1:
        return ""
    lines = text[3:end].splitlines()
    for i, line in enumerate(lines):
        key, _, rest = line.partition(":")
        if key.strip() != "base_model":
            continue
        rest = rest.strip()
        if rest.startswith("["):
            try:
                vals = json.loads(rest)
            except ValueError:
                return ""
            return str(vals[0]) if vals else ""
        if rest:
            return rest.strip("'\"")
        for follow in lines[i + 1:]:
            s = follow.strip()
            if s.startswith("- "):
                return s[2:].strip().strip("'\"")
            if s and not s.startswith("#"):
                break
    return ""


def pick_quant(files) -> str:
    """The DiT GGUF to take from a repo's file list, best quant first."""
    dit = [f for f in files if f.lower().endswith(".gguf")
           and "mmproj" not in f.lower() and "mtp" not in f.lower()]
    for q in QUANTS:
        cand = sorted(f for f in dit if q.lower() in f.lower())
        if cand:
            return cand[0]
    raise ValueError(f"no DiT GGUF in {QUANTS} among {sorted(files)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--checkpoint", type=Path, help="diffusers dir: compare the modulation order")
    ap.add_argument("--repo", help="base HF repo: compare the modulation order by range read")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import klein_pipeline as kp

    g = Gguf(a.gguf)
    print(f"{a.gguf.name}: {g.kv.get('general.architecture')} {len(g.names())} tensors {g.types()}")
    print("geometry:", geometry(g))
    bfl = detect_bfl_modulation(a.gguf, transformer_dir=a.checkpoint / "transformer" if a.checkpoint else None,
                                repo=a.repo)
    print("modulation:", {True: "bfl", False: "diffusers", None: "undetermined"}[bfl])
    gw = GgufWeights(a.gguf, bool(bfl))
    failed = 0
    for name, (K, N, build) in sorted(kp.dit_weight_specs().items()):
        try:
            shape = build(gw.get).shape
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
            failed += 1
            continue
        if shape != (K, N):
            print(f"  FAIL {name}: {shape} != {(K, N)}")
            failed += 1
    print(f"{len(kp.dit_weight_specs())} schedule weights, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
