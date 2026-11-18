r"""flux2_native_generate: FLUX.2 [klein] text-to-image and one-reference edits, every op on
the NPU, for ``phlegm image`` / ``phlegm serve-images`` (backend ``flux2``).

Runs ``klein_pipeline.plan(R, steps, edit)``'s schedule (text encoder, conditioning, the
denoising steps, the VAE; plus the VAE encoder for an edit) through the pyxrt host
(``harness/npu_host.py``), one hardware context per kernel set. Ported from OFLM's
``utilities/dit-chain/generate.py``; weights are read straight from the published
diffusers checkpoint (``transformer/``, ``text_encoder/``, ``vae/``, ``tokenizer/``), no
q4nx bundle. The reference is ``flux2_klein_reference.py`` (the unchanged diffusers
``Flux2KleinPipeline``).

    python -m open_kernels.flux2_native_generate --model <checkpoint> --kernels <set dir> \
        --prompt "a red fox in fresh snow" --size 512 --seed 1 --out <dir> \
        [--image ref.png] [--noise noise.npy] [--steps 4]
        [--projected-conditioning target.safetensors]  # experimental, skips text encoding
        [--transformer-gguf flux-2-klein-4b-Q8_0.gguf]  # the DiT from a GGUF (klein_gguf.py)

Kernel directory: ``export_dit_kernels.py --resolutions ... [--edits ...] --out <dir>``.
Packed weights are cached in ``<kernels>/packed/`` (dit_gemm's bfp16 layout, ~8 GB for
4B) and ``<kernels>/vae_packed.npz`` / ``venc_packed.npz``; each carries a fingerprint of
the checkpoint it came from.

Noise: ``--noise`` takes the reference's ``noise.npy`` ([128, R/16, R/16]); otherwise
``torch.randn``-compatible noise is not available without torch, so the seed drives
numpy's PCG64 (images are then not comparable with the reference's for the same seed).

Writes ``<out>/image.png``, ``<out>/latents.npy`` (packed, bf16 bits) and
``<out>/summary.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "harness"))
import klein_pipeline as kp  # noqa: E402

SETS = {"gemm": ".", "fa": "fa", "ew": "ew", "conv": "conv", "conv1": "conv1", "vew": "vew"}
_DT = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "I64": np.int64,
       "I32": np.int32, "U8": np.uint8}


class SafeTensors:
    """Tensors of a (sharded) safetensors directory, numpy only; bf16 -> float32."""

    def __init__(self, directory: Path):
        d = Path(directory)
        idx = d / "model.safetensors.index.json"
        files = (sorted(set(json.loads(idx.read_text())["weight_map"].values()))
                 if idx.exists() else [p.name for p in sorted(d.glob("*.safetensors"))])
        self._where, self._maps = {}, {}
        for f in files:
            with open(d / f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            self._maps[f] = (np.memmap(d / f, dtype=np.uint8, mode="r"), 8 + n)
            for name, meta in hdr.items():
                if name != "__metadata__":
                    self._where[name] = (f, meta)

    def names(self) -> list[str]:
        return list(self._where)

    def raw(self, name: str) -> np.ndarray:
        f, meta = self._where[name]
        mm, base = self._maps[f]
        a, b = meta["data_offsets"]
        return np.asarray(mm[base + a:base + b]).view(_DT[meta["dtype"]]).reshape(meta["shape"])

    def get(self, name: str) -> np.ndarray:
        raw = self.raw(name)
        if self._where[name][1]["dtype"] == "BF16":
            return (raw.astype(np.uint32) << 16).view(np.float32)
        return raw.astype(np.float32)


def fingerprint(model: Path, subs: tuple[str, ...], code: str, extra: str = "") -> str:
    """The checkpoint's weight files (name, size, head/middle/tail samples) plus the packing
    code: a packed cache from another checkpoint or packer is refused, not reused."""
    h = hashlib.sha256(code.encode())
    h.update(extra.encode())
    for sub in subs:
        for p in sorted((model / sub).glob("*.safetensors")):
            n = p.stat().st_size
            h.update(f"{sub}/{p.name}:{n}".encode())
            with open(p, "rb") as f:
                for at in (0, max(0, n // 2 - (1 << 19)), max(0, n - (1 << 20))):
                    f.seek(at)
                    h.update(f.read(1 << 20))
    return h.hexdigest()


def dit_reader(model: Path, gguf: tuple[str, bool] | None):
    """The DiT's `get`: the checkpoint's transformer/, or a GGUF (path, BFL modulation order)."""
    if gguf is None:
        return SafeTensors(Path(model) / "transformer")
    from klein_gguf import GgufWeights
    return GgufWeights(gguf[0], gguf[1])


def _pack_one(sub: str, name: str, cache: str, model: str,
              gguf: tuple[str, bool] | None = None) -> tuple[str, float]:
    t0 = time.time()
    specs = kp.dit_weight_specs() if sub == "transformer" else kp.te_weight_specs()
    K, N, build = specs[name]
    src = dit_reader(Path(model), gguf) if sub == "transformer" else SafeTensors(Path(model) / sub)
    b = build(src.get)
    assert b.shape == (K, N), (name, b.shape, (K, N))
    out = Path(cache) / f"{name}.bin"
    tmp = out.with_suffix(".tmp")
    kp.pack_b_cols(b).tofile(tmp)
    tmp.replace(out)
    return name, time.time() - t0


def ensure_packed(cache: Path, model: Path, jobs: int, *, text_encoder: bool = True,
                  gguf: tuple[str, bool] | None = None) -> dict[str, tuple[Path, int]]:
    cache.mkdir(parents=True, exist_ok=True)
    stamp = cache / "provenance.txt"
    subs = (() if gguf else ("transformer",)) + (("text_encoder",) if text_encoder else ())
    extra = ""
    if gguf:
        from klein_gguf import file_identity
        extra = f"gguf:{file_identity(Path(gguf[0]))}:bfl={gguf[1]}"
    fp = fingerprint(model, subs, inspect.getsource(kp.pack_b_cols), extra)
    if stamp.exists() and stamp.read_text().strip() != fp:
        print(f"{cache}: packed from another checkpoint or packer; repacking", flush=True)
        for p in cache.glob("*.bin"):
            p.unlink()
    stamp.write_text(fp + "\n")
    want = {n: ("transformer", K * N * 9 // 8) for n, (K, N, _) in kp.dit_weight_specs().items()
            if text_encoder or n != "ctx_emb"}
    if text_encoder:
        want |= {n: ("text_encoder", K * N * 9 // 8) for n, (K, N, _) in kp.te_weight_specs().items()}
    todo = [n for n, (_, nb) in want.items()
            if not (cache / f"{n}.bin").exists() or (cache / f"{n}.bin").stat().st_size != nb]
    if todo:
        print(f"packing {len(todo)} weights -> {cache} ({jobs} processes)", flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(jobs) as ex:
            for i, (n, s) in enumerate(ex.map(_pack_one, [want[n][0] for n in todo], todo,
                                              [str(cache)] * len(todo),
                                              [str(model)] * len(todo),
                                              [gguf] * len(todo))):
                if i % 20 == 0 or i == len(todo) - 1:
                    print(f"  [{i + 1}/{len(todo)}] {n} {s:.1f} s", flush=True)
        print(f"packed in {time.time() - t0:.0f} s", flush=True)
    return {n: (cache / f"{n}.bin", nb) for n, (_, nb) in want.items()}


def packed_coder(cache: Path, model: Path, coder):
    """vae_decoder / vae_encoder pack_weights output, cached with the checkpoint's VAE
    fingerprint: (weight bytes, table, group-norm blocks, index)."""
    fp = fingerprint(model, ("vae",), inspect.getsource(coder.pack_weights))
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if str(z["fingerprint"]) == fp:
            return z["w"], z["table"].item(), z["blocks"].view(bfloat16), z["index"].item()
    t0 = time.time()
    st = SafeTensors(model / "vae")
    sd = {k: st.get(k) for k in st.names() if not k.endswith("num_batches_tracked")}
    w, table, blocks, index = coder.pack_weights(sd)
    np.savez(cache, w=w, table=np.array(table, dtype=object), blocks=blocks.view(np.uint16),
             index=np.array(index, dtype=object), fingerprint=np.array(fp))
    print(f"packed {coder.__name__} in {time.time() - t0:.0f} s -> {cache}", flush=True)
    return w, table, blocks, index


def embed_rows(st: SafeTensors, ids: np.ndarray) -> np.ndarray:
    """model.embed_tokens rows for ids (bf16 bits), zero-padded to TE_PAD: XT. A BF16 table
    is copied bit for bit; any other dtype (an F16 re-save) is rounded to bf16, not
    reinterpreted."""
    name = "model.embed_tokens.weight"
    rows = st.raw(name)[ids]
    if st._where[name][1]["dtype"] != "BF16":
        rows = rows.astype(np.float32).astype(bfloat16).view(np.uint16)
    x = np.zeros((len(ids), kp.TE_PAD), np.uint16)
    x[:, :rows.shape[1]] = rows
    return x


def check_checkpoint(model: Path, *, projected: bool = False, gguf: Path | None = None) -> dict:
    """The schedule's constants against the checkpoint's configs (or the GGUF's tensor
    shapes): refuse a different geometry (another klein size) instead of running it with
    the wrong layout."""
    if gguf is not None:
        from klein_gguf import Gguf, geometry
        tc = geometry(Gguf(gguf))
    else:
        tc = json.loads((model / "transformer" / "config.json").read_text())
    want_t = {"num_layers": kp.N_DBL, "num_single_layers": kp.N_SGL,
              "num_attention_heads": kp.HEADS, "attention_head_dim": kp.HD,
              "joint_attention_dim": 3 * kp.TE_HID, "in_channels": kp.LAT_CH}
    want_e = {"hidden_size": kp.TE_HID, "intermediate_size": kp.TE_MLP}
    bad = [f"transformer.{k}={tc.get(k)} (schedule {v})" for k, v in want_t.items() if tc.get(k) != v]
    if not projected:
        te = json.loads((model / "text_encoder" / "config.json").read_text())
        bad += [f"text_encoder.{k}={te.get(k)} (schedule {v})" for k, v in want_e.items() if te.get(k) != v]
    if bad:
        raise SystemExit("checkpoint geometry differs from klein_pipeline's: " + "; ".join(bad))
    mi = json.loads((model / "model_index.json").read_text())
    if not mi.get("is_distilled", False):
        raise SystemExit("undistilled (Base) checkpoints need CFG, which this route does not run")
    return {"transformer": {k: tc[k] for k in want_t}, "is_distilled": True}


class Runner:
    def __init__(self, model: Path, kdir: Path, R: int, steps: int, edit: bool, pack_jobs: int,
                 *, projected: bool = False, gguf: tuple[str, bool] | None = None,
                 elf: bool = False):
        from npu_host import Npu
        import vae_decoder as vd

        t0 = time.time()
        self.R, self.steps = R, steps
        self.projected, self._conditioning_bits = projected, None
        self.pl = kp.plan(R, steps=steps, edit=edit)
        self.dit = dit_reader(model, gguf)
        self.te = None if projected else SafeTensors(model / "text_encoder")
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(model / "tokenizer" / "tokenizer.json"))
        cache = "packed_projected" if projected else "packed"
        if gguf:
            cache += "_gguf_" + Path(gguf[0]).stem
        packed = ensure_packed(kdir / cache, model, pack_jobs, text_encoder=not projected, gguf=gguf)

        self.elf = elf
        if elf:
            manifest = json.loads((kdir / "diffusion_elf.json").read_text())
            if self.pl.key not in manifest["elf"]:
                raise SystemExit(f"{kdir}: no full ELF for configuration {self.pl.key}")
            from npu_host import ElfNpu
            self.npu = npu = ElfNpu(kdir / manifest["elf"][self.pl.key], manifest["cfg"])
            self.vl_head = manifest["valid_len"]["head"]
            self.vl = int(manifest["valid_len"]["max"])
        else:
            self.npu = npu = Npu()
            self.sets = {n: npu.kernel_set(n, kdir / d) for n, d in SETS.items()}
        if not projected and not elf:
            fa_meta = json.loads((kdir / "fa" / "dit_fa.json").read_text())
            self.vl_words = fa_meta["patch"]["te_attn"]["valid_len"]

        self.bufs = {n: npu.buf(n, nb) for n, nb in self.pl.buffers.items()}
        for b in self.bufs.values():
            b.zero()
        self.bufs["PARAMS"].write(kp.fill_params(self.dit.get, self.te.get if self.te else None))
        qin = self.pl.vae.buffers["QIN"]
        q = np.zeros((qin.H * qin.W, qin.C), bfloat16)
        q[:, 512] = 1                                  # the VAE attention GEMM's bias row
        self.bufs["v_QIN"].write(q)
        wbytes, self.vtable, blocks, self.vindex = packed_coder(kdir / "vae_packed.npz", model, vd)
        self.vW, self.vS = npu.buf("vae_W", wbytes.size), npu.buf("vae_S", blocks.size * 2)
        self.vW.write(wbytes)
        self.vS.write(blocks)
        self.blk_bytes = vd.BLOCK * vd.EL * 2
        if edit:
            import vae_encoder as ve
            wbytes, self.etable, blocks, self.eindex = packed_coder(kdir / "venc_packed.npz",
                                                                    model, ve)
            self.eW, self.eS = npu.buf("venc_W", wbytes.size), npu.buf("venc_S", blocks.size * 2)
            self.eW.write(wbytes)
            self.eS.write(blocks)

        t1 = time.time()
        self.w = {}
        for n, (path, nb) in packed.items():
            self.w[n] = npu.buf(f"w:{n}", nb)
            self.w[n].load(path)
        gb = sum(nb for _, nb in packed.values()) / 2 ** 30
        print(f"weights: {gb:.2f} GiB in {time.time() - t1:.1f} s", flush=True)

        self.sig = kp.sigmas(R, steps)
        self.bufs["TF"].write(kp.timestep_features(self.sig, steps))
        self.bufs["DT"].write(kp.dt_params(self.sig, steps))
        self.ops = []
        for o in self.pl.ops:
            if projected and o["phase"] == "text":
                continue
            if projected and o["what"] == "ctx_emb":
                # Host input boundary at every step; X's text rows were mutated by the last step.
                self.ops.append((o["set"], None, [], o["phase"], o))
                continue
            if elf:
                st = f"{o['set']}:{o['stream']}"
                npu.kernel(st)
            else:
                st = self.sets[o["set"]].stream(o["stream"])
            self.ops.append((o["set"], st, [self._arg(a) for a in o["args"]], o["phase"], o))
        if not projected and not elf:
            self.vl_stream = self.sets["fa"].stream("te_attn")
        self.load_s = time.time() - t0
        print(f"loaded {R}x{R} ({'edit' if edit else 't2i'}, {steps} steps): "
              f"{self.dispatch_count} dispatches in {self.load_s:.1f} s", flush=True)

    @property
    def dispatch_count(self) -> int:
        return sum(st is not None for _, st, _, _, _ in self.ops)

    def _arg(self, ref):
        kind = ref[0]
        if kind == "buf":
            _, n, off, nb = ref
            return self.bufs[n].view(off, nb)
        if kind == "w":
            return self.w[ref[1]].bo
        if kind == "vae_w":
            off, n = self.vtable[ref[1]]
            return self.vW.view(off, n)
        if kind == "vae_gn":
            return self.vS.view(self.vindex[ref[1]] * self.blk_bytes, self.blk_bytes)
        if kind == "venc_w":
            off, n = self.etable[ref[1]]
            return self.eW.view(off, n)
        if kind == "venc_gn":
            return self.eS.view(self.eindex[ref[1]] * self.blk_bytes, self.blk_bytes)
        raise ValueError(ref)

    def set_prompt(self, prompt: str) -> tuple[np.ndarray, int]:
        if self.projected:
            raise ValueError("use set_projected_conditioning in projected-input mode")
        ids, n_real = kp.token_ids(self.tok, prompt)
        self.bufs["XT"].write(embed_rows(self.te, ids))
        if getattr(self, "elf", False):
            self.vl = n_real
        else:
            self.vl_stream.patch(self.vl_words, n_real)
        return ids, n_real

    def set_projected_conditioning(self, conditioning, prompt: str) -> tuple[np.ndarray, int]:
        if not self.projected:
            raise ValueError("runner was not created for projected conditioning")
        # Invalidate the old input even if the new prompt check fails.
        self._conditioning_bits = None
        ids, n_real = kp.token_ids(self.tok, prompt)
        conditioning.require_prompt(ids, n_real)
        self._conditioning_bits = conditioning.bits.copy()
        return ids, n_real

    def set_noise(self, lat_bits: np.ndarray) -> None:
        self.bufs["LAT"].write(np.ascontiguousarray(lat_bits, np.uint16))

    def set_reference(self, rgb: np.ndarray) -> None:
        """uint8 [R, R, 3] as bf16 2 (x / 255) - 1 in channels 0-2 of the encoder's
        zero-bordered input (diffusers' float32 arithmetic, then bf16)."""
        b = self.pl.enc.buffers["IN"]
        x = np.zeros((b.H + 2, b.pitch, b.C), bfloat16)
        x[1:b.H + 1, 1:b.W + 1, :3] = (np.float32(2) * (rgb.astype(np.float32) / np.float32(255))
                                       - np.float32(1)).astype(bfloat16)
        self.bufs[self.pl.enc_buffers["IN"]].write(x)

    def _elf_runs(self, kset, name, o):
        """The runs an op adds to a stretch: te_attn's valid_len head first, then the op."""
        pre = []
        if o["stream"] == "te_attn":
            pre.append(self.npu.run(f"fa:{self.vl_head.format(n=self.vl)}"))
        return pre

    def _run_elf(self, profile: bool = False) -> dict:
        """One runlist per set change: its configure kernel (alternate _a/_b so the firmware resets) and the ops after it."""
        from collections import deque
        npu = self.npu
        phases: dict[str, float] = {}
        prof: dict[str, dict] = {}
        inflight: deque = deque()
        cur: list = []
        flip = 0

        def close():
            if not cur:
                return
            if len(inflight) >= 32:
                wait_oldest()
            cur[0].execute()
            inflight.append(tuple(cur))
            cur.clear()

        def wait_oldest():
            lst, _runs, what = inflight.popleft()
            try:
                lst.wait()
            except Exception as e:  # noqa: BLE001 - name the stretch
                raise RuntimeError(f"{what}: {e}") from e

        def drain():
            close()
            while inflight:
                wait_oldest()

        cur_set, cur_phase = None, None
        t_phase = t0 = time.perf_counter()
        for kset, name, bos, phase, o in self.ops:
            if phase != cur_phase:
                drain()
                now = time.perf_counter()
                if cur_phase is not None:
                    phases[cur_phase] = now - t_phase
                t_phase, cur_phase, cur_set = now, phase, None
            if name is None:
                drain()
                self.bufs["X"].write(self._conditioning_bits)
                cur_set = None
                continue
            ts = time.perf_counter()
            if not cur or kset != cur_set:
                close()
                cfg = __import__("pyxrt").run(npu.cfg[kset][flip])
                flip ^= 1
                cur.extend([npu.runlist(), [cfg], f"{kset}/{o['stream']}"])
                cur[0].add(cfg)
                cur_set = kset
            for r in self._elf_runs(kset, name, o) + [npu.run(name, *bos)]:
                cur[1].append(r)
                cur[0].add(r)
            if profile:
                drain()
                cur_set = None
                group = "steps" if phase.startswith("step") else phase
                import re
                key = f"{group}/{kset}/{re.sub(r'[0-9]+', '#', str(o['what']))}"
                e = prof.setdefault(key, {"ops": 0, "s": 0.0})
                e["ops"] += 1
                e["s"] += time.perf_counter() - ts
        drain()
        if profile:
            return dict(sorted(prof.items(), key=lambda kv: -kv[1]["s"]))
        now = time.perf_counter()
        phases[cur_phase] = now - t_phase
        phases["total"] = now - t0
        return phases

    def run(self) -> dict[str, float]:
        if self.projected and self._conditioning_bits is None:
            raise ValueError("projected conditioning has not been set")
        if getattr(self, "elf", False):
            return self._run_elf()
        from npu_host import COMPLETED

        def wait(h, what):
            st = h.wait()
            if st != COMPLETED:
                raise RuntimeError(f"{what}: {st}")

        # Every queued run's handle stays alive until it has completed: Linux XRT aborts
        # the process when an xrt::run is destroyed while its command is in flight (the
        # Windows runner overwrote all but the last handle).
        phases: dict[str, float] = {}
        cur_set, cur_phase = None, None
        inflight: list = []

        def drain(what):
            for h in inflight:
                wait(h, what)
            inflight.clear()

        t_phase = t0 = time.perf_counter()
        for kset, st, bos, phase, o in self.ops:
            if kset != cur_set or phase != cur_phase:
                drain(f"{cur_set} before {o['what']}")
                if phase != cur_phase:
                    now = time.perf_counter()
                    if cur_phase is not None:
                        phases[cur_phase] = now - t_phase
                    t_phase, cur_phase = now, phase
                cur_set = kset
            if st is None:
                drain("before projected conditioning upload")
                self.bufs["X"].write(self._conditioning_bits)
                continue
            if len(inflight) >= 64:                    # same context: in order, so the
                wait(inflight.pop(0), f"{kset}/{st.name}")   # oldest finishes first
            inflight.append(st.start(*bos))
        drain("last ops")
        now = time.perf_counter()
        phases[cur_phase] = now - t_phase
        phases["total"] = now - t0
        return phases

    def profile(self) -> dict[str, dict]:
        """One serialised run: seconds per (phase group, kernel set, op kind); for the split, not the time."""
        import re
        from npu_host import COMPLETED

        if self.projected:
            raise ValueError("profile does not cover projected-conditioning mode")
        if getattr(self, "elf", False):
            return self._run_elf(profile=True)
        out: dict[str, dict] = {}
        for kset, st, bos, phase, o in self.ops:
            t = time.perf_counter()
            state = st.start(*bos).wait()
            dt = time.perf_counter() - t
            if state != COMPLETED:
                raise RuntimeError(f"{kset}/{st.name}: {state}")
            group = "steps" if phase.startswith("step") else phase
            key = f"{group}/{kset}/{re.sub(r'[0-9]+', '#', str(o['what']))}"
            e = out.setdefault(key, {"ops": 0, "s": 0.0})
            e["ops"] += 1
            e["s"] += dt
        return dict(sorted(out.items(), key=lambda kv: -kv[1]["s"]))

    def image(self) -> np.ndarray:
        R = self.R
        return self.bufs["v_RGBA"].read(np.uint8).reshape(-1, 8192)[:, :4096] \
            .reshape(R, R, 4)[..., :3].copy()

    def latents(self) -> np.ndarray:
        T = kp.image_tokens(self.R)
        return self.bufs["LAT"].read(np.uint16, 0, T * kp.LAT_CH).reshape(T, kp.LAT_CH)


def load_reference(path: Path, R: int) -> np.ndarray:
    """uint8 [R, R, 3]: EXIF orientation, RGB, centre crop to a square, LANCZOS to R."""
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    s = min(img.size)
    left, top = (img.width - s) // 2, (img.height - s) // 2
    img = img.crop((left, top, left + s, top + s))
    if s != R:
        img = img.resize((R, R), Image.Resampling.LANCZOS)
    return np.asarray(img, np.uint8)


def pack_noise(noise: np.ndarray) -> np.ndarray:
    """[128, h, w] -> diffusers' _pack_latents order [h*w, 128], as bf16 bits."""
    c, h, w = noise.shape
    return np.ascontiguousarray(noise.reshape(c, h * w).T).astype(bfloat16).view(np.uint16)


def gguf_modulation(gguf: Path, model: Path, choice: str) -> bool:
    """The GGUF's adaLN order: as named, else compared with the checkpoint's own tensor
    (local transformer/, else the base repo by range read), else BFL with a warning."""
    if choice != "auto":
        return choice == "bfl"
    from klein_gguf import detect_bfl_modulation
    found = detect_bfl_modulation(gguf, transformer_dir=model / "transformer",
                                  repo="black-forest-labs/FLUX.2-klein-4B")
    if found is None:
        print("[WARN] no reference for the GGUF's modulation order; assuming BFL's (scale, shift)",
              flush=True)
        return True
    return found


def seed_noise(seed: int, tokens: int, edit: bool) -> np.ndarray:
    # An edit draws from its own stream: with the text-to-image noise of the same seed
    # (the server's default seed is shared), editing an image generated with that seed
    # reproduces the reference instead of applying the edit (measured 2026-11-08).
    rng = np.random.default_rng([seed, 1] if edit else seed)
    return rng.standard_normal((tokens, kp.LAT_CH), np.float32).astype(bfloat16).view(np.uint16)


def serve(a) -> int:
    """Resident worker for ``phlegm serve``: one JSON request per stdin line, one JSON reply per
    stdout line. The Runner is kept while (size, steps, edit) stays the same."""
    import gc
    import os
    from PIL import Image

    replies = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)  # progress output from the runner and pyxrt must not reach the reply stream
    sys.stdout = sys.stderr

    def send(value: dict) -> None:
        replies.write(json.dumps(value) + "\n")

    try:
        geometry = check_checkpoint(a.model, gguf=a.transformer_gguf)
        gguf = None
        if a.transformer_gguf:
            gguf = (str(a.transformer_gguf), gguf_modulation(a.transformer_gguf, a.model, a.gguf_modulation))
    except (Exception, SystemExit) as e:
        send({"ok": False, "error": f"startup: {e}"})
        return 1
    send({"ok": True, "protocol": 1})
    runner, key = None, None
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            q = json.loads(line)
            size, steps, seed = int(q["size"]), int(q["steps"]), int(q["seed"])
            ref = Path(q["image"]) if q.get("image") else None
            why = kp.check_resolution(size) or (kp.check_edit(size, size) if ref else None)
            if why:
                raise ValueError(f"unsupported size: {why}")
            if not 1 <= steps <= 50:
                raise ValueError("steps must be between 1 and 50")
            reused = key == (size, steps, ref is not None)
            if not reused:
                runner, key = None, None
                gc.collect()  # the old hardware context and buffers go before the new ones
                runner = Runner(a.model, a.kernels, size, steps, ref is not None, a.pack_jobs,
                                gguf=gguf, elf=a.dispatch == "elf")
                key = (size, steps, ref is not None)
            out = Path(q["out"])
            out.mkdir(parents=True, exist_ok=True)
            t = time.perf_counter()
            _, n_real = runner.set_prompt(q["prompt"])
            runner.set_noise(seed_noise(seed, kp.image_tokens(size), ref is not None))
            if ref is not None:
                runner.set_reference(load_reference(ref, size))
            host_inputs = time.perf_counter() - t
            phases = runner.run()
            image_path = out / "image.png"
            Image.fromarray(runner.image()).save(image_path)
            summary = {"image": str(image_path), "prompt": q["prompt"], "size": size, "steps": steps,
                       "seed": seed, "edit": ref is not None, "reference": str(ref) if ref else None,
                       "tokens": n_real, "model": str(a.model), "kernels": str(a.kernels),
                       "checkpoint": geometry, "dispatches": runner.dispatch_count,
                       "dispatch": a.dispatch, "resident": True, "reused_runner": reused,
                       "load_s": 0.0 if reused else round(runner.load_s, 2),
                       "npu_phases_s": {k: round(v, 4) for k, v in phases.items()},
                       "host_inputs_s": round(host_inputs, 4)}
            (out / "summary.json").write_text(json.dumps(summary, indent=1))
            send({"ok": True, "image": str(image_path), "reused_runner": reused,
                  "load_s": summary["load_s"], "npu_s": round(phases["total"], 4)})
        except (Exception, SystemExit) as e:
            if not isinstance(e, (ValueError, KeyError)):
                runner, key = None, None  # device state is unknown after a failed build or run
                gc.collect()
            send({"ok": False, "error": f"{type(e).__name__}: {e}"})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", type=Path, required=True, help="diffusers checkpoint directory")
    ap.add_argument("--kernels", type=Path, required=True)
    ap.add_argument("--serve", action="store_true",
                    help="resident worker: JSON requests on stdin, replies on stdout (phlegm serve)")
    ap.add_argument("--prompt")
    ap.add_argument("--projected-conditioning", type=Path,
                    help="experimental 512x3072 artifact; bypass native text encoding/projection")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=kp.STEPS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=Path, help="initial latents [128, R/16, R/16] (.npy)")
    ap.add_argument("--image", type=Path, help="reference image: a one-reference edit")
    ap.add_argument("--pack-jobs", type=int, default=4)
    ap.add_argument("--dispatch", choices=("elf", "sets"), default="elf",
                    help="elf: one hardware context from the resolution's full ELF (configure-only "
                         "kernels between sets); sets: one context per kernel set")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run the same request N times in one process; summary gives each and the median")
    ap.add_argument("--profile", action="store_true",
                    help="after the timed runs, one serialised run timed per op kind")
    ap.add_argument("--transformer-gguf", type=Path,
                    help="take the DiT from this GGUF; --model still supplies the text encoder, VAE, tokenizer")
    ap.add_argument("--gguf-modulation", choices=("auto", "bfl", "diffusers"), default="auto",
                    help="the GGUF's final adaLN half order (auto: compare with a reference)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.serve:
        if a.prompt or a.out or a.image or a.noise or a.projected_conditioning or a.repeat != 1 or a.profile:
            ap.error("--serve takes its requests on stdin; give only model, kernels and dispatch options")
        return serve(a)
    if not a.prompt or not a.out:
        ap.error("--prompt and --out are required")

    why = kp.check_resolution(a.size) or (kp.check_edit(a.size, a.size) if a.image else None)
    if why:
        raise SystemExit(f"unsupported size: {why}")
    if not 1 <= a.steps <= 50:
        raise SystemExit("steps must be between 1 and 50")
    geometry = check_checkpoint(a.model, projected=a.projected_conditioning is not None,
                                gguf=a.transformer_gguf)
    gguf = None
    if a.transformer_gguf:
        gguf = (str(a.transformer_gguf), gguf_modulation(a.transformer_gguf, a.model, a.gguf_modulation))
    conditioning = None
    if a.projected_conditioning:
        from klein_conditioning import load_projected_conditioning
        from tokenizers import Tokenizer
        conditioning = load_projected_conditioning(a.projected_conditioning)
        tokenizer = Tokenizer.from_file(str(a.model / "tokenizer" / "tokenizer.json"))
        conditioning.require_prompt(*kp.token_ids(tokenizer, a.prompt))
    a.out.mkdir(parents=True, exist_ok=True)
    T = kp.image_tokens(a.size)
    if a.noise:
        noise = np.load(a.noise).astype(np.float32)
        if noise.shape != (kp.LAT_CH, a.size // 16, a.size // 16):
            raise SystemExit(f"noise shape {noise.shape} does not fit {a.size}")
        lat = pack_noise(noise)
    else:
        lat = seed_noise(a.seed, T, a.image is not None)

    r = Runner(a.model, a.kernels, a.size, a.steps, a.image is not None, a.pack_jobs,
               projected=conditioning is not None, gguf=gguf, elf=a.dispatch == "elf")
    from PIL import Image
    ref_rgb = load_reference(a.image, a.size) if a.image else None
    runs, host = [], []
    for _ in range(max(1, a.repeat)):
        t = time.perf_counter()
        ids, n_real = (r.set_projected_conditioning(conditioning, a.prompt) if conditioning is not None
                       else r.set_prompt(a.prompt))
        t_prompt = time.perf_counter()
        r.set_noise(lat)
        if ref_rgb is not None:
            r.set_reference(ref_rgb)
        t_inputs = time.perf_counter()
        runs.append(r.run())
        host.append({"prompt": t_prompt - t, "noise_reference": t_inputs - t_prompt})
    phases = runs[-1]
    image_path = a.out / "image.png"
    t = time.perf_counter()
    rgb = r.image()
    t_read = time.perf_counter()
    Image.fromarray(rgb).save(image_path)
    host[-1] |= {"readback": t_read - t, "png": time.perf_counter() - t_read}
    np.save(a.out / "latents.npy", r.latents())
    median = {k: float(np.median([p[k] for p in runs])) for k in phases}
    summary = {"image": str(image_path), "prompt": a.prompt, "size": a.size,
               "steps": a.steps, "seed": a.seed, "noise": str(a.noise) if a.noise else None,
               "edit": a.image is not None, "reference": str(a.image) if a.image else None,
               "tokens": n_real, "model": str(a.model), "kernels": str(a.kernels),
               "checkpoint": geometry, "dispatches": r.dispatch_count, "dispatch": a.dispatch,
               "transformer_gguf": ({"path": gguf[0], "modulation": "bfl" if gguf[1] else "diffusers",
                                     "types": r.dit.gguf.types()} if gguf else None),
               "projected_conditioning": ({"path": str(a.projected_conditioning),
                                            **conditioning.provenance()} if conditioning else None),
               "load_s": round(r.load_s, 2),
               "npu_phases_s": {k: round(v, 4) for k, v in phases.items()},
               "host_s": [{k: round(v, 4) for k, v in h.items()} for h in host]}
    if len(runs) > 1:
        summary["repeat"] = {"n": len(runs), "median_s": {k: round(v, 4) for k, v in median.items()},
                             "runs_s": [{k: round(v, 4) for k, v in p.items()} for p in runs]}
    if a.profile:
        summary["profile"] = {k: {"ops": v["ops"], "s": round(v["s"], 4)} for k, v in r.profile().items()}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{phases['total']:.2f} s on the NPU (median of {len(runs)}: {median['total']:.2f} s) -> {image_path}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
