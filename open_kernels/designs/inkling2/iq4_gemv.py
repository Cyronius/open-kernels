"""Stream one compact IQ4_NL projection on the NPU."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from ml_dtypes import bfloat16
import numpy as np

HERE = Path(__file__).resolve().parent
GEMV = HERE.parent / "gemv_q4"
sys.path.insert(0, str(HERE.parents[1]))
from gguf_pool import IQ4_NL_VALUES  # noqa: E402
from recipes.gptoss_pack import CH, MX_CH  # noqa: E402


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def iq4_gemv(w: In, x: In, y: Out, *,
              n: CompileTime[int], k: CompileTime[int], cores: CompileTime[int],
              source_hash: CompileTime[int] = 0):
    if n <= 0 or k <= 0 or cores <= 0 or n % (64 * cores) or k % 256 or k > 2304:
        raise ValueError("IQ4 GEMV needs evenly divided 64-row bands and K in 256-wide tiles up to 2304")
    bands = n // 64
    per_core = bands // cores
    groups = k // 256
    group_bytes, band_bytes = 4 * MX_CH, 4 * MX_CH * groups
    weights_type = np.ndarray[(bands * band_bytes,), np.dtype[np.uint8]]
    group_type = np.ndarray[(group_bytes,), np.dtype[np.uint8]]
    activation_type = np.ndarray[(k,), np.dtype[bfloat16]]
    quantized_type = np.ndarray[(9 * k // 4,), np.dtype[np.uint8]]
    lut_type = np.ndarray[(32,), np.dtype[np.int16]]
    expanded_type = np.ndarray[(CH,), np.dtype[np.uint8]]
    band_type = np.ndarray[(64,), np.dtype[np.float32]]
    output_type = np.ndarray[(n,), np.dtype[np.float32]]
    root = Path(config.cxx_header_path())
    includes = [str(root), str(root / "aie_kernels"), str(root / "aie_kernels" / "aie2p"), str(GEMV)]
    prep = ExternalFunction("inkling_iq4_prep", source_file=str(HERE / "iq4_gemv_prep.cc"),
                            arg_types=[activation_type, quantized_type, np.int32], include_dirs=includes)
    gemv = ExternalFunction("inkling_iq4_group", source_file=str(HERE / "iq4_gemv_group.cc"),
                            arg_types=[group_type, quantized_type, lut_type, lut_type, expanded_type, band_type,
                                       np.int32, np.int32], include_dirs=includes, compile_flags=["-Os"])
    weights = [ObjectFifo(group_type, depth=2, name=f"weights{c}") for c in range(cores)]
    bands_out = [ObjectFifo(band_type, depth=2, name=f"out{c}") for c in range(cores)]
    activation = ObjectFifo(activation_type, depth=1, name="activation")
    # The AIE2P LUT interleaves copies at 32-byte granularity. The two
    # complete buffers must land in separate core memory banks.
    table = np.tile(IQ4_NL_VALUES, 2).astype(np.int16)

    def core_body(rx, xin, tx, tab, lut0, lut1, expanded, setup, project):
        a = xin.acquire(1)
        setup(a, tab, k)
        for _ in range_(per_core):
            out = tx.acquire(1)
            for g in range_(groups):
                part = rx.acquire(1)
                project(part, tab, lut0, lut1, expanded, out, g, k)
                rx.release(1)
            tx.release(1)
        xin.release(1)

    workers = [
        Worker(core_body, stack_size=0x1000, fn_args=[
            weights[c].cons(), activation.cons(), bands_out[c].prod(), Buffer(quantized_type),
            Buffer(lut_type, initial_value=table, address=0xA000),
            Buffer(lut_type, initial_value=table, address=0xC000), Buffer(expanded_type), prep, gemv,
        ]) for c in range(cores)
    ]
    weight_taps = TensorTiler2D.simple_tiler((1, bands * band_bytes), (1, per_core * band_bytes))
    output_taps = [
        TensorAccessPattern((1, n), c * per_core * 64, [1, 1, 1, per_core * 64], [0, 0, 0, 1])
        for c in range(cores)
    ]

    def sequence(w_in, x_in, out, w_prod, x_prod, y_cons):
        tasks = TaskGroup()
        x_prod.fill(x_in, group=tasks)
        for c in range(cores):
            w_prod[c].fill(w_in, tap=weight_taps[c], group=tasks)
            y_cons[c].drain(out, tap=output_taps[c], wait=True, group=tasks)
        tasks.finish()

    runtime = Runtime(sequence, [weights_type, activation_type, output_type,
                                 [f.prod() for f in weights], activation.prod(), [f.cons() for f in bands_out]])
    return Program(iron.get_current_device(), runtime, workers=workers).resolve_program()


DESIGN = iq4_gemv
SPECIALIZE = {
    "n": int(os.environ.get("INKLING2_GEMV_N", 2304)),
    "k": int(os.environ.get("INKLING2_GEMV_K", 2304)),
    "cores": int(os.environ.get("INKLING2_GEMV_CORES", 4)),
    "source_hash": int(hashlib.sha256(IQ4_NL_VALUES.tobytes() + b"".join(p.read_bytes() for p in (
        HERE / "iq4_gemv_prep.cc", HERE / "iq4_gemv_group.cc", HERE / "iq4_lut.h", GEMV / "gemv_q8.h", GEMV / "gemv_tab.h",
    ))).hexdigest()[:8], 16),
}
