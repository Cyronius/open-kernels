"""Check a bidirectional BERT attention build with the Windows harness."""

import argparse
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "open_kernels"))
from export_bert_attention import HEADS, HEAD_WIDTH, LENGTH_MARKER, SEQUENCES, check_artifact, validate_geometry


def bf16_bits(values):
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype("<u2")


def from_bf16(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def reference(query, key, value, rows):
    scores = query[:, rows].astype(np.float64) @ key.astype(np.float64).swapaxes(1, 2)
    scores /= np.sqrt(query.shape[-1])
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ value.astype(np.float64)


# The compiled mlir-aie transaction uses WRITE32 operations for length registers.
# Word layout follows open_kernels/harness/stream_patch.hpp.
def patch_key_length(source, target, marker, length):
    stream = bytearray(source.read_bytes())
    _, _, operation_count, stream_size = struct.unpack_from("<4I", stream)
    if stream_size != len(stream):
        raise ValueError("length register patch needs a complete instruction transaction")
    operation_sizes = {0: 24, 1: 48, 3: 28, 0x80: 16, 0x81: 48}
    writes = {}
    expected_tiles = {(col, row) for col in range(8) for row in range(2, 6)}
    offset = 16
    for _ in range(operation_count):
        opcode, = struct.unpack_from("<I", stream, offset)
        if opcode == 0:
            _, _, low_address, high_address, value, size = struct.unpack_from("<6I", stream, offset)
            if size != operation_sizes[opcode]:
                raise ValueError("unexpected WRITE32 size")
            if value == marker:
                tile = (low_address >> 25, (low_address >> 20) & 31)
                if high_address or tile not in expected_tiles or tile in writes:
                    raise ValueError("invalid or duplicate compute-tile length register write")
                writes[tile] = offset + 16
        offset += operation_sizes[opcode]
    if offset != len(stream) or writes.keys() != expected_tiles:
        raise ValueError(f"expected one length register write per compute tile, found {len(writes)}")
    for value_offset in writes.values():
        struct.pack_into("<I", stream, value_offset, length)
    target.write_bytes(stream)


def synthetic_fixture(shape, valid_length):
    generator = np.random.default_rng(42)
    query, key, value = [generator.normal(0, 0.5, shape).astype(np.float32) for _ in range(3)]
    # The first query must attend to the last valid key in every head.
    query[:, 0] = 0
    query[:, 0, 0] = 7
    key[:, valid_length - 1, 0] = 16
    value[:, valid_length - 1, 0] = 8
    if valid_length < shape[1]:
        # This key would dominate the first query if padding were visible.
        key[:, valid_length, 0] = 24
        value[:, valid_length, 0] = -8
    return query, key, value


def captured_fixture(path, shape, valid_length):
    with np.load(path, allow_pickle=False) as data:
        tensors = tuple(data[name] for name in ("query", "key", "value"))
    if any(tensor.shape != (shape[0], valid_length, shape[2]) or tensor.dtype != np.float32
           or not np.isfinite(tensor).all()
           for tensor in tensors):
        raise ValueError("Q/K/V captures must be finite float32 arrays shaped [heads, valid_length, width]")
    padding = ((0, 0), (0, shape[1] - valid_length), (0, 0))
    return tuple(np.pad(tensor, padding) for tensor in tensors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", required=True, type=Path)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--sequence", required=True, type=int, choices=SEQUENCES)
    parser.add_argument("--heads", type=int, default=HEADS)
    parser.add_argument("--head-width", type=int, default=HEAD_WIDTH)
    parser.add_argument("--valid-length", type=int, help="key length to test")
    parser.add_argument("--rtp-marker", type=int, help="compiled register value for a runtime-length build")
    parser.add_argument("--fixture", type=Path, help="use captured float32 Q/K/V in an NPZ file")
    args = parser.parse_args()
    valid_length = args.sequence if args.valid_length is None else args.valid_length
    if not 2 <= valid_length <= args.sequence:
        parser.error("--valid-length must be at least two and within the sequence")

    work = args.work.resolve()
    sequence = args.sequence
    validate_geometry(sequence, args.heads, args.head_width)
    build = args.build.resolve()
    rtp_marker = args.rtp_marker
    if (build / "attention.json").is_file():
        check_artifact(build, sequence, args.heads, args.head_width)
        if rtp_marker not in (None, LENGTH_MARKER):
            parser.error("--rtp-marker disagrees with the exported artifact")
        rtp_marker = LENGTH_MARKER
    work.mkdir(parents=True, exist_ok=True)
    shape = (args.heads, sequence, args.head_width)
    original = (captured_fixture(args.fixture, shape, valid_length) if args.fixture
                else synthetic_fixture(shape, valid_length))
    tensors = []
    for name, values in zip(("query", "key", "value"), original, strict=True):
        bits = bf16_bits(values)
        bits.tofile(work / f"{name}.bin")
        tensors.append(from_bf16(bits))

    size_bytes = int(np.prod(shape)) * 2
    instructions = build / "insts.bin"
    if rtp_marker is not None:
        patched = work / "insts-length.bin"
        patch_key_length(instructions, patched, rtp_marker, valid_length)
        instructions = patched
    config = work / "attention.cfg"
    config.write_text(
        f"device\nxclbin G {build / 'final.xclbin'}\nkernelx k G {instructions}\n"
        f"buf q {size_bytes} query.bin\nbuf kb {size_bytes} key.bin\nbuf v {size_bytes} value.bin\n"
        f"buf o {size_bytes}\nrun k q kb v o\ndump o output-first.bin {size_bytes}\n"
        f"run k q kb v o\ndump o output.bin {size_bytes}\n",
        encoding="ascii",
    )
    subprocess.run([str(args.harness.resolve()), config.name], cwd=work, check=True, timeout=400)
    first = np.fromfile(work / "output-first.bin", dtype="<u2")
    second = np.fromfile(work / "output.bin", dtype="<u2")
    repeatable = np.array_equal(first, second)
    output = from_bf16(second).reshape(shape)

    # Sample both sides of each query-tile boundary and the last query.
    tile_boundary_rows = {row for edge in range(256, valid_length, 256) for row in (edge - 1, edge)}
    rows = sorted(tile_boundary_rows | {row for row in (0, 1, 63, 64, valid_length - 1) if row < valid_length})
    query, key, value = tensors
    expected = reference(query, key[:, :valid_length], value[:, :valid_length], rows)
    actual = output[:, rows].astype(np.float64)
    finite = bool(np.isfinite(output).all())
    if args.fixture:
        original_query, original_key, original_value = original
        full_precision = reference(original_query, original_key[:, :valid_length],
                                   original_value[:, :valid_length], rows)
        rounding_relative_l2 = np.linalg.norm(expected - full_precision) / np.linalg.norm(full_precision)
        fixture_stats = f"bf16_input_relative_l2={rounding_relative_l2:.6g}"
        fixture_pass = True
    else:
        tail_error = float(np.abs(actual[:, 0, 0] - expected[:, 0, 0]).max())
        fixture_stats = f"last_key_error={tail_error:.6g} last_key_min_output={output[:, 0, 0].min():.6g}"
        fixture_pass = tail_error < 0.1
        actual, expected = actual[:, 1:], expected[:, 1:]
    relative_l2 = np.linalg.norm(actual - expected, axis=(1, 2)) / np.linalg.norm(expected, axis=(1, 2))
    cosine = (actual * expected).sum(axis=2) / (np.linalg.norm(actual, axis=2) * np.linalg.norm(expected, axis=2))
    print(f"sequence={sequence} valid_keys={valid_length} heads={args.heads} "
          f"head_width={args.head_width} sampled_rows={len(rows)} "
          f"finite={finite} repeatable={repeatable} "
          f"worst_head_relative_l2={relative_l2.max():.6g} min_row_cosine={cosine.min():.8f} "
          f"{fixture_stats}")
    passed = finite and repeatable and relative_l2.max() < 0.03 and cosine.min() > 0.995 and fixture_pass
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
