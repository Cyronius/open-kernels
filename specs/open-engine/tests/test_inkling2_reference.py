"""Inkling 2 Mini host and reference checks."""

import os
import subprocess

import numpy as np
import pytest

from gguf_pool import dequant_iq4_nl
from model.inkling2_gguf_reference import attention, convolve


def test_iq4_rows_decode_both_halves_and_scale_each_block():
    blocks = np.zeros((2, 2, 18), dtype=np.uint8)
    blocks[..., :2] = np.array([[0.5, -2.0], [1.0, 4.0]], dtype="<f2").view(np.uint8).reshape(2, 2, 2)
    blocks[..., 2:] = 0xF0
    values = np.array([-127] * 16 + [113] * 16, dtype=np.float32)
    expected = (np.array([[0.5, -2.0], [1.0, 4.0]])[..., None] * values).reshape(2, 64)
    np.testing.assert_array_equal(dequant_iq4_nl(blocks), expected)


def test_relative_extent_and_sliding_window_are_independent():
    q = np.zeros((3, 2, 1), np.float32)
    k = np.zeros((3, 1, 1), np.float32)
    v = np.array([1, 3, 9], np.float32).reshape(3, 1, 1)
    relative = np.ones_like(q)
    table = np.log(np.array([[2.0], [3.0]], np.float32))
    sinks = np.zeros((3, 2), np.float32)
    full = attention(q, k, v, relative, table, sinks, window=0)
    local = attention(q, k, v, relative, table, sinks, window=2)
    np.testing.assert_allclose(full[..., 0], [[2 / 3] * 2, [1.5] * 2, [4.0] * 2])
    np.testing.assert_allclose(local[..., 0], [[2 / 3] * 2, [1.5] * 2, [4.5] * 2])
    cached = attention(q[-1:], k, v, relative[-1:], table, sinks[-1:], window=2)
    np.testing.assert_allclose(cached, local[-1:])


def test_short_convolution_dynamic_taps_are_current_first_and_per_head():
    x = np.array([[1, 2], [3, 4]], np.float32)
    static = np.array([[1, 2], [3, 4]], np.float32)
    dynamic = np.array([[[5, 6], [7, 8]], [[10, 20], [30, 40]]], np.float32)
    fixed, _ = convolve(x, static)
    varying, expected_history = convolve(x, static, dynamic)
    np.testing.assert_allclose(fixed, np.array([[3, 10], [10, 26]]) / np.sqrt(3))
    np.testing.assert_allclose(varying, np.array([[8, 24], [60, 226]]) / np.sqrt(5))
    first, history = convolve(x[:1], static, dynamic[:1])
    second, history = convolve(x[1:], static, dynamic[1:], history)
    np.testing.assert_array_equal(np.concatenate((first, second)), varying)
    np.testing.assert_array_equal(history, expected_history)


def test_native_inkling2_block_and_chunked_prefill(tmp_path):
    exe = os.environ.get("INKLING2_HOST_CHECK")
    model = os.environ.get("INKLING2_IQ4_GGUF")
    if not exe or not model:
        pytest.skip("set INKLING2_HOST_CHECK and INKLING2_IQ4_GGUF to the native driver and pinned IQ4_NL model")
    ids = tmp_path / "five.ids"
    ids.write_text("371 151451 149879 151194 149884\n")
    env = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "8")}
    logits = []
    for chunk in (5, 2):
        out = tmp_path / f"chunk-{chunk}.f32"
        subprocess.run([exe, model, str(ids), str(out), "0", str(chunk)], env=env, check=True)
        logits.append(np.fromfile(out, dtype="<f4"))
    assert logits[0].size == 200058
    assert np.isfinite(logits[0]).all()
    np.testing.assert_array_equal(logits[0], logits[1])
    assert np.argsort(logits[0])[-5:][::-1].tolist() == [152717, 167815, 149881, 149876, 145038]
