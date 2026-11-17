"""KVJOIN: decode layer cores past the narrow-kv cap (recipes/dense.kv_join_cores).

MiniCPM5-2B's k / v are 256 rows each, so cores_for caps it at 4 while every other width
divides by 8. Banded as one [k | v] run of 512 rows it splits one band per core over 8.
The pool and the activation layout stay as they were, so only dx's geometry moves."""
from pathlib import Path

from recipes.cache import build_key
from recipes.load import load_spec
from recipes.manifest import manifest
from recipes import dense

SPECS = Path(__file__).resolve().parents[3] / 'open_kernels/recipes/specs'
FLAGS = ('DENSE_GGUF_GEMM', 'DENSE_GGUF_HEAD_Q8', 'DENSE_GGUF_COMPACT', 'DENSE_GGUF_BFP16')


def test_kvjoin_lifts_minicpm5_to_eight_cores(monkeypatch):
    for flag in FLAGS:
        monkeypatch.setenv(flag, '1')
    monkeypatch.setenv('DENSE_KVJOIN', '0')
    spec = load_spec(SPECS / 'minicpm5-2b.json')
    g0, m0, k0 = dense.geometry(spec), manifest(spec), build_key(spec)
    assert (g0.N_CORES, g0.KV_PC, g0.KVJOIN) == (4, 1, 0)
    monkeypatch.delenv('DENSE_KVJOIN')               # measured: on by default for this shape
    g1, m1 = dense.geometry(spec), manifest(spec)
    assert (g1.N_CORES, g1.Q_PC, g1.KV_PC, g1.O_PC, g1.UP_PC, g1.DOWN_PC, g1.KVJOIN) == (8, 4, 0, 4, 12, 4, 1)
    # the same q | k | v rows per layer, one [k | v] band per core
    assert g1.Q_PC * 8 + 8 == g0.Q_PC * 4 + 2 * g0.KV_PC * 4
    assert build_key(spec) != k0
    # the pool, the head and the engine-facing layout do not move
    lay = dense.layout(spec)
    assert lay.POOL_V == lay.POOL_K + spec.attn_kv_width // dense.BAND_ROWS * dense.band_bytes(spec.hidden, dense.pool_quant(spec))
    for view in (m0, m0.get('gguf', {})):
        other = m1 if view is m0 else m1['gguf']
        for field in ('layout', 'pack', 'tail', 'manifest_version'):
            assert other.get(field) == view.get(field)


def test_kvjoin_leaves_eight_core_specs_alone(monkeypatch):
    for name in ('llama32-1b.json', 'qwen3-1.7b.json'):
        spec = load_spec(SPECS / name)
        monkeypatch.delenv('DENSE_KVJOIN', raising=False)
        g0 = dense.geometry(spec)
        monkeypatch.setenv('DENSE_KVJOIN', '1')
        assert dense.kv_join_cores(spec) == 0
        assert dense.geometry(spec) == g0


def test_minicpm5_streams_through_the_memtile_ring(monkeypatch):
    monkeypatch.delenv('DX_WRING', raising=False)
    monkeypatch.delenv('DX_WORDER', raising=False)
    assert dense.stream_knobs(load_spec(SPECS / 'minicpm5-2b.json')) == (dense.WRING_DEPTH, True)
    assert dense.stream_knobs(load_spec(SPECS / 'llama32-1b.json')) == (0, False)


def test_minicpm5_head_on_eight_cores(monkeypatch):
    for flag in FLAGS:
        monkeypatch.setenv(flag, '1')
    spec = load_spec(SPECS / 'minicpm5-2b.json')
    head = dense.builds(spec)['lm_head_q8_gguf']
    assert head['env']['LMHEAD_CORES'] == '8' and head['build_dir'].endswith('_c8')
    assert dense.head_cores(load_spec(SPECS / 'llama32-1b.json')) == dense.cores_for(load_spec(SPECS / 'llama32-1b.json'))


def test_minicpm5_attention_takes_the_fused_exponential():
    from recipes import attnknobs
    assert attnknobs.g5_expa(load_spec(SPECS / 'minicpm5-2b.json')) == 2
    assert attnknobs.g5_expa(load_spec(SPECS / 'llama32-1b.json')) == 0
