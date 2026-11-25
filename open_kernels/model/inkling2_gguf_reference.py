"""FP32 text reference for the fixed-loop Inkling 2 Mini GGUF.

Compares prefill logits, per-visit residuals, and greedy decode with the published model.
See specs/open-engine/plans/inkling2-mini-production-bar.md for the pinned
llama.cpp graph and GGUF revisions.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gguf_pool import dequant_iq4_nl, dequant_q8_0, read_gguf  # noqa: E402
from recipes.gptoss_pack import bf16_bits, bf16_value  # noqa: E402


class Weights:
    def __init__(self, path: Path):
        self.metadata, self.tensors, self.base = read_gguf(str(path))
        self.data = np.memmap(path, dtype=np.uint8, mode="r")

    def __call__(self, name, selection=slice(None)):
        info = self.tensors[name]
        shape = tuple(reversed(info["dims"]))
        offset = self.base + info["offset"]
        kind = info["type"]
        if kind in ("F32", "F16", "BF16"):
            dtype = {"F32": "<f4", "F16": "<f2", "BF16": "<u2"}[kind]
            x = np.ndarray(shape, dtype, self.data, offset)[selection]
            if kind == "BF16":
                return (x.astype(np.uint32) << 16).view(np.float32)
            return x.astype(np.float32, copy=False)
        decode, block_bytes = {"IQ4_NL": (dequant_iq4_nl, 18),
                               "Q8_0": (dequant_q8_0, 34)}[kind]
        cols = shape[-1]
        blocks = np.ndarray((*shape[:-1], cols // 32, block_bytes), np.uint8,
                            self.data, offset)[selection]
        return decode(blocks.reshape(-1, cols // 32, block_bytes)).reshape(*blocks.shape[:-2], cols)


def rms(x, eps=1e-6):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def sigmoid(x):
    return np.exp(-np.logaddexp(np.float32(0), -x))


def softcap(x, cap):
    return cap * np.tanh(x / cap)


def clip(x, threshold):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x * np.minimum(1.0, threshold / (norm + 1e-6))


def mix(blocks, current, query, eps):
    states = rms(np.stack([*blocks, current], axis=1), eps)
    scores = np.atleast_2d(query) @ states.transpose(0, 2, 1) / np.sqrt(np.float32(states.shape[-1]))
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return (probs / probs.sum(axis=-1, keepdims=True)) @ states


def convolve(x, weight, dynamic=None, history=None):
    """Depthwise causal taps. Dynamic taps are per head, with tap 0 at lag 0."""
    taps = weight.shape[-1]
    if history is None:
        history = np.zeros((taps - 1, x.shape[-1]), np.float32)
    padded = np.concatenate((history, x), axis=0)
    result = x.copy()
    for lag in range(taps):
        start = taps - 1 - lag
        values = padded[start:start + len(x)]
        result += values * weight[:, start]
        if dynamic is not None:
            result += (values.reshape(len(x), dynamic.shape[1], -1)
                       * dynamic[:, :, lag, None]).reshape(x.shape)
    banks = 1 if dynamic is None else 2
    return result / np.sqrt(np.float32(banks * taps + 1)), padded[len(x):].copy()


def attention(q, k, v, relative, table, sinks, window):
    """Causal grouped-query attention with a query-dependent zero-value sink."""
    positions, heads, width = q.shape
    kv_heads = k.shape[1]
    groups = heads // kv_heads
    relative_scores = relative @ table.T
    extent = table.shape[0]
    out = np.empty_like(q)
    prefix = len(k) - positions
    for row, t in enumerate(range(prefix, len(k))):
        start = max(0, t - window + 1) if window else 0
        keys, values = k[start:t + 1], v[start:t + 1]
        scores = np.einsum("hgd,shd->hgs", q[row].reshape(kv_heads, groups, width), keys)
        scores /= np.sqrt(np.float32(width))
        count = min(t + 1 - start, extent)
        scores = scores.reshape(heads, -1)
        scores[:, -count:] += relative_scores[row, :, :count][:, ::-1]
        peak = np.maximum(sinks[row], scores.max(axis=-1))
        probs = np.exp(scores - peak[:, None])
        probs /= (probs.sum(axis=-1) + np.exp(sinks[row] - peak))[:, None]
        out[row] = np.einsum("hgs,shd->hgd", probs.reshape(kv_heads, groups, -1), values).reshape(heads, width)
    return out


@dataclass(frozen=True)
class VisitState:
    k: np.ndarray
    v: np.ndarray
    conv: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class TextState:
    position: int = 0
    visits: tuple[VisitState, ...] = ()


class Inkling2:
    def __init__(self, weights: Weights, *, bf16_kv: bool = False):
        self.w = weights
        self.bf16_kv = bf16_kv
        md = weights.metadata
        arch = "inkling2_loop"
        if md["general.architecture"] != arch or md.get(f"{arch}.loop.varying", False):
            raise ValueError("expected the fixed-loop inkling2_loop GGUF layout")

        def field(name):
            return md[f"{arch}.{name}"]

        self.layers = field("block_count")
        self.start, self.end, self.passes = (field(f"loop.{key}") for key in ("start", "end", "count"))
        self.block = field("attn_res.block_size")
        self.sliding = field("attention.sliding_window_pattern")
        self.window = field("attention.sliding_window")
        self.taps = field("shortconv.l_cache")
        self.head_dim = field("attention.key_length")
        self.eps = field("attention.layer_norm_rms_epsilon")
        self.qk_clip = field("attention.qk_clip")
        self.rel_clip = field("attention.relative_clip")
        self.floor = field("attention.log_scale_floor")
        self.alpha = field("attention.log_scale_alpha")
        self.top_k = field("expert_used_count")
        self.route_scale = field("expert_weights_scale")
        self.gate_cap = field("activation.situ_beta")
        self.up_cap = field("activation.situ_linear_beta")
        self.vocab = field("vocab_size_unpadded")
        self.logit_cap = field("final_logit_softcapping")
        self.context = field("context_length")

    def control(self, x, message, weight, name):
        gate = np.concatenate([rms(x, self.eps), rms(message, self.eps)], axis=-1)
        gate = gate @ weight(f"loop_{name}_ctrl_down").T
        gain = np.tanh(gate @ weight(f"loop_{name}_ctrl_gain").T)
        forget = np.tanh(gate @ weight(f"loop_{name}_ctrl_forget").T)
        return x - forget * x + gain * message

    def moe(self, x, weight, prefix):
        scores = sigmoid(x @ weight("ffn_gate_inp").T)
        bias_name = prefix + "exp_probs_b.bias"
        selection = scores + self.w(bias_name) if bias_name in self.w.tensors else scores
        ids = np.argsort(-selection, axis=-1, kind="stable")[:, :self.top_k]
        routes = np.take_along_axis(scores, ids, axis=-1)
        routes *= self.route_scale / np.maximum(routes.sum(axis=-1, keepdims=True), 1e-20)
        out = np.zeros_like(x)
        for e in np.unique(ids):
            tokens, slots = np.nonzero(ids == e)
            both = self.w(prefix + "ffn_gate_up_exps.weight", e)
            gate, up = np.split(x[tokens] @ both.T, 2, axis=-1)
            hidden = softcap(gate, self.gate_cap) * sigmoid(gate) * softcap(up, self.up_cap)
            down = self.w(prefix + "ffn_down_exps.weight", e)
            out[tokens] += routes[tokens, slots, None] * (hidden @ down.T)
        return out

    def layer(self, i, x, blocks, previous, position, state):
        prefix = f"blk.{i}."

        def weight(name):
            return self.w(prefix + name + ".weight")

        histories = iter(state.conv if state else (None,) * 4)
        next_histories = []

        def conv(x, name, dynamic=None):
            output, tail = convolve(x, weight(name), dynamic, next(histories))
            next_histories.append(tail)
            return output

        sliding = self.sliding[i]
        inputs = mix(blocks, x, weight("attn_res_query"), self.eps) if i else x[:, None, :]
        messages = None
        if previous is not None:
            messages = mix(previous[:-1], previous[-1], weight("loop_msg_query"), self.eps)

        projected = []
        for index, name in enumerate(("q", "k", "v")):
            u = inputs[:, 0 if sliding else index]
            if messages is not None:
                u = self.control(u, messages[:, index], weight, name)
            u = rms(u, self.eps) * weight("attn_norm" if sliding else f"attn_{name}_inp_norm")
            h = u @ weight(f"attn_{name}").T
            if index == 0:
                q_input = u
            else:
                dynamic = None if sliding else (u @ weight(f"attn_{name}_conv_proj").T).reshape(
                    len(x), -1, self.taps)
                h = conv(h, f"attn_{name}_conv", dynamic)
            projected.append(h.reshape(len(x), -1, self.head_dim))
        q, k, v = projected
        q = clip(q * weight("attn_q_clip"), self.qk_clip)
        k = clip(k, self.qk_clip)
        v = rms(v, self.eps)
        if self.bf16_kv:
            k, v = (bf16_value(bf16_bits(x)) for x in (k, v))
        table = weight("attn_rel_bias")
        relative = (q_input @ weight("attn_rel").T).reshape(len(x), q.shape[1], table.shape[-1])
        relative = clip(relative * weight("attn_rel_clip"), self.rel_clip)
        if not sliding:
            positions = np.arange(position + 1, position + len(x) + 1, dtype=np.float32)
            tau = (1 + self.alpha * np.log(np.maximum(1, positions / self.floor)))[:, None, None]
            q *= tau
            relative *= tau
        sinks = q_input @ weight("attn_sink_proj").T
        if state is not None:
            k = np.concatenate((state.k, k), axis=0)
            v = np.concatenate((state.v, v), axis=0)
        a = attention(q, k, v, relative, table, sinks, self.window if sliding else 0)
        a = a.reshape(len(x), -1) * sigmoid(q_input @ weight("attn_gate").T)
        a = conv(a @ weight("attn_output").T, "attn_post_conv")
        if i % self.block == 0:
            blocks.append(x)
        else:
            a += x
        f = mix(blocks, a, weight("ffn_res_query"), self.eps)[:, 0]
        if messages is not None:
            f = self.control(f, messages[:, 3], weight, "ffn")
        f = self.moe(rms(f, self.eps) * weight("ffn_norm"), weight, prefix)
        out = a + conv(f, "ffn_post_conv")
        if sliding and len(k) > self.window:
            k, v = k[-self.window:].copy(), v[-self.window:].copy()
        return out, VisitState(k, v, tuple(next_histories))

    def forward(self, ids, state=None, observe=None):
        """Evaluate a chunk. Pass an empty TextState to retain per-visit caches."""
        position = state.position if state else 0
        if not len(ids) or position + len(ids) > self.context or np.any((ids < 0) | (ids >= self.vocab)):
            raise ValueError("input token IDs or context length outside the model range")
        states = []
        visit = 0

        def advance(physical, x, blocks, previous=None):
            nonlocal visit
            history = state.visits[visit] if state and state.visits else None
            x, next_state = self.layer(physical, x, blocks, previous, position, history)
            if observe is not None:
                observe(visit, physical, x)
            if state is not None:
                states.append(next_state)
            visit += 1
            return x

        x = rms(self.w("token_embd.weight", ids), self.eps) * self.w("token_embd_norm.weight")
        blocks = []
        for i in range(self.start):
            x = advance(i, x, blocks)
        prelude, prelude_blocks = x, blocks.copy()
        previous = None
        for _ in range(self.passes):
            x, blocks = prelude, prelude_blocks.copy()
            for i in range(self.start, self.end):
                x = advance(i, x, blocks, previous)
            previous = [*blocks[len(prelude_blocks) + 1:], x]
        for i in range(self.end, self.layers):
            x = advance(i, x, blocks)
        x = mix(blocks, x, self.w("output_res_query.weight"), self.eps)[:, 0]
        next_text = TextState(position + len(ids), tuple(states)) if state is not None else None
        return rms(x, self.eps) * self.w("output_norm.weight"), next_text

    def logits(self, states):
        out = np.empty((len(states), self.vocab), dtype=np.float32)
        for start in range(0, self.vocab, 4096):
            block = slice(start, min(start + 4096, self.vocab))
            out[:, block] = states @ self.w("output.weight", block).T
        return softcap(out, self.logit_cap)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path, required=True, help="whitespace-separated token IDs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dump-visits", action="store_true", help="save execution-order residuals")
    parser.add_argument("--bf16-kv", action="store_true", help="round K/V as the native host cache does")
    parser.add_argument("--decode", type=int, default=0, help="number of greedy tokens to generate")
    args = parser.parse_args()
    if args.decode < 0:
        parser.error("--decode must be nonnegative")
    ids = np.array([int(token) for token in args.ids_file.read_text().split()], dtype=np.int64)
    args.out.mkdir(parents=True, exist_ok=True)

    def observe(visit, physical, residual):
        residual.tofile(args.out / f"visit{visit:02d}_layer{physical:02d}.f32")

    model = Inkling2(Weights(args.gguf), bf16_kv=args.bf16_kv)
    state = TextState() if args.decode > 1 else None
    states, state = model.forward(ids, state, observe if args.dump_visits else None)
    logits = model.logits(states[-1:])
    logits.tofile(args.out / "logits.f32")
    if args.decode:
        with (args.out / "greedy.txt").open("w") as output:
            for step in range(args.decode):
                token = int(logits.argmax())
                print(token, file=output, flush=True)
                if step + 1 < args.decode:
                    states, state = model.forward(np.array([token]), state)
                    logits = model.logits(states)


if __name__ == "__main__":
    main()
