"""Measure exact-length OFLM embedding requests against a local CPU reference.

The tokenizer counts include special tokens and any task prefix. For the CPU
reference, use a local Transformers checkpoint with a pooling configuration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from tokenizers import Tokenizer


def fixtures(tokenizer, lengths, prefix):
    tokenizer.no_truncation()
    tokenizer.no_padding()
    for length in lengths:
        for tail in ("amber", "ocean"):
            ending = f" The final subject is {tail}."
            base = prefix + ending
            count = len(tokenizer.encode(base).ids)
            if count > length:
                raise ValueError(f"{length} tokens cannot fit the prefix and fixture")
            text = "cat " * (length - count) + ending.lstrip()
            if len(tokenizer.encode(prefix + text).ids) != length:
                raise ValueError(f"cannot construct a {length}-token fixture with this tokenizer")
            yield length, text


def request_embedding(url, model, prompt_name, text):
    body = {"model": model, "input": text}
    if prompt_name is not None:
        body["prompt_name"] = prompt_name
    request = Request(url + "/v1/embeddings", json.dumps(body).encode(),
                      {"Content-Type": "application/json"})
    with urlopen(request, timeout=300) as response:
        return json.load(response)


def embed_server(url, model, prompt_name, text, length):
    result = request_embedding(url, model, prompt_name, text)
    if result.get("model") != model:
        raise ValueError(f"server returned model {result.get('model')!r}, expected {model!r}")
    count = result["usage"]["prompt_tokens"]
    if count != length:
        raise ValueError(f"server processed {count} tokens, expected {length}")
    return result["data"][0]["embedding"]


def cpu_reference(directory, prefix, max_length):
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    config = AutoConfig.from_pretrained(directory, local_files_only=True)
    options = {}
    if config.model_type == "nomic_bert" and max_length > 2048:
        options["rope_parameters"] = {"rope_theta": 1000.0, "rope_type": "dynamic", "factor": 2.0}
    model = AutoModel.from_pretrained(
        directory, local_files_only=True, dtype=torch.float32, **options
    ).eval()
    pooling = json.loads((directory / "1_Pooling" / "config.json").read_text())
    modes = [mode for mode, active in pooling.items()
             if mode.startswith("pooling_mode_") and active is True]
    if modes not in (["pooling_mode_cls_token"], ["pooling_mode_mean_tokens"]):
        raise ValueError(f"unsupported checkpoint pooling modes: {modes}")
    use_cls = modes == ["pooling_mode_cls_token"]

    @torch.inference_mode()
    def embed(text, length):
        inputs = tokenizer(prefix + text, return_tensors="pt", truncation=False)
        if inputs.input_ids.shape[-1] != length:
            raise ValueError("reference tokenizer disagrees with the fixture length")
        hidden = model(**inputs).last_hidden_state
        pooled = hidden[:, 0] if use_cls else hidden.mean(dim=1)
        if config.model_type == "nomic_bert":
            pooled = torch.nn.functional.layer_norm(pooled, (pooled.shape[-1],))
        return torch.nn.functional.normalize(pooled, dim=1)[0].numpy()

    return embed


def measure(embed, text, length, repeats):
    vector = np.asarray(embed(text, length), dtype=np.float64)
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        raise ValueError("embedding must be a nonempty finite vector")
    if not np.isclose(np.linalg.norm(vector), 1, atol=1e-4):
        raise ValueError("embedding must have unit norm")
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        embed(text, length)
        times.append(1000 * (time.perf_counter() - start))
    return vector, statistics.median(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--lengths", type=int, nargs="+", required=True)
    parser.add_argument("--url")
    parser.add_argument("--prompt-name")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--cpu-reference", action="store_true")
    parser.add_argument("--cpu-checkpoint", type=Path,
                        help="use a different local checkpoint directory for the reference")
    parser.add_argument("--max-input", type=int,
                        help="also check that the server rejects max-input + 1 tokens")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not (args.url or args.cpu_reference or args.cpu_checkpoint) or args.repeats < 1:
        parser.error("select a backend and at least one timed repeat")
    if args.max_input is not None and (not args.url or args.max_input not in args.lengths):
        parser.error("--max-input requires a server and a matching length to test")

    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    cases = list(fixtures(tokenizer, args.lengths, args.prefix))
    cpu = None
    if args.cpu_reference or args.cpu_checkpoint:
        cpu = cpu_reference(args.cpu_checkpoint or args.checkpoint, args.prefix, max(args.lengths))
    url = args.url.rstrip("/") if args.url else None

    def server(text, length):
        return embed_server(url, args.model, args.prompt_name, text, length)

    for length, text in cases:
        record = {"model": args.model, "tokens": length, "tail": text.split()[-1]}
        if cpu:
            reference, record["cpu_ms"] = measure(cpu, text, length, args.repeats)
        if url:
            vector, record["npu_http_ms"] = measure(server, text, length, args.repeats)
            if cpu:
                record["cosine"] = float(np.dot(reference, vector) /
                                         (np.linalg.norm(reference) * np.linalg.norm(vector)))
        print(json.dumps(record, allow_nan=False), flush=True)

    if args.max_input is not None:
        length, text = next(fixtures(tokenizer, [args.max_input + 1], args.prefix))
        try:
            request_embedding(url, args.model, args.prompt_name, text)
        except HTTPError as exc:
            error = json.load(exc)["error"]
            print(json.dumps({"model": args.model, "tokens": length,
                              "http_status": exc.code, "error": error}), flush=True)
            if exc.code != 400:
                raise ValueError("over-limit input should be rejected with HTTP 400") from exc
        else:
            raise ValueError("server accepted an over-limit input")


if __name__ == "__main__":
    main()
