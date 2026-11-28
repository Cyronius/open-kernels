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


MIXED_PASSAGE = (
    "A librarian catalogued a collection of maps from coastal towns. "
    "A gardener studied the soil before planting the seedlings. "
    "The mechanic replaced a bearing and measured the vibration. "
    "A baker recorded how room temperature affected the dough. "
    "The river rose after three days of rain in the hills. "
)


def fixtures(tokenizer, lengths, prefix, mixed=False):
    tokenizer.no_truncation()
    tokenizer.no_padding()
    for length in lengths:
        for tail in ("amber", "ocean"):
            ending = f" The final subject is {tail}."
            passage = ""
            count = len(tokenizer.encode(prefix + ending).ids)
            if mixed:
                passage_tokens = len(tokenizer.encode(prefix + MIXED_PASSAGE + ending).ids) - count
                copies = max(0, (length - count) // passage_tokens)
                passage = MIXED_PASSAGE * copies
                count = len(tokenizer.encode(prefix + passage + ending).ids)
            if count > length:
                raise ValueError(f"{length} tokens cannot fit the prefix and fixture")
            text = passage + "cat " * (length - count) + ending.lstrip()
            if len(tokenizer.encode(prefix + text).ids) != length:
                raise ValueError(f"cannot construct a {length}-token fixture with this tokenizer")
            yield length, text


def request_embedding(url, model, prompt_name, text, dimensions=None):
    body = {"model": model, "input": text}
    if prompt_name is not None:
        body["prompt_name"] = prompt_name
    if dimensions is not None:
        body["dimensions"] = dimensions
    request = Request(url + "/v1/embeddings", json.dumps(body).encode(),
                      {"Content-Type": "application/json"})
    with urlopen(request, timeout=300) as response:
        return json.load(response)


def embed_server_batch(url, model, prompt_name, texts, length, dimensions=None):
    result = request_embedding(url, model, prompt_name, texts[0] if len(texts) == 1 else texts,
                               dimensions)
    if result.get("model") != model:
        raise ValueError(f"server returned model {result.get('model')!r}, expected {model!r}")
    count = result["usage"]["prompt_tokens"]
    expected = length * len(texts)
    if count != expected:
        raise ValueError(f"server processed {count} tokens, expected {expected}")
    if [item["index"] for item in result["data"]] != list(range(len(texts))):
        raise ValueError("server returned missing or out-of-order embedding indices")
    vectors = [item["embedding"] for item in result["data"]]
    if dimensions is not None and any(len(vector) != dimensions for vector in vectors):
        raise ValueError(f"server did not honor dimensions={dimensions}")
    return vectors


def cpu_reference(directory, prefix, max_length, dimensions=None):
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    config = AutoConfig.from_pretrained(directory, local_files_only=True)
    if dimensions is not None:
        nomic_width = config.model_type == "nomic_bert" and dimensions in (64, 128, 256, 512, 768)
        gte_width = config.model_type in ("gte", "new") and 128 <= dimensions <= config.hidden_size
        if not (nomic_width or gte_width):
            raise ValueError("unsupported configurable embedding width for this CPU reference")
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
    def embed(texts, length):
        inputs = tokenizer([prefix + text for text in texts], return_tensors="pt",
                           truncation=False, padding=False)
        if inputs.input_ids.shape != (len(texts), length):
            raise ValueError("reference tokenizer disagrees with the fixture length")
        # Transformers keeps a dynamic rotary base after a longer call unless the next is below 2048 tokens.
        for module in model.modules():
            if hasattr(module, "original_inv_freq") and hasattr(module, "original_max_seq_len"):
                module.inv_freq = module.original_inv_freq.clone()
                module.max_seq_len_cached = module.original_max_seq_len
        hidden = model(**inputs).last_hidden_state
        pooled = hidden[:, 0] if use_cls else hidden.mean(dim=1)
        if config.model_type == "nomic_bert":
            pooled = torch.nn.functional.layer_norm(pooled, (pooled.shape[-1],))
        if dimensions is not None:
            pooled = pooled[:, :dimensions]
        return torch.nn.functional.normalize(pooled, dim=1).numpy()

    return embed


def measure(embed, texts, length, repeats):
    vectors = np.asarray(embed(texts, length), dtype=np.float64)
    if (vectors.ndim != 2 or vectors.shape[0] != len(texts) or not vectors.size
            or not np.isfinite(vectors).all()):
        raise ValueError("expected one nonempty finite vector per input")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError("embeddings must have unit norm")
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        embed(texts, length)
        times.append(1000 * (time.perf_counter() - start))
    return vectors, statistics.median(times)


def compare(record, texts, length, cpu, server, repeats):
    if cpu:
        reference, record["cpu_ms"] = measure(cpu, texts, length, repeats)
    if server:
        vectors, record["npu_http_ms"] = measure(server, texts, length, repeats)
        if cpu:
            if reference.shape != vectors.shape:
                raise ValueError("embedding width differs from the CPU reference")
            norms = np.linalg.norm(reference, axis=1) * np.linalg.norm(vectors, axis=1)
            record["cosine"] = float(np.min((reference * vectors).sum(axis=1) / norms))
    print(json.dumps(record, allow_nan=False), flush=True)
    return record.get("cosine")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--lengths", type=int, nargs="+", required=True)
    parser.add_argument("--url")
    parser.add_argument("--prompt-name")
    parser.add_argument("--dimensions", type=int)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--mixed", action="store_true",
                        help="fill long probes with varied passages and keep an exact token count")
    parser.add_argument("--cpu-reference", action="store_true")
    parser.add_argument("--cpu-checkpoint", type=Path,
                        help="use a different local checkpoint directory for the reference")
    parser.add_argument("--min-cosine", type=float,
                        help="fail if any single or batched embedding falls below this similarity")
    parser.add_argument("--max-input", type=int,
                        help="also check that the server rejects max-input + 1 tokens")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batches", type=int, nargs="+", default=[],
                        help="also measure ordered batches of these sizes")
    args = parser.parse_args()
    if not (args.url or args.cpu_reference or args.cpu_checkpoint) or args.repeats < 1:
        parser.error("select a backend and at least one timed repeat")
    if args.max_input is not None and (not args.url or args.max_input not in args.lengths):
        parser.error("--max-input requires a server and a matching length to test")
    if args.min_cosine is not None and (not args.url or not (args.cpu_reference or args.cpu_checkpoint)
                                        or not -1 <= args.min_cosine <= 1):
        parser.error("--min-cosine requires both backends and a threshold between -1 and 1")
    if any(batch < 1 for batch in args.batches):
        parser.error("--batches must be positive")

    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    cases = list(fixtures(tokenizer, args.lengths, args.prefix, args.mixed))
    cpu = None
    if args.cpu_reference or args.cpu_checkpoint:
        cpu = cpu_reference(args.cpu_checkpoint or args.checkpoint, args.prefix,
                            max(args.lengths), args.dimensions)
    url = args.url.rstrip("/") if args.url else None

    def server(texts, length):
        return embed_server_batch(url, args.model, args.prompt_name, texts, length, args.dimensions)

    similarities = []
    for length, text in cases:
        record = {"model": args.model, "tokens": length, "tail": text.split()[-1]}
        similarities.append(compare(record, [text], length, cpu, server if url else None, args.repeats))

    for length in args.lengths:
        texts = [text for size, text in cases if size == length]
        for batch in args.batches:
            inputs = [texts[index % len(texts)] for index in range(batch)]
            record = {"model": args.model, "tokens": length, "batch": batch}
            similarities.append(compare(record, inputs, length, cpu, server if url else None, args.repeats))

    parity_failure = False
    if args.min_cosine is not None:
        worst = min(similarities)
        parity_failure = worst < args.min_cosine
        print(json.dumps({"model": args.model, "worst_cosine": worst, "required_cosine": args.min_cosine,
                          "parity_pass": not parity_failure}), flush=True)

    if args.max_input is not None:
        length, text = next(fixtures(tokenizer, [args.max_input + 1], args.prefix, args.mixed))
        try:
            request_embedding(url, args.model, args.prompt_name, text, args.dimensions)
        except HTTPError as exc:
            error = json.load(exc)["error"]
            print(json.dumps({"model": args.model, "tokens": length,
                              "http_status": exc.code, "error": error}), flush=True)
            if exc.code != 400:
                raise ValueError("over-limit input should be rejected with HTTP 400") from exc
        else:
            raise ValueError("server accepted an over-limit input")

    if parity_failure:
        raise ValueError("embedding parity missed the requested similarity threshold")


if __name__ == "__main__":
    main()
