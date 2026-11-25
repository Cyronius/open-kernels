"""Summarize committed prefill blocks from a live or saved Phlegm server log."""

import argparse
import json
from pathlib import Path
from statistics import median

from run import read_profile


def summarize(path, prompt_tokens, interval):
    # A live writer may have emitted only part of its last timing record.
    records = read_profile(path, complete_lines_only=True)
    ranges = []
    pending = []
    position = 0
    for row in records:
        if row["position"] >= prompt_tokens:
            break
        if row["position"] != position or not 0 < row["rows"] <= prompt_tokens - position:
            raise ValueError(f"expected contiguous prefill at position {position}")
        position += row["rows"]
        pending.append(row)
        if position - pending[0]["position"] >= interval:
            ranges.append(summarize_interval(pending))
            pending.clear()
    if pending:
        ranges.append(summarize_interval(pending))
    return {
        "prompt_tokens": prompt_tokens,
        "committed_tokens": position,
        "prefill_complete": position == prompt_tokens,
        "profiled_prefill_seconds": sum(row["profiled_seconds"] for row in ranges),
        "intervals": ranges,
    }


def summarize_interval(rows):
    token_count = sum(row["rows"] for row in rows)
    elapsed = sum(row["block_ms"] for row in rows) / 1000
    return {
        "first_position": rows[0]["position"],
        "end_position": rows[-1]["position"] + rows[-1]["rows"],
        "blocks": len(rows),
        "profiled_seconds": elapsed,
        "profiled_tokens_per_second": token_count / elapsed if elapsed else None,
        "median_ms": {name: median(row[name] for row in rows) for name in rows[0] if name.endswith("_ms")},
        "attention_host_calls": sum(row.get("attention_host_calls", 0) for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--interval", type=int, default=4096)
    args = parser.parse_args()
    if min(args.prompt_tokens, args.interval) <= 0:
        parser.error("prompt length and interval must be positive")
    print(json.dumps(summarize(args.log, args.prompt_tokens, args.interval), indent=2))


if __name__ == "__main__":
    main()
