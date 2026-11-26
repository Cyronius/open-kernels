# Needle prompts for qwen36moe: needle.py make <tokens> <depth> <out.txt> | needle.py check <cli.log> (2026-11-24).
import re
import sys
from pathlib import Path

from tokenizers import Tokenizer

M = Path(r"C:\models\qwen36moe\Qwen3.6-35B-A3B-UD-Q4_K_S")
tok = Tokenizer.from_file(str(M / "tokenizer.json"))
NEEDLE = "The secret code for the blue vault is 58213."
ASK = "\n\nWhat is the secret code for the blue vault? Reply with the number only."


def make(n, depth, out):
    essays = Path(r"C:\models\qwen36moe\kl\essays.txt").read_text(encoding="utf-8")
    hay = tok.encode(essays, add_special_tokens=False).ids
    pre = tok.encode("<|im_start|>user\n", add_special_tokens=False).ids
    post = tok.encode(ASK + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False).ids
    nd = tok.encode(" " + NEEDLE + " ", add_special_tokens=False).ids
    body_n = n - len(pre) - len(post) - len(nd)
    body = (hay * (body_n // len(hay) + 1))[:body_n]
    at = int(body_n * depth)
    # cut at a sentence end so the needle does not split a token run mid-word
    while at > 0 and not tok.decode(body[at - 1:at]).rstrip().endswith("."):
        at -= 1
    ids = pre + body[:at] + nd + body[at:] + post
    Path(out).write_text(",".join(map(str, ids)))
    print(f"{len(ids)} tokens, needle at {len(pre) + at} ({(len(pre) + at) / len(ids):.2f})")


def check(log):
    text = Path(log).read_text(errors="replace")
    ids = [int(t) for t in re.findall(r"^token (\d+)", text, re.M)]
    ans = tok.decode(ids)
    pf = re.search(r"prefill (\d+) tokens: (\d+) ms", text)
    print(f"answer {ans!r} {'PASS' if '58213' in ans else 'FAIL'}", f"prefill {pf.group(1)} in {int(pf.group(2)) / 1000:.1f} s" if pf else "")


if sys.argv[1] == "make":
    make(int(sys.argv[2]), float(sys.argv[3]), sys.argv[4])
else:
    check(sys.argv[2])
