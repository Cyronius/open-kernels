# A needle at depth 0.5 in a chat message of about N tokens, through phlegm serve (2026-11-26):
# needle_serve.py <base url> <tokens> -- greedy, thinking off; prints the answer, PASS / FAIL and the wall time.
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer

BASE, N = sys.argv[1], int(sys.argv[2])
W = Path(os.environ.get("QW", os.path.expanduser("~/models/qwen36moe")))
M = Path(os.environ.get("QM", W / "Qwen3.6-35B-A3B-UD-Q4_K_S"))
tok = Tokenizer.from_file(str(M / "tokenizer.json"))
NEEDLE = " The secret code for the blue vault is 58213. "
ASK = "\n\nWhat is the secret code for the blue vault? Reply with the number only."
essays = (W / "kl" / "essays.txt").read_text(encoding="utf-8")
hay = tok.encode(essays, add_special_tokens=False).ids
body_n = N - 40   # the chat template, the needle and the question
body = (hay * (body_n // len(hay) + 1))[:body_n]
at = body_n // 2
while at > 0 and not tok.decode(body[at - 1:at]).rstrip().endswith("."):
    at -= 1
text = tok.decode(body[:at]) + NEEDLE + tok.decode(body[at:]) + ASK
for _ in range(1800):
    try:
        urllib.request.urlopen(BASE + "/v1/models", timeout=5)
        break
    except Exception:
        time.sleep(2)
body = {"messages": [{"role": "user", "content": text}], "max_tokens": 12, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
t = time.time()
r = json.load(urllib.request.urlopen(req, timeout=36000))
ans = r["choices"][0]["message"]["content"]
print(f"{N} tokens: answer {ans!r} {'PASS' if '58213' in ans else 'FAIL'}, {time.time() - t:.0f} s, usage {r.get('usage')}",
      flush=True)
