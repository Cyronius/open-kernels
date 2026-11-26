# Decode rate through phlegm serve, from streamed chunk times: (chunks - 1) / (last - first); greedy (2026-11-25).
import json
import os
import sys
import time
import urllib.request

BASE, tag = sys.argv[1], sys.argv[2]
PROMPTS = {
    "rewrite": open(sys.argv[3]).read() if len(sys.argv) > 3 else "",
}
for _ in range(600):
    try:
        urllib.request.urlopen(BASE + "/v1/models", timeout=5)
        break
    except Exception:
        time.sleep(2)
for name, text in [("rewrite", PROMPTS["rewrite"]),
                   ("explain", "Explain in detail how a hash map works, with its resizing and collision handling.")]:
    body = {"messages": [{"role": "user", "content": text}], "max_tokens": 256, "temperature": 0, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    times, out = [], []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])["choices"][0]["delta"]
            if d.get("content"):
                times.append(time.time())
                out.append(d["content"])
    n = len(times)
    rate = (n - 1) / (times[-1] - times[0]) if n > 1 else 0
    open(os.path.join(os.environ.get("QW", os.path.expanduser("~/models/qwen36moe")), "mtp", f"serve_{tag}_{name}.txt"), "w", encoding="utf-8").write("".join(out))
    print(f"{tag} {name}: {n} chunks, {rate:.2f} chunks/s, {1000 / rate if rate else 0:.1f} ms/chunk")
# one more short request makes the bridge print the explain request's stats
body = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False}}
urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                              {"Content-Type": "application/json"}), timeout=600).read()
