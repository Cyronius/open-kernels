# An image chat through phlegm serve (SERVER-IMAGE-INPUT, OPEN-VISION-BLOCK, 2026-11-26):
# vision_serve.py <base url> <image file> -- a question about the image, then a text-only follow-up in the same
# conversation, then the same image again (prompt reuse must not mix images up), greedy, thinking off.
import base64
import json
import sys
import time
import urllib.request

BASE, image = sys.argv[1], sys.argv[2]
for _ in range(900):
    try:
        urllib.request.urlopen(BASE + "/v1/models", timeout=5)
        break
    except Exception:
        time.sleep(2)
url = "data:image/png;base64," + base64.b64encode(open(image, "rb").read()).decode()


def chat(messages, max_tokens=96):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    m = r["choices"][0]["message"]
    print(f"[{time.time() - t:.1f} s, usage {r.get('usage')}] {m['content']!r}", flush=True)
    return m


q1 = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                   {"type": "text", "text": "What text is written in this image, and what animal is shown? Answer in two short sentences."}]}]
a1 = chat(q1)
q2 = q1 + [{"role": "assistant", "content": a1["content"]},
           {"role": "user", "content": "What colour is the animal's fur? One word."}]
chat(q2, 16)
chat([{"role": "user", "content": [{"type": "text", "text": "Describe this image in one sentence."},
                                   {"type": "image_url", "image_url": {"url": url}}]}])
