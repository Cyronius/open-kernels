# phlegm serve feature check (2026-11-26): thinking on / off, streaming, a tool-call round trip.
# serve_features.py <base url>
import json
import sys
import time
import urllib.request

BASE = sys.argv[1]
for _ in range(900):
    try:
        urllib.request.urlopen(BASE + "/v1/models", timeout=5)
        break
    except Exception:
        time.sleep(2)


def post(body, stream=False):
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=1800)
    if not stream:
        return json.load(r)
    chunks = []
    for line in r:
        line = line.decode().strip()
        if line.startswith("data:") and line != "data: [DONE]":
            chunks.append(json.loads(line[5:]))
    return chunks


ok = True
r = post({"messages": [{"role": "user", "content": "What is 17 * 23? Reply with the number only."}], "max_tokens": 16,
          "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
m = r["choices"][0]["message"]
print("thinking off:", repr(m["content"]), "reasoning", repr(m.get("reasoning_content")))
ok &= "391" in (m["content"] or "")
r = post({"messages": [{"role": "user", "content": "What is 17 * 23? Reply with the number only."}], "max_tokens": 1024,
          "temperature": 0})
m = r["choices"][0]["message"]
print("thinking on:", repr(m["content"]), f"reasoning {len(m.get('reasoning_content') or '')} chars,", r["choices"][0]["finish_reason"])
ok &= "391" in (m["content"] or "") and len(m.get("reasoning_content") or "") > 0
ch = post({"messages": [{"role": "user", "content": "Name three colours of the rainbow, comma separated."}], "max_tokens": 32,
           "temperature": 0, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}, stream=True)
text = "".join(c["choices"][0]["delta"].get("content") or "" for c in ch if c.get("choices"))
print("streamed:", repr(text), len(ch), "chunks")
ok &= len(ch) > 3 and "," in text
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather in a city",
                                           "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                          "required": ["city"]}}}]
msgs = [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}]
r = post({"messages": msgs, "tools": tools, "max_tokens": 256, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
c = r["choices"][0]
calls = c["message"].get("tool_calls") or []
print("tool call:", c["finish_reason"], [(t["function"]["name"], t["function"]["arguments"]) for t in calls])
ok &= c["finish_reason"] == "tool_calls" and calls and calls[0]["function"]["name"] == "get_weather"
if calls:
    msgs += [{"role": "assistant", "content": c["message"].get("content") or "", "tool_calls": calls},
             {"role": "tool", "tool_call_id": calls[0]["id"], "content": json.dumps({"weather": "light rain", "temp_c": 14})}]
    r = post({"messages": msgs, "tools": tools, "max_tokens": 96, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
    m = r["choices"][0]["message"]
    print("after the tool:", repr(m["content"]), r.get("usage"))
    ok &= "14" in (m["content"] or "")
print("FEATURES PASS" if ok else "FEATURES FAIL")
