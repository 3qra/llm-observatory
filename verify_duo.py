"""Live integration checks for the two-screen protocol and real inference."""
import argparse
import json
import threading
import time
import uuid

import httpx

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://150.89.227.185:8090")
parser.add_argument("--gpt-oss", action="store_true")
args = parser.parse_args()
client = httpx.Client(base_url=args.url, timeout=300)
model = "openai/gpt-oss-20b" if args.gpt_oss else "Qwen/Qwen3-0.6B"
available = {item["id"]: item["available"] for item in client.get("/api/models").json()["models"]}
assert available.get(model), f"Model not installed: {model}"
session = uuid.uuid4().hex


def payload(label="single", temperature=0.7, comparison=None, max_tokens=5, delay_ms=0,
            prompt="大阪の名物を3つ教えて"):
    return {"model": model, "session_id": session, "run_label": label,
            "comparison_id": comparison, "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "top_k": 20, "top_p": 0.95,
            "seed": 42, "max_tokens": max_tokens, "delay_ms": delay_ms}


def generate(data):
    events = []
    with client.stream("POST", "/api/generate", json=data) as response:
        assert response.status_code == 200, response.read()
        for line in response.iter_lines():
            event = json.loads(line)
            assert event["type"] != "error", event
            events.append(event)
    assert events[-1]["type"] == "done", events[-1]
    return events


first = generate(payload())
start = next(e for e in first if e["type"] == "start")
pca = next(e for e in first if e["type"] == "pca")
tokens = [e for e in first if e["type"] == "token"]
assert start["tokens"] and len(pca["points"]) == len(start["tokens"])
assert all(point["index"] == i and point["id"] == start["tokens"][i]["id"]
           for i, point in enumerate(pca["points"]))
assert tokens and all(e["run_id"] == start["run_id"] for e in tokens)
assert all(len(e["hidden"]) == start["layers"] + 1 for e in tokens)
print("real generation + input-token PCA: OK", flush=True)

replayed = []
with client.stream("GET", f"/api/events/{session}") as response:
    assert response.status_code == 200
    for line in response.iter_lines():
        if line.startswith("data: "):
            event = json.loads(line[6:])
            replayed.append(event)
            if event["type"] == "done":
                break
assert [e["type"] for e in replayed] == [e["type"] for e in first]
print("late-opened visualization SSE replay: OK", flush=True)

comparison = uuid.uuid4().hex
a = generate(payload("A", 0.0, comparison))
b = generate(payload("B", 1.3, comparison))
assert next(e for e in a if e["type"] == "start")["settings"]["temperature"] == 0.0
assert next(e for e in b if e["type"] == "start")["settings"]["temperature"] == 1.3
assert a[0]["run_id"] != b[0]["run_id"]
assert all(e["comparison_id"] == comparison for e in a+b)
print("same-question temperature A/B: OK", flush=True)

if not args.gpt_oss:
    pause_events = []
    first_token = threading.Event()
    finished = threading.Event()
    def collect():
        try:
            with client.stream("POST", "/api/generate", json=payload(max_tokens=32, delay_ms=300,
                                 prompt="1から100までの整数を順番に、途中で省略せず列挙してください。")) as response:
                assert response.status_code == 200
                for line in response.iter_lines():
                    event = json.loads(line)
                    pause_events.append(event)
                    if event["type"] == "token": first_token.set()
        finally:
            finished.set()
    thread = threading.Thread(target=collect, daemon=True)
    thread.start()
    assert first_token.wait(120), "No first token"
    paused = client.post("/api/control", json={"session_id": session, "action": "pause"})
    assert paused.status_code == 200 and not paused.json()["playing"]
    time.sleep(0.8)
    paused_count = sum(e["type"] == "token" for e in pause_events)
    stepped = client.post("/api/control", json={"session_id": session, "action": "step"})
    assert stepped.status_code == 200 and not stepped.json()["playing"]
    deadline = time.monotonic() + 30
    while sum(e["type"] == "token" for e in pause_events) <= paused_count and time.monotonic() < deadline:
        time.sleep(0.1)
    assert sum(e["type"] == "token" for e in pause_events) > paused_count, "Step produced no token"
    resumed = client.post("/api/control", json={"session_id": session, "action": "resume"})
    assert resumed.status_code == 200 and resumed.json()["playing"]
    assert finished.wait(120), "Run did not finish"
    assert pause_events[-1]["type"] == "done"
    print("pause / single-token step / resume: OK", flush=True)

print(json.dumps({"model": model, "checks": "passed", "session": session}, ensure_ascii=False))
