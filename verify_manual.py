"""Check that manual generation starts paused and each step releases one token."""
import json
import queue
import threading
import time
import uuid

import httpx

BASE = "http://150.89.227.185:8090"
SESSION = uuid.uuid4().hex
events = queue.Queue()
failures = []
body = {
    "session_id": SESSION,
    "model": "openai/gpt-oss-20b",
    "messages": [{"role": "user", "content": "1から10までの数を順に書いてください。"}],
    "temperature": 0,
    "top_k": 0,
    "top_p": 1,
    "seed": 42,
    "max_tokens": 4,
    "delay_ms": 0,
    "manual_mode": True,
}


def stream():
    try:
        with httpx.Client(timeout=300) as client:
            with client.stream("POST", BASE + "/api/generate", json=body) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        events.put(json.loads(line))
    except Exception as exc:
        failures.append(exc)
    finally:
        events.put({"type": "closed"})


def next_event(kind, timeout=240):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        event = events.get(timeout=max(0.1, deadline - time.monotonic()))
        if event["type"] in ("error", "closed"):
            raise AssertionError(event if not failures else failures[0])
        if event["type"] == kind:
            return event
    raise AssertionError(f"No {kind} event")


def no_token(seconds=0.7):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            event = events.get(timeout=deadline - time.monotonic())
        except queue.Empty:
            break
        assert event["type"] not in ("token", "done", "error", "closed"), event


def control(action):
    with httpx.Client(timeout=10) as client:
        response = client.post(BASE + "/api/control", json={"session_id": SESSION, "action": action})
        response.raise_for_status()
        return response.json()


thread = threading.Thread(target=stream, daemon=True)
thread.start()
start = next_event("start")
assert start["manual_mode"] is True
no_token()
assert control("step")["playing"] is False
assert next_event("token")["step"] == 1
no_token()
assert control("step")["playing"] is False
assert next_event("token")["step"] == 2
no_token()
assert control("resume")["playing"] is True
assert next_event("done")["reason"] in ("length", "eos")
thread.join(timeout=10)
assert not failures, failures
print("manual start / one token per step / resume: OK")
