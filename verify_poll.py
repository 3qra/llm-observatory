"""Verify the browser's short-poll manual generation protocol on the live app."""
import sys
import time
import uuid

import httpx

BASE = "http://150.89.227.185:8090"
session = uuid.uuid4().hex
print(f"poll test session: {session}", flush=True)
body = {
    "session_id": session,
    "model": sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-oss-20b",
    "messages": [{"role": "user", "content": "1から10までの数を順に書いてください。"}],
    "temperature": 0,
    "top_k": 0,
    "top_p": 1,
    "seed": 42,
    "max_tokens": 4,
    "delay_ms": 0,
    "manual_mode": True,
}

with httpx.Client(timeout=15) as client:
    response = client.post(BASE + "/api/start", json=body)
    response.raise_for_status()
    cursor = response.json()["cursor"]

    def next_event(kind, timeout=900):  # gpt-oss-120b first load can take minutes
        global cursor
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = client.get(BASE + f"/api/events/{session}/poll", params={"cursor": cursor})
            result.raise_for_status()
            batch = result.json()
            cursor = batch["cursor"]
            for event in batch["events"]:
                if event["type"] == "error":
                    raise AssertionError(event)
                if event["type"] == kind:
                    return event
        raise AssertionError(f"No {kind} event")

    def action(name):
        result = client.post(BASE + "/api/control", json={"session_id": session, "action": name})
        result.raise_for_status()

    assert next_event("start")["manual_mode"] is True
    action("step")
    first = next_event("token")
    assert first["step"] == 1
    action("step")
    second = next_event("token")
    assert second["step"] == 2
    action("back")
    assert next_event("rewind")["step"] == 1
    action("step")
    replay = next_event("token")
    assert replay["step"] == 2 and replay["id"] == second["id"]
    action("resume")
    assert next_event("done")["tokens"] == 4
    print("short-poll manual steps, rewind, resume: OK")
