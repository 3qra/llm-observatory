"""Real-time, local Transformer inspection. No synthetic inference data."""
import asyncio
import gc
import json
import logging
import os
from pathlib import Path
import queue
import threading
import time
import uuid
import re

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent
MODELS = [
    {"id": "Qwen/Qwen3-0.6B", "name": "Qwen3 · 0.6B", "description": "軽量・すばやく観察", "layers": 28},
    {"id": "Qwen/Qwen3-8B", "name": "Qwen3 · 8B", "description": "高品質・36層を観察", "layers": 36},
    {"id": "openai/gpt-oss-20b", "name": "OpenAI gpt-oss · 20B", "description": "MoE・36層 / 初回読み込みに時間がかかります", "layers": 36},
]
MAX_CONTEXT = 512
WINDOW = 48
app = FastAPI(title="LLM Observatory", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
gpu_lock = threading.Lock()
loaded = {"id": None, "model": None, "tokenizer": None}
logger = logging.getLogger("observatory")
session_lock = threading.Lock()
sessions = {}
active_run = None


def get_session(session_id):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{8,64}", session_id):
        raise HTTPException(400, "セッションIDが無効です。")
    with session_lock:
        now = time.monotonic()
        for key in list(sessions):
            if now - sessions[key]["touched"] > 3600:
                del sessions[key]
        if session_id not in sessions:
            sessions[session_id] = {"events": [], "offset": 0, "touched": now}
        sessions[session_id]["touched"] = now
        return sessions[session_id]


def publish(session, payload):
    if session is None:
        return
    with session_lock:
        session["events"].append(payload)
        if len(session["events"]) > 1024:
            del session["events"][:256]
            session["offset"] += 256
        session["touched"] = time.monotonic()


class RunControl:
    def __init__(self, session_id):
        self.session_id = session_id
        self.condition = threading.Condition()
        self.playing = True
        self.budget = 0

    def change(self, action):
        with self.condition:
            if action == "pause":
                self.playing = False
            elif action == "resume":
                self.playing, self.budget = True, 0
            elif action == "step":
                self.playing = False
                self.budget += 1
            self.condition.notify_all()
            return {"type": "control", "playing": self.playing, "step_budget": self.budget}

    def wait(self, stop):
        with self.condition:
            while not self.playing and not self.budget and not stop.is_set():
                self.condition.wait(timeout=0.2)
            if stop.is_set():
                return False
            if not self.playing:
                self.budget -= 1
            return True


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=4000)


class GenerateRequest(BaseModel):
    model: str = "Qwen/Qwen3-0.6B"
    messages: list[Message] = Field(min_length=1, max_length=24)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_k: int = Field(default=20, ge=0, le=100)
    top_p: float = Field(default=0.95, ge=0.05, le=1)
    max_tokens: int = Field(default=128, ge=1, le=256)
    seed: int = Field(default=42, ge=0, le=2147483647)
    delay_ms: int = Field(default=80, ge=0, le=1000)
    session_id: str | None = None
    run_label: str = Field(default="single", pattern="^(single|A|B)$")
    comparison_id: str | None = Field(default=None, max_length=64)


def load_model(model_id):
    if loaded["id"] == model_id:
        return loaded["model"], loaded["tokenizer"]
    loaded.update(id=None, model=None, tokenizer=None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True, trust_remote_code=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto" if model_id.startswith("openai/") else dtype, attn_implementation="eager",
        local_files_only=True, trust_remote_code=False,
    ).to(device).eval()
    loaded.update(id=model_id, model=model, tokenizer=tokenizer)
    return model, tokenizer


def token_label(tokenizer, token_id):
    decoded = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    if "\ufffd" in decoded:
        return str(tokenizer.convert_ids_to_tokens(int(token_id)))
    return decoded


def locate_user_tokens(tokenizer, messages, template_args, context_ids):
    """Map visible user chips to exact positions of the model's prompt IDs."""
    latest = messages[-1]["content"]
    try:
        rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                  tokenize=False, **template_args)
        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        if encoded["input_ids"] == context_ids:
            start_char = rendered.rfind(latest)
            end_char = start_char + len(latest)
            positions = [i for i, (a, b) in enumerate(encoded["offset_mapping"])
                         if start_char >= 0 and b > start_char and a < end_char and b > a]
            if positions:
                return positions[0], context_ids[positions[0]:positions[-1] + 1]
    except (TypeError, ValueError, KeyError):
        pass
    user_ids = tokenizer.encode(latest, add_special_tokens=False)
    for pos in range(len(context_ids)-len(user_ids), -1, -1):
        if context_ids[pos:pos+len(user_ids)] == user_ids:
            return pos, user_ids
    return -1, []


def sampling_distribution(logits, temperature, top_k, top_p):
    """Full-vocabulary probabilities after precisely the displayed controls."""
    if temperature == 0:
        probs = torch.zeros_like(logits)
        probs[logits.argmax()] = 1
        return probs
    scores = logits / temperature
    if top_k:
        scores = scores.masked_fill(scores < torch.topk(scores, min(top_k, scores.numel())).values[-1], -torch.inf)
    if top_p < 1:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        cumulative = torch.softmax(sorted_scores, -1).cumsum(-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        scores[sorted_indices[remove]] = -torch.inf
    return torch.softmax(scores, -1)


def infer(data, events, stop, control, session, run_id):
    def emit(payload):
        payload.update(run_id=run_id, run_label=data.run_label, comparison_id=data.comparison_id)
        publish(session, payload)
        while not stop.is_set():
            try:
                events.put(payload, timeout=0.2)
                return True
            except queue.Full:
                pass
        return False

    try:
        emit({"type": "status", "message": "モデルをGPUに読み込んでいます…"})
        model, tokenizer = load_model(data.model)
        if stop.is_set():
            return
        role = "developer" if data.model.startswith("openai/") else "system"
        messages = [{"role": role, "content": "あなたは親切なアシスタントです。日本語で簡潔に回答してください。"}]
        messages += [m.model_dump() for m in data.messages]
        dropped = 0
        while True:
            template_args = {"reasoning_effort": "low"} if data.model.startswith("openai/") else {"enable_thinking": False}
            ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", **template_args)
            if ids.shape[-1] <= MAX_CONTEXT:
                break
            if len(messages) <= 2:
                emit({"type": "error", "message": f"入力が長すぎます。会話を含め{MAX_CONTEXT}トークン以内になるよう短くしてください。"})
                return
            # Remove an old user turn together with its assistant response.
            del messages[1]
            dropped += 1
            if len(messages) > 2 and messages[1]["role"] == "assistant":
                del messages[1]
                dropped += 1
        ids = ids.to(model.device)
        context_ids = ids[0].tolist()
        prompt_count = len(context_ids)
        user_start, user_ids = locate_user_tokens(tokenizer, messages, template_args, context_ids)
        config = model.config
        emit({"type": "start", "model": data.model,
              "tokens": [{"id": i, "text": token_label(tokenizer, i)} for i in user_ids],
              "user_start": user_start,
              "settings": {"temperature": data.temperature, "top_k": data.top_k, "top_p": data.top_p, "seed": data.seed},
              "prompt_tokens": prompt_count, "dropped_messages": dropped,
              "layers": config.num_hidden_layers, "heads": config.num_attention_heads,
              "hidden_size": config.hidden_size, "dtype": str(model.dtype).split(".")[-1],
              "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"})
        rng = torch.Generator(device=model.device).manual_seed(data.seed)
        cache = None
        generated = []
        elapsed_compute = 0
        start = time.perf_counter()
        eos_ids = model.generation_config.eos_token_id
        if not isinstance(eos_ids, list):
            eos_ids = [eos_ids]
        reason = "length"
        with torch.inference_mode():
            for step in range(data.max_tokens):
                if not control.wait(stop):
                    return
                tick = time.perf_counter()
                outputs = model(input_ids=ids, past_key_values=cache, use_cache=True,
                                output_attentions=True, output_hidden_states=True, return_dict=True)
                if step == 0 and user_ids:
                    vectors = outputs.hidden_states[-1][0, user_start:user_start+len(user_ids)].float().cpu()
                    centered = vectors - vectors.mean(0, keepdim=True)
                    if len(user_ids) >= 2:
                        try:
                            _, _, basis = torch.pca_lowrank(centered, q=min(2, len(user_ids)))
                            coords = centered @ basis[:, :2]
                        except RuntimeError:
                            coords = torch.zeros((len(user_ids), 2))
                    else:
                        coords = torch.zeros((len(user_ids), 2))
                    if coords.shape[-1] == 1:
                        coords = torch.nn.functional.pad(coords, (0, 1))
                    emit({"type": "pca", "points": [{"index": j, "id": i, "text": token_label(tokenizer, i),
                           "x": float(coords[j, 0]), "y": float(coords[j, 1])} for j, i in enumerate(user_ids)],
                           "layer": config.num_hidden_layers})
                    del vectors, centered, coords
                cache = outputs.past_key_values
                logits = outputs.logits[0, -1].float()
                probs = sampling_distribution(logits.clone(), data.temperature, data.top_k, data.top_p)
                next_id = int(torch.multinomial(probs, 1, generator=rng).item())
                best_probs, best_ids = torch.topk(probs, 8)
                candidates = [{"id": int(i), "text": token_label(tokenizer, i),
                               "prob": float(p), "logit": float(logits[i])}
                              for i, p in zip(best_ids.tolist(), best_probs.tolist()) if p > 0]
                states = torch.stack([s[0, -1].float() for s in outputs.hidden_states])
                # RMS over contiguous dimension groups, not invented activations.
                grouped = torch.stack([part.square().mean(-1).sqrt() for part in torch.tensor_split(states, 48, dim=-1)], dim=-1)
                scale = grouped.amax(-1, keepdim=True).clamp_min(1e-8)
                hidden = (grouped / scale).cpu().tolist()
                hidden_scale = scale.flatten().cpu().tolist()
                # Last context query; average all heads; retain actual (unrenormalized) weights.
                if not outputs.attentions or outputs.attentions[0] is None:
                    raise RuntimeError("Model did not return attention weights")
                attention = torch.stack([a[0, :, -1, -WINDOW:].float().mean(0) for a in outputs.attentions]).cpu().tolist()
                offset = max(0, len(context_ids) - WINDOW)
                keys = [{"position": offset+j, "id": i, "text": token_label(tokenizer, i)}
                        for j, i in enumerate(context_ids[-WINDOW:])]
                query = token_label(tokenizer, context_ids[-1])
                generated.append(next_id)
                text = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                # An unfinished UTF-8 byte sequence should not flash as replacement text.
                text = text.rstrip("\ufffd")
                elapsed_compute += time.perf_counter() - tick
                payload = {"type": "token", "step": step+1, "id": next_id,
                           "token": token_label(tokenizer, next_id), "text": text,
                           "eos": next_id in eos_ids, "selected_prob": float(probs[next_id]),
                           "selected_logit": float(logits[next_id]),
                           "candidates": candidates, "other_prob": max(0.0, 1-sum(c["prob"] for c in candidates)),
                           "hidden": hidden, "hidden_scale": hidden_scale,
                           "attention": attention, "keys": keys, "query": query,
                           "context_tokens": len(context_ids), "compute_tps": (step+1)/elapsed_compute,
                           "seconds": time.perf_counter()-start}
                if not emit(payload):
                    return
                del outputs, states, grouped, scale, logits, probs
                context_ids.append(next_id)
                if next_id in eos_ids:
                    reason = "eos"
                    break
                ids = torch.tensor([[next_id]], device=model.device)
                if stop.wait(data.delay_ms / 1000):
                    return
        emit({"type": "done", "reason": reason, "tokens": len(generated),
              "seconds": time.perf_counter()-start, "compute_tps": len(generated)/max(elapsed_compute, 1e-6)})
    except Exception as exc:
        logger.exception("Generation failed")
        emit({"type": "error", "message": f"生成に失敗しました（{type(exc).__name__}）。サーバーログを確認してください。"})
    finally:
        global active_run
        if active_run is control:
            active_run = None
        # Never release the GPU to another request until this worker has really stopped.
        gpu_lock.release()
        while not stop.is_set():
            try:
                events.put(None, timeout=0.2)
                break
            except queue.Full:
                pass


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "launcher.html")


@app.get("/chat")
def chat():
    return FileResponse(ROOT / "static" / "chat.html")


@app.get("/viz")
def viz():
    return FileResponse(ROOT / "static" / "viz.html")


@app.get("/single")
def single():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/models")
def models():
    from huggingface_hub import try_to_load_from_cache
    items = []
    for model in MODELS:
        config = try_to_load_from_cache(model["id"], "config.json")
        index = try_to_load_from_cache(model["id"], "model.safetensors.index.json")
        weights = try_to_load_from_cache(model["id"], "model.safetensors")
        available = isinstance(config, str) and Path(config).exists()
        if isinstance(index, str) and Path(index).exists():
            with open(index, encoding="utf-8") as handle:
                shards = set(json.load(handle)["weight_map"].values())
            available = available and all(isinstance(path := try_to_load_from_cache(model["id"], shard), str)
                                          and Path(path).exists() for shard in shards)
        else:
            available = available and isinstance(weights, str) and Path(weights).exists()
        items.append({**model, "available": available})
    return {"models": items, "loaded": loaded["id"], "max_context": MAX_CONTEXT}


@app.get("/api/events/{session_id}")
async def session_events(session_id: str, request: Request):
    session = get_session(session_id)
    async def stream():
        cursor = 0
        idle = 0
        while not await request.is_disconnected():
            with session_lock:
                if cursor < session["offset"]:
                    cursor = session["offset"]
                index = cursor - session["offset"]
                batch = session["events"][index:index+16]
                cursor += len(batch)
            if batch:
                for event in batch:
                    yield "data: " + json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n\n"
                idle = 0
            else:
                await asyncio.sleep(0.15)
                idle += 1
                if idle >= 100:
                    yield ": keepalive\n\n"
                    idle = 0
    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


class ControlRequest(BaseModel):
    session_id: str
    action: str = Field(pattern="^(pause|resume|step)$")


@app.post("/api/control")
def control_run(data: ControlRequest):
    session = get_session(data.session_id)
    control = active_run
    if control is None or control.session_id != data.session_id:
        raise HTTPException(409, "このセッションに実行中の生成はありません。")
    event = control.change(data.action)
    publish(session, event)
    return event


@app.get("/api/system")
def system():
    gpu = torch.cuda.is_available()
    return {"device": torch.cuda.get_device_name() if gpu else "CPU",
            "cuda": torch.version.cuda, "torch": torch.__version__, "loaded_model": loaded["id"],
            "busy": gpu_lock.locked(), "allocated_gb": round(torch.cuda.memory_allocated()/1024**3, 2) if gpu else 0,
            "backend": "Transformers / eager attention", "local_inference": True}


@app.post("/api/generate")
async def generate(data: GenerateRequest, request: Request):
    global active_run
    # Same-origin POST only. No remote model URLs or arbitrary code loading.
    origin = request.headers.get("origin")
    if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
        raise HTTPException(403, "異なるサイトからの生成リクエストは受け付けません。")
    if data.model not in {m["id"] for m in MODELS}:
        raise HTTPException(400, "モデルが見つかりません。")
    if data.messages[-1].role != "user":
        raise HTTPException(400, "最後のメッセージはユーザー入力にしてください。")
    if sum(len(m.content) for m in data.messages) > 16000:
        raise HTTPException(400, "会話が長すぎます。新しい会話を始めてください。")
    session = get_session(data.session_id) if data.session_id else None
    if not gpu_lock.acquire(blocking=False):
        raise HTTPException(409, "別の生成を処理しています。少し待って再実行してください。")
    events = queue.Queue(maxsize=4)
    stop = threading.Event()
    control = RunControl(data.session_id)
    active_run = control
    worker = threading.Thread(target=infer, args=(data, events, stop, control, session, uuid.uuid4().hex), daemon=True)
    worker.start()

    async def stream():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = events.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.02)
                    continue
                if event is None:
                    break
                yield json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n"
        finally:
            stop.set()

    return StreamingResponse(stream(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
