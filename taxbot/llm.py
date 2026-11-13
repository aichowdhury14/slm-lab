"""Minimal client for the local Ollama server.

Standard library only, so you can see exactly what goes over the wire:
every call is one HTTP POST with a JSON body to http://127.0.0.1:11434.
API reference: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Iterator

HOST = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
# 1.7b fits this PC's free memory; qwen3:4b answers better but needs ~3.5 GB free (see README).
CHAT_MODEL = os.environ.get("TAXBOT_MODEL", "qwen3:1.7b")
EMBED_MODEL = os.environ.get("TAXBOT_EMBED_MODEL", "bge-m3")
KEEP_ALIVE = "10m"  # keep the model in RAM between questions; loading it is the slowest step
# Context window in tokens. The KV cache grows with it (qwen3:4b: ~150 KB per token, so 4096 ≈ 600 MB).
# TaxBot prompts stay under ~2,000 tokens; raise this only if you add longer documents.
CONTEXT = int(os.environ.get("TAXBOT_CTX", "4096"))


def _post(path: str, payload: dict, stream: bool = False):
    req = urllib.request.Request(
        HOST + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=900)
    except urllib.error.HTTPError as e:  # surface Ollama's own message, e.g. "model not found"
        raise RuntimeError(f"Ollama {e.code}: {e.read().decode('utf-8', 'replace')}") from None
    if not stream:
        with resp:
            return json.loads(resp.read())

    def lines():
        with resp:
            for line in resp:  # Ollama streams one JSON object per line
                if line.strip():
                    yield json.loads(line)

    return lines()


# Qwen3's recommended sampling for non-thinking mode. Near-greedy decoding (temperature ~0) makes
# small models fall into endless repetition loops; presence_penalty discourages repeating tokens.
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}


def _chat_payload(messages, model, temperature, schema=None, max_tokens=None, stream=False) -> dict:
    options = {**SAMPLING, "num_ctx": CONTEXT, "num_predict": max_tokens}  # num_predict caps the output
    if temperature is not None:
        options["temperature"] = temperature
    payload = {
        "model": model or CHAT_MODEL,
        "messages": messages,
        "stream": stream,
        "think": False,  # Qwen3 can "think" first; on a CPU that costs minutes, so we turn it off
        "keep_alive": KEEP_ALIVE,
        "options": options,
    }
    if schema:
        payload["format"] = schema  # constrained decoding: the output is forced to match this JSON schema
    return payload


def stats(resp: dict) -> dict:
    """Speed numbers Ollama reports at the end of a generation (durations are in nanoseconds)."""
    gen_s = resp.get("eval_duration", 0) / 1e9
    prompt_s = resp.get("prompt_eval_duration", 0) / 1e9
    return {
        "model": resp.get("model"),
        "load_seconds": round(resp.get("load_duration", 0) / 1e9, 2),  # >1s means the model was (re)loaded
        "prompt_tokens": resp.get("prompt_eval_count", 0),
        "output_tokens": resp.get("eval_count", 0),
        "prompt_seconds": round(prompt_s, 2),
        "output_seconds": round(gen_s, 2),
        "tokens_per_second": round(resp.get("eval_count", 0) / gen_s, 1) if gen_s else None,
        "total_seconds": round(resp.get("total_duration", 0) / 1e9, 2),
    }


def chat(messages: list[dict], model: str | None = None, schema: dict | None = None,
         temperature: float | None = None, max_tokens: int = 500) -> tuple[str, dict]:
    """One complete reply. Returns (text, stats)."""
    out = _post("/api/chat", _chat_payload(messages, model, temperature, schema, max_tokens))
    return out["message"]["content"], stats(out)


def chat_stream(messages: list[dict], model: str | None = None,
                temperature: float | None = None, max_tokens: int = 500) -> Iterator[dict]:
    """Yields {"text": piece} while generating, then a final {"stats": {...}}."""
    payload = _chat_payload(messages, model, temperature, max_tokens=max_tokens, stream=True)
    for chunk in _post("/api/chat", payload, stream=True):
        if chunk.get("done"):
            yield {"stats": stats(chunk)}
        else:
            yield {"text": chunk["message"]["content"]}


def embed(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Turn texts into vectors; texts with similar meaning get nearby vectors."""
    out = _post("/api/embed", {"model": model or EMBED_MODEL, "input": texts, "keep_alive": KEEP_ALIVE})
    return out["embeddings"]


def list_models() -> list[str]:
    req = urllib.request.urlopen(HOST + "/api/tags", timeout=10)
    with req:
        return sorted(m["name"] for m in json.loads(req.read())["models"])
