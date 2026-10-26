"""Thin wrapper around Groq's OpenAI-compatible REST API (free tier).

Models are auto-detected from your key's /models list, so a retired model name
won't break the app. Override any of them with env vars GROQ_TEXT_MODEL,
GROQ_VISION_MODEL, GROQ_STT_MODEL.
"""
import os
import re
import time
import requests

BASE = "https://api.groq.com/openai/v1"

# Preference order: first one your key can access is used.
PREFS = {
    "text": ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "qwen/qwen3.8-27b",
             "openai/gpt-oss-20b", "llama-3.1-8b-instant"],
    "vision": ["qwen/qwen3.8-27b", "meta-llama/llama-4-scout-17b-16e-instruct",
               "meta-llama/llama-4-maverick-17b-128e-instruct", "meta-models/muse-spark-open"],
    "stt": ["whisper-large-v3-turbo", "whisper-large-v3"],
}
ENV = {"text": "GROQ_TEXT_MODEL", "vision": "GROQ_VISION_MODEL", "stt": "GROQ_STT_MODEL"}

_available = None
_resolved = {}


class GroqError(Exception):
    pass


def _headers():
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not key:
        raise GroqError("GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys")
    return {"Authorization": f"Bearer {key}"}


def available_models(refresh=False):
    global _available
    if _available is None or refresh:
        r = requests.get(f"{BASE}/models", headers=_headers(), timeout=30)
        if not r.ok:
            raise GroqError(f"Could not list models ({r.status_code}): {r.text[:300]}")
        _available = [m["id"] for m in r.json().get("data", [])]
    return _available


def model_for(kind, refresh=False):
    if os.getenv(ENV[kind]):
        return os.getenv(ENV[kind]).strip()
    if kind in _resolved and not refresh:
        return _resolved[kind]
    avail = available_models(refresh)
    for m in PREFS[kind]:
        if m in avail:
            _resolved[kind] = m
            return m
    raise GroqError(
        f"No supported {kind} model found for your key. Available: {', '.join(avail) or 'none'}. "
        f"Set {ENV[kind]} to one of them."
    )


def _post(path, retries=3, **kwargs):
    """POST with backoff for rate limits (429)."""
    for attempt in range(retries):
        r = requests.post(f"{BASE}{path}", headers=_headers(), timeout=90, **kwargs)
        if r.status_code == 429 and attempt < retries - 1:
            time.sleep(min(float(r.headers.get("retry-after", 2 * (attempt + 1))), 15))
            continue
        if not r.ok:
            raise GroqError(f"Groq API error {r.status_code}: {r.text[:400]}")
        return r.json()
    raise GroqError("Rate limited by Groq. Wait a few seconds and retry.")


def _clean(text):
    # some reasoning models emit <think>...</think>
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def chat(messages, model=None, kind="text", tools=None, temperature=0.4, max_tokens=1500):
    def call(mdl):
        body = {"model": mdl, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        return _post("/chat/completions", json=body)["choices"][0]["message"]

    try:
        msg = call(model or model_for(kind))
    except GroqError as e:
        if "model_not_found" in str(e) and not model and not os.getenv(ENV[kind]):
            _resolved.pop(kind, None)
            msg = call(model_for(kind, refresh=True))
        else:
            raise
    msg["content"] = _clean(msg.get("content"))
    return msg


def describe_image(data_url, question="Describe this image in detail."):
    """Multi-modal input: image -> text using a Groq vision model."""
    msg = [{"role": "user", "content": [
        {"type": "text", "text": question},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]}]
    return chat(msg, kind="vision", max_tokens=800)["content"]


def transcribe(filename, audio_bytes, mimetype="audio/webm"):
    """Voice input: audio -> text with Whisper on Groq."""
    files = {"file": (filename, audio_bytes, mimetype)}
    data = {"model": model_for("stt"), "response_format": "json"}
    return _post("/audio/transcriptions", files=files, data=data)["text"]
