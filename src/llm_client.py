"""
Thin wrapper around a local Ollama server (open-source models, no API key,
no cost -- see decision log for why: this project was built without a paid
LLM budget). Retries on transient errors, and a helper that asks for
JSON-only output using Ollama's structured-output mode and parses it
defensively.

Requires:
    - Ollama installed and running locally (`ollama serve`)
    - A model pulled, e.g. `ollama pull llama3.2:3b`

Model + host are configurable via env vars so this runs on any machine
without code changes:
    OLLAMA_MODEL (default: llama3.1:8b)
    OLLAMA_HOST  (default: http://localhost:11434)
"""
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import requests
from dotenv import load_dotenv

# Load .env from repo root regardless of current working directory.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0
_TIMEOUT_SECONDS = 120


def _post_chat(system: str, user: str, max_tokens: int, temperature: float,
               json_mode: bool = False) -> str:
    payload = {
        "model": _MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if json_mode:
        payload["format"] = "json"  # Ollama's native structured-output mode

    resp = requests.post(f"{_HOST}/api/chat", json=payload, timeout=_TIMEOUT_SECONDS)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


def call(system: str, user: str, max_tokens: int = 1000, temperature: float = 0.0) -> str:
    """Single-turn call. Returns raw text response. Retries with
    exponential backoff on transient connection errors (e.g. Ollama still
    loading the model into memory on first call)."""
    last_err = None
    for attempt in range(_MAX_RETRIES):
        try:
            return _post_chat(system, user, max_tokens, temperature, json_mode=False)
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            last_err = e
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_BASE_DELAY * (2 ** attempt))
    raise RuntimeError(
        f"Ollama call failed after {_MAX_RETRIES} attempts: {last_err}\n"
        f"Is `ollama serve` running? Is model '{_MODEL}' pulled "
        f"(`ollama pull {_MODEL}`)?"
    )


def call_json(system: str, user: str, max_tokens: int = 1000) -> Dict[str, Any]:
    """Calls the model in JSON mode and parses the result. Ollama's
    format='json' mode guarantees syntactically valid JSON but NOT that it
    matches your intended schema -- the system prompt still has to spell
    out the exact keys expected. Strips markdown fences defensively anyway
    in case a given model ignores json_mode."""
    last_err = None
    for attempt in range(_MAX_RETRIES):
        try:
            raw = _post_chat(system, user, max_tokens, temperature=0.0, json_mode=True)
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip()
            return json.loads(cleaned)
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            last_err = e
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_BASE_DELAY * (2 ** attempt))
        except json.JSONDecodeError as e:
            last_err = ValueError(f"Could not parse JSON from Ollama response: {e}\nRaw:\n{raw}")
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_BASE_DELAY * (2 ** attempt))
    raise RuntimeError(f"Ollama JSON call failed after {_MAX_RETRIES} attempts: {last_err}")
