"""
Thin wrapper around the Anthropic SDK: retries on transient errors, and a
helper that asks for JSON-only output and parses it defensively (strips
markdown fences the model sometimes adds despite instructions).
"""
import json
import os
import time
from typing import Any, Dict, Optional

from anthropic import Anthropic, APIError, APIStatusError

_MODEL = "claude-sonnet-5"
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0

_client: Optional[Anthropic] = None


def get_client() -> Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Copy .env.example to .env, fill in "
                "your key, and `export ANTHROPIC_API_KEY=...` or use python-dotenv."
            )
        _client = Anthropic(api_key=api_key)
    return _client


def call(system: str, user: str, max_tokens: int = 1000, temperature: float = 0.0) -> str:
    """Single-turn call. Returns raw text of the first text block.
    Retries with exponential backoff on transient API errors."""
    client = get_client()
    last_err = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            for block in resp.content:
                if block.type == "text":
                    return block.text
            return ""
        except (APIError, APIStatusError) as e:
            last_err = e
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_BASE_DELAY * (2 ** attempt))
    raise RuntimeError(f"LLM call failed after {_MAX_RETRIES} attempts: {last_err}")


def call_json(system: str, user: str, max_tokens: int = 1000) -> Dict[str, Any]:
    """Calls the model expecting JSON-only output, strips markdown fences if
    present, and parses. Raises ValueError with the raw text on parse
    failure so callers can see what actually came back."""
    raw = call(system, user, max_tokens=max_tokens, temperature=0.0)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON from LLM response: {e}\nRaw response:\n{raw}")
