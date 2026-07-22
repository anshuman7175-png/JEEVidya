"""
JEEVidya V5 — LLM Substrate (Tier 3)
════════════════════════════════════
One thin, dependency-free client under both agents:

  • Gemini REST (generativelanguage.googleapis.com) — text AND vision,
    structured JSON output, no SDK required.
  • Free-tier quota ledger persisted in .cache/llm_quota.json — the
    agents THROTTLE THEMSELVES before Google does, and requests queue
    locally with backoff instead of dying.
  • Optional Ollama fallback (localhost:11434) for fully-offline drafts.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from config import settings

GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
              "{model}:generateContent?key={key}")
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.environ.get("JV_OLLAMA_MODEL", "llama3.1")

# Conservative self-imposed ceilings (real free tier is higher; headroom
# means overnight batches never brown out mid-run).
DAILY_BUDGET = 900
RPM_BUDGET = 8

_QUOTA_PATH = os.path.join(settings.PROJECT_ROOT, ".cache", "llm_quota.json")


class QuotaExhausted(RuntimeError):
    pass


class QuotaTracker:
    """Tiny persistent day/minute request ledger."""

    def __init__(self, path: str = _QUOTA_PATH):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def _load(self) -> Dict[str, Any]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _save(self, data: Dict[str, Any]) -> None:
        tmp = self.path + ".part"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, self.path)

    def acquire(self, wait: bool = True) -> None:
        """Block (politely) until a request slot is available."""
        while True:
            data = self._load()
            day = time.strftime("%Y-%m-%d")
            if data.get("day") != day:
                data = {"day": day, "count": 0, "minute": 0, "minute_count": 0}
            if data.get("count", 0) >= DAILY_BUDGET:
                raise QuotaExhausted(
                    f"Self-imposed daily Gemini budget ({DAILY_BUDGET}) spent. "
                    "Resumes tomorrow, or set JV_OLLAMA_MODEL for offline drafting.")
            minute = int(time.time() // 60)
            if data.get("minute") != minute:
                data["minute"], data["minute_count"] = minute, 0
            if data.get("minute_count", 0) < RPM_BUDGET:
                data["count"] = data.get("count", 0) + 1
                data["minute_count"] += 1
                self._save(data)
                return
            if not wait:
                raise QuotaExhausted("Per-minute budget hit (no-wait mode)")
            time.sleep(61 - time.time() % 60)      # sleep to next minute

    def remaining_today(self) -> int:
        data = self._load()
        if data.get("day") != time.strftime("%Y-%m-%d"):
            return DAILY_BUDGET
        return max(0, DAILY_BUDGET - data.get("count", 0))


class LLM:
    """Gemini-first, Ollama-fallback text/vision client."""

    def __init__(self, api_key: Optional[str] = None,
                 model: str = GEMINI_MODEL):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model
        self.quota = QuotaTracker()

    @property
    def online(self) -> bool:
        return bool(self.api_key)

    # ─── Core call ─────────────────────────────────────────

    def generate(self, prompt: str,
                 system: Optional[str] = None,
                 images_jpeg: Optional[List[bytes]] = None,
                 json_output: bool = False,
                 temperature: float = 0.7,
                 max_retries: int = 3) -> str:
        """Text (and optional vision) generation. Returns raw model text."""
        if self.online:
            last: Exception = RuntimeError("no attempts made")
            for attempt in range(max_retries):
                try:
                    self.quota.acquire()
                    return self._gemini(prompt, system, images_jpeg,
                                        json_output, temperature)
                except QuotaExhausted:
                    break                       # fall through to Ollama
                except urllib.error.HTTPError as e:
                    last = e
                    if e.code in (429, 500, 503):
                        time.sleep(2 ** attempt * 3)
                        continue
                    raise
                except (urllib.error.URLError, TimeoutError) as e:
                    last = e
                    time.sleep(2 ** attempt * 2)
            else:
                raise RuntimeError(f"Gemini failed after retries: {last}")
        # Offline path
        if images_jpeg:
            raise RuntimeError("Vision requires GEMINI_API_KEY (Ollama "
                               "fallback is text-only here)")
        return self._ollama(prompt, system, json_output, temperature)

    def generate_json(self, prompt: str, system: Optional[str] = None,
                      images_jpeg: Optional[List[bytes]] = None,
                      temperature: float = 0.6) -> Any:
        """Generation that must come back as parsed JSON (fence-tolerant)."""
        text = self.generate(prompt, system=system, images_jpeg=images_jpeg,
                             json_output=True, temperature=temperature)
        return parse_json_loose(text)

    # ─── Backends ──────────────────────────────────────────

    def _gemini(self, prompt, system, images_jpeg, json_output,
                temperature) -> str:
        parts: List[Dict[str, Any]] = []
        for img in images_jpeg or []:
            parts.append({"inline_data": {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img).decode("ascii")}})
        parts.append({"text": prompt})

        body: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": temperature},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if json_output:
            body["generationConfig"]["responseMimeType"] = "application/json"

        url = GEMINI_URL.format(model=self.model, key=self.api_key)
        req = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Gemini returned no text: {data}") from e

    def _ollama(self, prompt, system, json_output, temperature) -> str:
        body = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                "options": {"temperature": temperature}}
        if system:
            body["system"] = system
        if json_output:
            body["format"] = "json"
        req = urllib.request.Request(
            OLLAMA_URL, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read().decode("utf-8"))["response"]
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(
                "No GEMINI_API_KEY and no local Ollama reachable at "
                f"{OLLAMA_URL} — the agents need one of the two.") from e


def parse_json_loose(text: str) -> Any:
    """Parse JSON that may arrive wrapped in ``` fences or prose."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
    try:
        return json.loads(t)
    except ValueError:
        start = min((i for i in (t.find("{"), t.find("[")) if i >= 0),
                    default=-1)
        if start < 0:
            raise
        closer = "}" if t[start] == "{" else "]"
        end = t.rfind(closer)
        return json.loads(t[start:end + 1])
