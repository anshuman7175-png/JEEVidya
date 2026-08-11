"""
JEEVidya — Audio-Source-Agnostic Contract (Terminal Plan, Part X §10.1)
═══════════════════════════════════════════════════════════════════════
ONE interface for "text of a turn → a wav on disk". Three interchangeable
backends, all of which lip-sync identically because Part VII derives every
timing from the *returned waveform*, never from TTS metadata:

    EdgeTTSSource     — current behavior; zero-setup default & fallback
                        (a *bridge* per Law 3: named local successor below).
    ClonedVoiceSource — your own voice, synthesized per line via
                        pipeline/voice_clone.py (IndexTTS-2 / Chatterbox).
    RecordedSource    — folder of hand-recorded wavs keyed by turn ID;
                        the manual escape hatch.

Doctrine hooks:
  • Law 1: exactly one place decides which voice backend runs
    (`get_audio_source`), so a second TTS call-path cannot exist.
  • Law 4: every backend is content-cached and deterministic for
    identical inputs.
  • Law 5: ClonedVoiceSource enforces the speaker-identity and
    pronunciation gates (§10.4) — a line can never drift off-voice.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from config import settings

# ═══════════════════════════════════════════
# Contract
# ═══════════════════════════════════════════


@dataclass
class AudioResult:
    """What every backend returns. Timings downstream ALWAYS come from
    engine/align.py run on `wav_path` — never from anything in `meta`."""
    wav_path: str
    sample_rate: int
    duration_ms: int
    vtt_path: Optional[str] = None      # legacy captions only, never timing
    backend: str = "unknown"
    cache_key: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class AudioSource(ABC):
    """The single voice contract (Part X §10.1)."""

    name: str = "abstract"

    @abstractmethod
    def render(self, turn_id: int, text: str, speaker: str,
               output_dir: str, emotion: str = "neutral",
               voice: Optional[str] = None) -> AudioResult:
        """Synthesize (or fetch) one dialogue turn. MUST be deterministic
        for identical (text, speaker, emotion, backend-config) and MUST
        return a real decodable file whose duration was *measured*."""

    def preview(self, turns, output_dir: str) -> list:
        """--voice-preview: render the whole script's audio only, for a
        fast listen before committing to a video render (§10.4)."""
        results = []
        for i, t in enumerate(turns):
            text = (t.get("text") or "").strip()
            if not text:
                continue
            results.append(self.render(
                i, text, t.get("speaker", "boy"), output_dir,
                emotion=t.get("emotion", "neutral")))
        return results


# ═══════════════════════════════════════════
# Backend 1 — edge-tts (the bridge)
# ═══════════════════════════════════════════


class EdgeTTSSource(AudioSource):
    """Wraps the existing VoiceEngine. Zero setup, network-bound, kept as
    default & fallback per Law 3 (named successor: IndexTTS-2)."""

    name = "edge-tts"

    def __init__(self):
        from pipeline.voice import VoiceEngine  # lazy: keeps import cheap
        self._engine = VoiceEngine()

    def render(self, turn_id: int, text: str, speaker: str,
               output_dir: str, emotion: str = "neutral",
               voice: Optional[str] = None) -> AudioResult:
        out = self._engine.generate_turn_audio(
            turn_id, text, speaker, output_dir, voice=voice)
        if not out.get("audio"):
            raise RuntimeError(
                f"EdgeTTSSource produced no audio for turn {turn_id}")
        return AudioResult(
            wav_path=out["audio"], sample_rate=24_000,
            duration_ms=int(out.get("duration_ms", 0)),
            vtt_path=out.get("vtt"), backend=self.name,
            cache_key=out.get("key"),
            meta={"emotion_requested": emotion, "emotion_applied": False})


# ═══════════════════════════════════════════
# Backend 2 — cloned voice (the recommended answer)
# ═══════════════════════════════════════════


class ClonedVoiceSource(AudioSource):
    """Your voice, recorded once (§10.2), synthesized per line with the
    per-turn `emotion` tag as the automatic style selector. All heavy
    lifting + the identity/pronunciation gates live in voice_clone.py."""

    name = "cloned"

    def __init__(self, engine: Optional[str] = None,
                 voices_dir: Optional[str] = None):
        from pipeline.voice_clone import VoiceCloneEngine  # lazy: torch
        self._engine = VoiceCloneEngine(engine=engine, voices_dir=voices_dir)

    def render(self, turn_id: int, text: str, speaker: str,
               output_dir: str, emotion: str = "neutral",
               voice: Optional[str] = None) -> AudioResult:
        out = self._engine.synthesize(text, speaker, emotion, output_dir,
                                      turn_id=turn_id)
        return AudioResult(
            wav_path=out["wav_path"], sample_rate=out["sample_rate"],
            duration_ms=out["duration_ms"], backend=self.name,
            cache_key=out["cache_key"],
            meta={"engine": out["engine"], "seed": out["seed"],
                  "identity_cos": out["identity_cos"],
                  "align_confidence": out["align_confidence"],
                  "emotion_requested": emotion, "emotion_applied": True})


# ═══════════════════════════════════════════
# Backend 3 — recorded wavs (the escape hatch)
# ═══════════════════════════════════════════


class RecordedSource(AudioSource):
    """Folder of hand-recorded wavs keyed by turn ID:
        <recordings_dir>/<speaker>/turn_<NNN>.wav
    Lip sync still works because timings derive from the wav itself."""

    name = "recorded"

    def __init__(self, recordings_dir: str):
        self.recordings_dir = recordings_dir
        if not os.path.isdir(recordings_dir):
            raise FileNotFoundError(
                f"RecordedSource: directory not found: {recordings_dir}")

    def render(self, turn_id: int, text: str, speaker: str,
               output_dir: str, emotion: str = "neutral",
               voice: Optional[str] = None) -> AudioResult:
        candidates = [
            os.path.join(self.recordings_dir, speaker,
                         f"turn_{turn_id:03d}.wav"),
            os.path.join(self.recordings_dir, f"turn_{turn_id:03d}.wav"),
        ]
        wav = next((c for c in candidates if os.path.exists(c)), None)
        if wav is None:
            raise FileNotFoundError(
                f"RecordedSource: no recording for turn {turn_id} "
                f"(looked in {candidates}). Record it or switch backend.")
        from engine.align import load_wav_mono
        x, sr = load_wav_mono(wav)
        return AudioResult(
            wav_path=wav, sample_rate=sr,
            duration_ms=int(1000.0 * len(x) / max(1, sr)),
            backend=self.name, meta={"emotion_requested": emotion})


# ═══════════════════════════════════════════
# The single resolver (Law 1)
# ═══════════════════════════════════════════

_BACKENDS = {"edge-tts": EdgeTTSSource, "cloned": ClonedVoiceSource,
             "recorded": RecordedSource}


def get_audio_source(name: Optional[str] = None, **kwargs) -> AudioSource:
    """THE way to obtain a voice backend. Resolution order:
    explicit arg → $JV_AUDIO_SOURCE → 'edge-tts'. A cloned-voice failure
    at *construction* time (missing models) falls back to edge-tts loudly
    — the face pipeline never breaks because the TTS stack didn't build."""
    name = name or os.environ.get("JV_AUDIO_SOURCE", "edge-tts")
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown audio source '{name}'. Options: {sorted(_BACKENDS)}")
    try:
        return _BACKENDS[name](**kwargs)
    except Exception as e:
        if name == "edge-tts":
            raise
        print(f"  [AudioSource] '{name}' unavailable ({e}); "
              f"falling back to edge-tts (Law 3 bridge).")
        return EdgeTTSSource()


__all__ = ["AudioSource", "AudioResult", "EdgeTTSSource",
           "ClonedVoiceSource", "RecordedSource", "get_audio_source"]
