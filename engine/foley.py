"""
JEEVidya — Multimodal Foley Coherence (Singularity Plan, Part XVIII)
════════════════════════════════════════════════════════════════════
Sound and image must agree at the millisecond level — audiences forgive
a flat image long before they forgive incoherent audio.

  BREATH FOLEY   Part VII.5 schedules visible inhales before phrases;
                 this module synthesizes the matching breath SOUND
                 (filtered noise burst shaped by breath depth from the
                 affect state) at the exact aligned instant, −38 dB
                 under dialogue. Seeing AND hearing the inhale is a
                 subliminal "alive" signal almost no automated channel
                 has.
  CLOTH FOLEY    Gesture onsets above an amplitude threshold trigger a
                 soft cloth swish from a seeded procedural synthesizer,
                 panned to the character's screen position. Sub-audible
                 consciously; missed when absent.

Everything is seeded and deterministic (Law 4): identical inputs give
bit-identical foley. The A/V co-occurrence QC gate lives in
tests/test_multimodal.py — every scheduled visible inhale must have its
sound within ±1 frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SR = 44100

# Breath foley sits far under dialogue: consciously inaudible,
# subliminally present.
BREATH_GAIN_DB = -38.0
CLOTH_GAIN_DB = -34.0
MIN_PAUSE_FOR_BREATH_MS = 400      # §VII.5: inter-phrase silences ≥ 400 ms
BREATH_LEAD_MS = 250               # inhale starts this long before speech
GESTURE_CLOTH_MIN_AMP = 0.5        # gesture amplitude floor for a swish


@dataclass(frozen=True)
class FoleyEvent:
    """One scheduled foley sound on the global millisecond axis."""
    t_ms: int
    kind: str                      # "breath" | "cloth"
    depth: float = 1.0             # breath depth / gesture amplitude 0..~1.5
    pan: float = 0.0               # −1 (left) .. +1 (right)


# ═══════════════════════════════════════════
# SYNTHESIS (seeded, deterministic)
# ═══════════════════════════════════════════

def _bandpass_fft(x: np.ndarray, lo_hz: float, hi_hz: float) -> np.ndarray:
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    gain = (1.0 / (1.0 + (lo_hz / np.maximum(f, 1.0)) ** 4)) \
        * (1.0 / (1.0 + (np.maximum(f, 1.0) / hi_hz) ** 4))
    return np.fft.irfft(spec * gain, n=len(x)).astype(np.float32)


def synth_breath(depth: float = 1.0, seconds: float = 0.42,
                 seed: int = 71) -> np.ndarray:
    """An inhale: band-limited noise with a rising amplitude arc and a
    slight upward spectral tilt over time (air accelerating through the
    nose). Depth scales duration, loudness, and bandwidth together so a
    deep on-screen breath sounds deeper, not merely louder."""
    depth = float(np.clip(depth, 0.3, 1.6))
    n = int(SR * seconds * (0.75 + 0.35 * depth))
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n).astype(np.float32)

    # Two-stage band: starts nasal (600–1800 Hz), opens to 400–3200 Hz.
    a = _bandpass_fft(noise, 600.0, 1200.0 + 900.0 * depth)
    b = _bandpass_fft(noise, 400.0, 2200.0 + 1400.0 * depth)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    x = a * (1.0 - t) + b * t

    # Rising arc with a soft comma at the end (the pre-speech catch).
    env = np.sin(np.pi * np.clip(t * 1.12, 0, 1)) ** 1.5
    env *= 1.0 - 0.35 * np.exp(-((t - 0.93) ** 2) / 0.0012)
    x *= env
    peak = float(np.abs(x).max()) + 1e-9
    return (x / peak * (0.6 + 0.4 * depth)).astype(np.float32)


def synth_cloth(amplitude: float = 1.0, seconds: float = 0.28,
                seed: int = 72) -> np.ndarray:
    """A cloth swish: high-passed noise with a fast asymmetric envelope
    and gentle spectral motion — reads as fabric, never as static."""
    amplitude = float(np.clip(amplitude, 0.3, 1.5))
    n = int(SR * seconds)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n).astype(np.float32)
    x = _bandpass_fft(noise, 1500.0, 6500.0 + 2500.0 * amplitude)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    env = (t ** 0.6) * np.exp(-t * 6.5)          # fast rise, natural decay
    env /= env.max() + 1e-9
    x *= env
    peak = float(np.abs(x).max()) + 1e-9
    return (x / peak * (0.5 + 0.5 * amplitude)).astype(np.float32)


# ═══════════════════════════════════════════
# SCHEDULING (from the timeline — the aligner's clock)
# ═══════════════════════════════════════════

def plan_breaths(spans: Sequence[Any],
                 breath_depth_of: Optional[Any] = None) -> List[FoleyEvent]:
    """One inhale before every phrase that follows ≥400 ms of silence
    (and before the very first phrase). `spans` are timeline TurnSpans;
    `breath_depth_of(turn) -> float` optionally maps the affect state's
    breath-depth multiplier onto each event."""
    events: List[FoleyEvent] = []
    prev_end = None
    for span in spans:
        gap = span.start_ms - prev_end if prev_end is not None else None
        if gap is None or gap >= MIN_PAUSE_FOR_BREATH_MS:
            depth = 1.0
            if breath_depth_of is not None:
                try:
                    depth = float(breath_depth_of(span.turn))
                except Exception:   # noqa: BLE001 — depth is cosmetic
                    depth = 1.0
            t = max(0, span.start_ms - BREATH_LEAD_MS)
            events.append(FoleyEvent(t_ms=t, kind="breath", depth=depth))
        prev_end = span.end_ms
    return events


def plan_cloth(gesture_events: Sequence[Tuple[int, float, float]]
               ) -> List[FoleyEvent]:
    """`gesture_events` = (onset_ms, amplitude, screen_x_norm −1..1).
    Only gestures above the amplitude floor earn a swish."""
    return [FoleyEvent(t_ms=int(t), kind="cloth", depth=amp, pan=float(x))
            for t, amp, x in gesture_events
            if amp >= GESTURE_CLOTH_MIN_AMP]


# ═══════════════════════════════════════════
# RENDER (place every event into a stereo pair)
# ═══════════════════════════════════════════

def render_foley(events: Sequence[FoleyEvent], total_ms: int
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic stereo render: constant-power panning, per-kind
    gain staging. Each event's seed derives from its timestamp so two
    breaths never sound like copy-paste, yet re-renders are identical."""
    n = int(total_ms * SR / 1000)
    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)
    gain = {"breath": 10 ** (BREATH_GAIN_DB / 20),
            "cloth": 10 ** (CLOTH_GAIN_DB / 20)}
    for ev in events:
        seed = (ev.t_ms * 2654435761) & 0x7FFFFFFF   # deterministic hash
        if ev.kind == "breath":
            x = synth_breath(ev.depth, seed=seed)
        elif ev.kind == "cloth":
            x = synth_cloth(ev.depth, seed=seed)
        else:
            continue
        start = int(ev.t_ms * SR / 1000)
        end = min(n, start + len(x))
        if end <= start:
            continue
        theta = (np.clip(ev.pan, -1, 1) + 1) * math.pi / 4
        chunk = x[:end - start] * gain[ev.kind]
        left[start:end] += chunk * math.cos(theta)
        right[start:end] += chunk * math.sin(theta)
    return left, right


def breath_schedule_manifest(events: Sequence[FoleyEvent]
                             ) -> List[Dict[str, Any]]:
    """Serializable record of every breath instant — the QC gate joins
    this against the visible-inhale schedule (±1 frame co-occurrence)."""
    return [{"t_ms": ev.t_ms, "kind": ev.kind, "depth": ev.depth}
            for ev in events]
