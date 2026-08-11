"""
JEEVidya — The Affect Engine (Singularity Plan, Part XVII)
══════════════════════════════════════════════════════════
ONE nervous system, not per-channel triggers. A single continuous
valence–arousal state per character that EVERY channel reads from, so
the whole body agrees about how the character feels — coherence is what
audiences unconsciously read as "real".

    state  ──second-order filter──►  (valence, arousal) per frame
      ▲                                     │
      │ script emotion tag (target)         ▼  ONE declarative table
      │ prosody push (RMS→arousal, F0)   mouth pull/press bias, brow,
      │ dialogue events (address/reveal) lids, blink rate, gaze hold,
      │ listener coupling (lag + gain)   head-tilt bias, gesture gain,
      └────────────────────────────────  breath rate/depth, voice ref

Micro-expression grammar: brief (120–200 ms) sub-threshold flickers on
affect transitions, scheduled deterministically from the trajectory.

QC gates (§XVII): affect-coherence audit (channel outputs must correlate
with the state trajectory) + state continuity (bounded derivative).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import settings
from engine.motion_grammar import SecondOrderFilter

# ═══════════════════════════════════════════
# Emotion tags → (valence, arousal) targets
# ═══════════════════════════════════════════
# Valence ∈ [−1, 1] (unpleasant→pleasant), Arousal ∈ [0, 1] (calm→activated)

EMOTION_TARGETS: Dict[str, Tuple[float, float]] = {
    "neutral":   (0.10, 0.30),
    "happy":     (0.75, 0.55),
    "excited":   (0.80, 0.90),
    "curious":   (0.35, 0.60),
    "surprised": (0.30, 0.85),
    "confused":  (-0.15, 0.55),
    "sad":       (-0.60, 0.20),
    "serious":   (0.00, 0.40),
    "angry":     (-0.70, 0.80),
    "proud":     (0.65, 0.50),
    "thinking":  (0.05, 0.45),
}

# ═══════════════════════════════════════════
# The channel-mapping matrix (§XVII) — ONE declarative table
# ═══════════════════════════════════════════
# Each channel bias = base + kv·valence + ka·arousal, clamped to (lo, hi).
#                          base    kv      ka      lo     hi

CHANNEL_MATRIX: Dict[str, Tuple[float, float, float, float, float]] = {
    "mouth_pull":        (0.00,  0.45,  0.05, -0.60, 0.80),
    "mouth_press":       (0.05, -0.10,  0.10,  0.00, 0.50),
    "brow_height":       (0.00,  0.10,  0.35, -0.40, 0.70),
    "brow_inner_raise":  (0.00, -0.30,  0.15,  0.00, 0.60),
    "lid_openness":      (0.75,  0.05,  0.25,  0.30, 1.00),
    "blink_rate_mult":   (1.00, -0.05,  0.60,  0.60, 2.20),
    "gaze_hold_mult":    (1.00,  0.10, -0.45,  0.45, 1.60),
    "head_tilt_bias":    (0.00,  0.06, -0.02, -0.12, 0.12),
    "gesture_gain":      (1.00,  0.10,  0.55,  0.60, 1.90),
    "breath_rate_mult":  (1.00, -0.05,  0.55,  0.70, 1.90),
    "breath_depth_mult": (1.00,  0.05, -0.30,  0.55, 1.30),
    "voice_emotion_intensity": (0.40, 0.15, 0.45, 0.10, 1.00),
}


@dataclass
class AffectSnapshot:
    valence: float
    arousal: float
    channels: Dict[str, float] = field(default_factory=dict)


def map_channels(valence: float, arousal: float) -> Dict[str, float]:
    """(v, a) → every channel bias, from the ONE table. An excited
    character blinks faster, breathes shallower, gestures bigger, holds
    gaze shorter and sounds brighter — all from one number pair."""
    out: Dict[str, float] = {}
    for name, (base, kv, ka, lo, hi) in CHANNEL_MATRIX.items():
        out[name] = float(np.clip(base + kv * valence + ka * arousal, lo, hi))
    return out


# ═══════════════════════════════════════════
# Micro-expression grammar (§XVII)
# ═══════════════════════════════════════════


@dataclass
class MicroExpression:
    """A brief sub-threshold flicker on an affect transition (the
    surprise onset *before* the smile). Scheduled deterministically."""
    start_ms: float
    duration_ms: float          # 120–200 ms
    channel: str                # channel it perturbs
    amplitude: float            # small: sub-threshold by definition

    def value_at(self, t_ms: float) -> float:
        u = (t_ms - self.start_ms) / self.duration_ms
        if not 0.0 <= u <= 1.0:
            return 0.0
        return self.amplitude * 0.5 * (1.0 - math.cos(2.0 * math.pi * u))


def schedule_micro_expressions(transitions: Sequence[Tuple[float, str, str]],
                               seed: int = 0) -> List[MicroExpression]:
    """From (t_ms, from_emotion, to_emotion) transitions, schedule the
    flicker grammar: a rising target (Δarousal > 0.2) fires a brow
    flash; a valence flip fires an inner-brow flicker. Deterministic
    per seed + transition list (Law 4)."""
    rng = np.random.default_rng(seed)
    out: List[MicroExpression] = []
    for t_ms, e_from, e_to in transitions:
        v0, a0 = EMOTION_TARGETS.get(e_from, EMOTION_TARGETS["neutral"])
        v1, a1 = EMOTION_TARGETS.get(e_to, EMOTION_TARGETS["neutral"])
        dur = float(rng.uniform(120.0, 200.0))
        if a1 - a0 > 0.2:
            out.append(MicroExpression(t_ms - dur * 0.5, dur,
                                       "brow_height",
                                       0.18 + 0.1 * float(rng.random())))
        if (v0 < 0) != (v1 < 0) and abs(v1 - v0) > 0.3:
            out.append(MicroExpression(t_ms - dur * 0.3, dur,
                                       "brow_inner_raise",
                                       0.12 + 0.08 * float(rng.random())))
    return out


# ═══════════════════════════════════════════
# The state — per character, per frame
# ═══════════════════════════════════════════


class AffectState:
    """Continuous (valence, arousal) driven through second-order filters
    (§XVI) — emotions *transition*, never snap. Update per frame with the
    script target, prosody push, and dialogue events."""

    # prosody→state gains (conservative on purpose)
    RMS_AROUSAL_GAIN = 0.35     # loudness pushes arousal
    F0_AROUSAL_GAIN = 0.20      # pitch excursion pushes arousal
    EVENT_DECAY_S = 0.9         # impulse events decay with this τ

    def __init__(self, character: str = "", seed: int = 0):
        self.character = character
        v0, a0 = EMOTION_TARGETS["neutral"]
        # slow, smooth emotional inertia: ~0.5 Hz response
        self._fv = SecondOrderFilter(f=0.55, zeta=1.0, x0=v0)
        self._fa = SecondOrderFilter(f=0.75, zeta=0.95, x0=a0)
        self.valence, self.arousal = v0, a0
        self._target = (v0, a0)
        self._event_v = 0.0
        self._event_a = 0.0
        self.micro: List[MicroExpression] = []
        self._seed = seed
        self.trajectory: List[Tuple[float, float]] = []

    # ── inputs ──

    def set_emotion(self, emotion: str) -> None:
        self._target = EMOTION_TARGETS.get(
            emotion, EMOTION_TARGETS["neutral"])

    def push_event(self, kind: str) -> None:
        """Dialogue events: being addressed, reveals, questions."""
        dv, da = {"addressed": (0.05, 0.15), "reveal": (0.20, 0.30),
                  "question": (0.05, 0.12)}.get(kind, (0.0, 0.1))
        self._event_v += dv
        self._event_a += da

    # ── per-frame update ──

    def step(self, rms_norm: float = 0.0, f0_excursion: float = 0.0,
             dt: Optional[float] = None) -> AffectSnapshot:
        """rms_norm ∈ [0,1] (speaker loudness this frame, 0 when
        listening); f0_excursion ∈ [0,1] normalized pitch deviation."""
        dt = dt if dt is not None else 1.0 / settings.FPS
        decay = math.exp(-dt / self.EVENT_DECAY_S)
        self._event_v *= decay
        self._event_a *= decay

        tv = self._target[0] + self._event_v
        ta = (self._target[1] + self._event_a
              + self.RMS_AROUSAL_GAIN * rms_norm
              + self.F0_AROUSAL_GAIN * f0_excursion)
        self.valence = float(np.clip(self._fv.step(tv, dt), -1.0, 1.0))
        self.arousal = float(np.clip(self._fa.step(ta, dt), 0.0, 1.0))
        self.trajectory.append((self.valence, self.arousal))

        return AffectSnapshot(self.valence, self.arousal,
                              map_channels(self.valence, self.arousal))

    def channels_with_micro(self, snapshot: AffectSnapshot,
                            t_ms: float) -> Dict[str, float]:
        ch = dict(snapshot.channels)
        for m in self.micro:
            ch[m.channel] = ch.get(m.channel, 0.0) + m.value_at(t_ms)
        return ch


# ═══════════════════════════════════════════
# Listener coupling (§XVII) — kills the dead-eyed listener
# ═══════════════════════════════════════════


class ListenerCoupling:
    """The non-speaking character's affect tracks the speaker's with
    300–500 ms lag and 0.4 gain → automatic reactive listening (nods on
    pitch accents, brow raise on reveals, natural check-away saccades).
    """

    def __init__(self, lag_ms: float = 400.0, gain: float = 0.4):
        self.lag_ms = lag_ms
        self.gain = gain
        self._buffer: List[Tuple[float, float, float]] = []  # (t, v, a)

    def observe(self, t_ms: float, speaker: AffectState) -> None:
        self._buffer.append((t_ms, speaker.valence, speaker.arousal))
        cutoff = t_ms - 4.0 * self.lag_ms
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.pop(0)

    def drive(self, t_ms: float, listener: AffectState) -> None:
        """Blend the lagged speaker state into the listener's target."""
        t_query = t_ms - self.lag_ms
        past = [(t, v, a) for (t, v, a) in self._buffer if t <= t_query]
        if not past:
            return
        _, v, a = past[-1]
        lv, la = listener._target
        listener._target = (lv * (1.0 - self.gain) + v * self.gain,
                            la * (1.0 - self.gain) + a * self.gain)


# ═══════════════════════════════════════════
# QC gates (§XVII)
# ═══════════════════════════════════════════


def verify_state_continuity(trajectory: Sequence[Tuple[float, float]],
                            fps: Optional[int] = None,
                            max_rate_per_s: float = 3.0) -> List[str]:
    """State derivative must stay bounded — emotions never snap."""
    f = float(fps or settings.FPS)
    tr = np.asarray(trajectory, dtype=np.float64)
    if len(tr) < 2:
        return []
    rate = np.max(np.abs(np.diff(tr, axis=0)), axis=0) * f
    out = []
    if rate[0] > max_rate_per_s:
        out.append(f"valence rate {rate[0]:.2f}/s > {max_rate_per_s}")
    if rate[1] > max_rate_per_s:
        out.append(f"arousal rate {rate[1]:.2f}/s > {max_rate_per_s}")
    return out


def coherence_audit(trajectory: Sequence[Tuple[float, float]],
                    channel_tracks: Dict[str, Sequence[float]],
                    min_abs_corr: float = 0.35) -> List[str]:
    """Per-channel correlation between rendered channel outputs and the
    state trajectory must exceed threshold *in the direction the matrix
    prescribes* — a smiling mouth with fear-brows fails the render.
    Channels whose matrix weight is ~0 for both axes are exempt."""
    tr = np.asarray(trajectory, dtype=np.float64)
    violations: List[str] = []
    if len(tr) < 8:
        return violations
    for name, track in channel_tracks.items():
        spec = CHANNEL_MATRIX.get(name)
        if spec is None:
            continue
        _, kv, ka, _, _ = spec
        x = np.asarray(track, dtype=np.float64)
        if len(x) != len(tr) or np.std(x) < 1e-9:
            continue
        # the state component this channel is supposed to follow
        driver = kv * tr[:, 0] + ka * tr[:, 1]
        if np.std(driver) < 1e-9:
            continue
        corr = float(np.corrcoef(driver, x)[0, 1])
        if corr < min_abs_corr:
            violations.append(
                f"coherence: channel '{name}' corr {corr:.2f} < "
                f"{min_abs_corr} against its affect driver")
    return violations


__all__ = ["AffectState", "AffectSnapshot", "ListenerCoupling",
           "MicroExpression", "schedule_micro_expressions",
           "map_channels", "EMOTION_TARGETS", "CHANNEL_MATRIX",
           "verify_state_continuity", "coherence_audit"]
