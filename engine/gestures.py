"""
JEEVidya V5 — Procedural Gesture Library (Tier 1)
═════════════════════════════════════════════════
10 gestures encoded as bone-channel keyframe curves with easing —
zero new art assets, ever. The Director Agent (Tier 3) or keyword
triggers schedule them; the Bone Engine renders them.

Channels (all additive on top of the base pose):
  lean       spine bend, degrees (+ = screen right)
  head_tilt  head roll, degrees
  head_yaw   fake 3D turn, −1..1
  head_nod   pitch, −1 (up) .. 1 (down)
  bounce     whole-body y offset, px (− = up)
  sway       whole-body x offset, px
  squash     scale-y delta (0 = none; −0.1 = squashed)
  brow       −1..1 (+ raised)

Keyword triggers fire at the EXACT edge-tts VTT word timestamp —
"dekho" points, "arre" recoils, "nahi" shakes, at the millisecond
the word is spoken.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from engine.easing import (
    ease_in_out_sine, ease_out_back, ease_out_cubic, ease_out_elastic,
    ease_in_out_quad, clamp,
)

_EASE = {
    "sine": ease_in_out_sine, "back": ease_out_back, "cubic": ease_out_cubic,
    "elastic": ease_out_elastic, "quad": ease_in_out_quad,
}

CHANNELS = ("lean", "head_tilt", "head_yaw", "head_nod",
            "bounce", "sway", "squash", "brow")


@dataclass
class Key:
    t: float                 # normalized 0..1 within the gesture
    value: float
    ease: str = "cubic"      # easing INTO this key


def _sample_curve(keys: List[Key], t: float) -> float:
    """Piecewise-eased interpolation through keyframes."""
    if not keys:
        return 0.0
    if t <= keys[0].t:
        return keys[0].value
    if t >= keys[-1].t:
        return keys[-1].value
    for i in range(1, len(keys)):
        if t <= keys[i].t:
            a, b = keys[i - 1], keys[i]
            span = max(1e-6, b.t - a.t)
            local = clamp((t - a.t) / span)
            return a.value + (b.value - a.value) * _EASE[b.ease](local)
    return keys[-1].value


@dataclass
class Gesture:
    name: str
    duration_ms: int
    channels: Dict[str, List[Key]] = field(default_factory=dict)
    body_pose: str = ""   # pose library image to switch to (empty = keep current)

    def sample(self, t_norm: float) -> Dict[str, float]:
        return {ch: _sample_curve(keys, t_norm)
                for ch, keys in self.channels.items()}


def _g(name: str, ms: int, **chans: List[Key]) -> Gesture:
    return Gesture(name=name, duration_ms=ms, channels=dict(chans))


# ═══════════════════════════════════════════
# THE 10 GESTURES
# ═══════════════════════════════════════════

GESTURES: Dict[str, Gesture] = {g.name: g for g in [

    # 1 · nod — emphatic agreement (double-dip)
    _g("nod", 700,
       head_nod=[Key(0, 0), Key(0.22, 0.9, "back"), Key(0.45, -0.15),
                 Key(0.68, 0.55), Key(1, 0, "sine")]),

    # 2 · shake — disagreement (yaw wobble)
    _g("shake", 800,
       head_yaw=[Key(0, 0), Key(0.2, -0.6, "sine"), Key(0.45, 0.55, "sine"),
                 Key(0.7, -0.35, "sine"), Key(1, 0, "sine")]),

    # 3 · point — "dekho!" whole body drives toward the content
    _g("point", 900,
       lean=[Key(0, 0), Key(0.25, 7, "back"), Key(0.75, 6), Key(1, 0, "sine")],
       head_yaw=[Key(0, 0), Key(0.25, 0.5, "back"), Key(0.75, 0.45), Key(1, 0)],
       head_tilt=[Key(0, 0), Key(0.3, -4, "back"), Key(1, 0, "sine")],
       brow=[Key(0, 0), Key(0.25, 0.7, "cubic"), Key(0.8, 0.6), Key(1, 0)]),

    # 4 · shrug — "pata nahi" pop up + tilt
    _g("shrug", 750,
       bounce=[Key(0, 0), Key(0.3, -18, "back"), Key(0.7, -14), Key(1, 0, "sine")],
       squash=[Key(0, 0), Key(0.3, 0.05, "back"), Key(1, 0, "sine")],
       head_tilt=[Key(0, 0), Key(0.35, 9, "back"), Key(0.75, 8), Key(1, 0, "sine")],
       brow=[Key(0, 0), Key(0.3, 1.0, "cubic"), Key(0.8, 0.9), Key(1, 0)]),

    # 5 · lean_in — interest on questions
    _g("lean_in", 900,
       lean=[Key(0, 0), Key(0.35, 6, "cubic"), Key(0.8, 5.5), Key(1, 0, "sine")],
       head_nod=[Key(0, 0), Key(0.35, 0.3, "cubic"), Key(1, 0, "sine")]),

    # 6 · recoil — "arre!?" snap back then settle forward
    _g("recoil", 850,
       lean=[Key(0, 0), Key(0.15, -9, "back"), Key(0.55, -6),
             Key(1, 0, "elastic")],
       head_nod=[Key(0, 0), Key(0.15, -0.6, "back"), Key(1, 0, "elastic")],
       brow=[Key(0, 0), Key(0.12, 1.0, "cubic"), Key(0.7, 0.9), Key(1, 0)],
       squash=[Key(0, 0), Key(0.15, 0.04, "back"), Key(1, 0, "sine")]),

    # 7 · facepalm — head drops, brows crash
    _g("facepalm", 1100,
       head_nod=[Key(0, 0), Key(0.3, 1.0, "cubic"), Key(0.8, 0.95),
                 Key(1, 0, "sine")],
       head_tilt=[Key(0, 0), Key(0.3, -6, "cubic"), Key(1, 0, "sine")],
       brow=[Key(0, 0), Key(0.25, -1.0, "cubic"), Key(0.85, -0.9), Key(1, 0)],
       lean=[Key(0, 0), Key(0.35, -3, "cubic"), Key(1, 0, "sine")]),

    # 8 · jump — squash-anticipate → leap → land with overshoot
    _g("jump", 900,
       bounce=[Key(0, 0), Key(0.15, 8, "quad"), Key(0.45, -55, "back"),
               Key(0.8, 4, "cubic"), Key(1, 0, "elastic")],
       squash=[Key(0, 0), Key(0.15, -0.10, "quad"), Key(0.45, 0.08, "back"),
               Key(0.8, -0.06, "cubic"), Key(1, 0, "elastic")],
       brow=[Key(0, 0), Key(0.4, 0.8, "cubic"), Key(1, 0)]),

    # 9 · think_tilt — slow contemplative tilt, gaze drifts up
    _g("think_tilt", 1500,
       head_tilt=[Key(0, 0), Key(0.3, 10, "sine"), Key(0.8, 9), Key(1, 0, "sine")],
       head_yaw=[Key(0, 0), Key(0.35, -0.3, "sine"), Key(0.8, -0.25), Key(1, 0)],
       head_nod=[Key(0, 0), Key(0.3, -0.25, "sine"), Key(1, 0, "sine")],
       brow=[Key(0, 0), Key(0.3, 0.4, "sine"), Key(1, 0)]),

    # 10 · excited_bounce — two happy hops
    _g("excited_bounce", 1000,
       bounce=[Key(0, 0), Key(0.2, -26, "back"), Key(0.4, 0, "quad"),
               Key(0.6, -20, "back"), Key(0.85, 0, "quad"), Key(1, 0)],
       squash=[Key(0, 0), Key(0.2, 0.05, "back"), Key(0.4, -0.05, "quad"),
               Key(0.6, 0.04, "back"), Key(1, 0, "sine")],
       head_tilt=[Key(0, 0), Key(0.3, -5, "sine"), Key(0.7, 5, "sine"),
                  Key(1, 0, "sine")]),

    # micro_nod — tiny speech-beat bob (internal, amplitude-scaled)
    _g("micro_nod", 200,
       head_nod=[Key(0, 0), Key(0.4, 0.22, "cubic"), Key(1, 0, "sine")]),
]}

# ═══════════════════════════════════════════
# GESTURE → BODY POSE MAPPING
# Assigns body_pose to gestures that should trigger a pose image swap.
# Gestures without a mapping keep whatever pose is already active.
# ═══════════════════════════════════════════

GESTURE_POSE_MAP: Dict[str, str] = {
    "point":          "pointing_up",
    "shrug":          "shrug",
    "think_tilt":     "thinking",
    "recoil":         "surprised",
    "excited_bounce": "excited",
    "facepalm":       "facepalm",
    "lean_in":        "presenting",
    "nod":            "confident",
    "jump":           "excited",
}

# Apply body_pose to all registered gestures
for _gname, _pname in GESTURE_POSE_MAP.items():
    if _gname in GESTURES:
        GESTURES[_gname].body_pose = _pname


# ═══════════════════════════════════════════
# KEYWORD → GESTURE TRIGGERS
# (normalized word → gesture, fired at VTT word start)
# ═══════════════════════════════════════════

KEYWORD_TRIGGERS: Dict[str, str] = {
    # pointing / directing attention
    "dekho": "point", "देखो": "point", "yeh": "point", "यह": "point",
    "ये": "point", "iska": "point", "isko": "point", "look": "point",
    # amazement / surprise
    "arre": "recoil", "अरे": "recoil", "wow": "recoil", "waah": "recoil",
    "वाह": "recoil", "amazing": "recoil", "kya!": "recoil", "oho": "recoil",
    # negation
    "nahi": "shake", "नहीं": "shake", "nahin": "shake", "no": "shake",
    "galat": "shake", "गलत": "shake",
    # agreement
    "haan": "nod", "हां": "nod", "हाँ": "nod", "yes": "nod", "bilkul": "nod",
    "बिल्कुल": "nod", "sahi": "nod", "सही": "nod", "exactly": "nod",
    # thinking
    "socho": "think_tilt", "सोचो": "think_tilt", "hmm": "think_tilt",
    "shayad": "think_tilt", "शायद": "think_tilt",
    # ease / dismissal
    "easy": "shrug", "simple": "shrug", "bas": "shrug", "बस": "shrug",
    # excitement
    "mast": "excited_bounce", "superb": "excited_bounce",
    "jhakkas": "excited_bounce", "great": "excited_bounce",
    # attention
    "suno": "lean_in", "सुनो": "lean_in", "chalo": "lean_in", "चलो": "lean_in",
    "batao": "lean_in", "बताओ": "lean_in",
    # impact words
    "boom": "jump", "dhamaka": "jump", "धमाका": "jump", "blast": "jump",
    # exasperation
    "uff": "facepalm", "उफ्फ": "facepalm", "offo": "facepalm",
}

_STRIP = ".,!?;:'\"()-—।"

# Small "beat" gestures used to fill keyword-less stretches of speech —
# real speakers gesticulate continuously, not only on trigger words.
# All carry a body_pose via GESTURE_POSE_MAP, so the HANDS visibly move.
BEAT_GESTURES: Tuple[str, ...] = ("point", "lean_in", "nod",
                                  "excited_bounce", "shrug")


def normalize_word(word: str) -> str:
    return word.strip().strip(_STRIP).lower()


def trigger_for(word: str) -> Optional[str]:
    return KEYWORD_TRIGGERS.get(normalize_word(word))


def _is_numeric(word: str) -> bool:
    w = normalize_word(word)
    return bool(w) and all(c.isdigit() or c in ".,%" for c in w) \
        and any(c.isdigit() for c in w)


# ═══════════════════════════════════════════
# GESTURE TRACK — schedule + additive sampling
# ═══════════════════════════════════════════

@dataclass
class _Scheduled:
    gesture: Gesture
    start_ms: float
    scale: float = 1.0


class GestureTrack:
    """Schedules gestures on the global ms axis; samples the additive sum
    of every active gesture at time t (overlaps blend naturally)."""

    def __init__(self) -> None:
        self._items: List[_Scheduled] = []
        self._starts: List[float] = []

    def schedule(self, name: str, start_ms: float, scale: float = 1.0) -> bool:
        g = GESTURES.get(name)
        if g is None:
            return False
        # Debounce: never stack the same gesture within its own duration
        for it in reversed(self._items):
            if it.gesture.name == name and abs(it.start_ms - start_ms) < g.duration_ms:
                return False
        idx = bisect.bisect_right(self._starts, start_ms)
        self._items.insert(idx, _Scheduled(g, start_ms, scale))
        self._starts.insert(idx, start_ms)
        return True

    def schedule_from_words(self, words: Sequence, energy: float = 1.0) -> int:
        """Scan Timeline WordEvents; fire keyword gestures at word starts."""
        n = 0
        for w in words:
            name = trigger_for(w.text)
            if name and self.schedule(name, w.start_ms, scale=energy):
                n += 1
        return n

    def schedule_beats(self, words: Sequence, rng,
                       energy: float = 1.0,
                       min_gap_ms: float = 4500.0) -> int:
        """Fill keyword-less stretches with small deterministic beat
        gestures (soft scale) so the speaker's hands stay ALIVE for the
        whole turn. Numbers get an automatic point — teachers count on
        their fingers. `rng` must be a seeded random.Random for
        bit-identical re-renders."""
        n = 0
        last_ms = -1e9
        for w in words:
            if trigger_for(w.text):            # keyword gesture owns this beat
                last_ms = w.start_ms
                continue
            if w.start_ms - last_ms < min_gap_ms:
                continue
            if _is_numeric(w.text):
                name, scale = "point", 0.6 * energy
            else:
                name = BEAT_GESTURES[rng.randrange(len(BEAT_GESTURES))]
                scale = rng.uniform(0.35, 0.55) * energy
            if self.schedule(name, w.start_ms, scale=scale):
                last_ms = w.start_ms
                n += 1
        return n

    # Hard per-channel limits: overlapping gestures blend ADDITIVELY, and
    # an entry gesture + keyword trigger + beat landing together used to
    # sum to 2×–3× a single gesture's amplitude — the head/eyes/brows
    # visibly "flew off" the face. Clamping the SUM (not each gesture)
    # keeps overlaps natural while making the extremes physically sane.
    _CHANNEL_LIMITS: Dict[str, float] = {
        "lean": 10.0, "head_tilt": 12.0, "head_yaw": 0.85,
        "head_nod": 1.0, "bounce": 60.0, "sway": 30.0,
        "squash": 0.12, "brow": 1.2,
    }

    def sample(self, t_ms: float) -> Dict[str, float]:
        out = {ch: 0.0 for ch in CHANNELS}
        # Only gestures whose window can contain t (max duration 1.5 s)
        lo = bisect.bisect_left(self._starts, t_ms - 2000)
        hi = bisect.bisect_right(self._starts, t_ms)
        for it in self._items[lo:hi]:
            t_norm = (t_ms - it.start_ms) / it.gesture.duration_ms
            if 0.0 <= t_norm <= 1.0:
                for ch, v in it.gesture.sample(t_norm).items():
                    out[ch] += v * it.scale
        for ch, lim in self._CHANNEL_LIMITS.items():
            if out[ch] > lim:
                out[ch] = lim
            elif out[ch] < -lim:
                out[ch] = -lim
        return out

    def clear_before(self, t_ms: float) -> None:
        """Drop long-finished gestures (keeps batch renders O(1) memory)."""
        cut = bisect.bisect_left(self._starts, t_ms - 5000)
        if cut > 0:
            del self._items[:cut]
            del self._starts[:cut]

    # A full-body pose swap is the most violent thing a gesture can do to
    # the silhouette. Soft beat gestures (scale ~0.35–0.55) should only
    # move the bones — reserving image swaps for deliberate, full-strength
    # gestures is what stops the "characters switching poses so fast" chaos.
    _POSE_SWAP_MIN_SCALE = 0.60

    def active_pose(self, t_ms: float) -> str:
        """Return the body_pose of the most recently triggered STRONG
        gesture that has a pose mapping. Returns '' if no pose-bearing
        gesture is active (caller should keep the current pose)."""
        lo = bisect.bisect_left(self._starts, t_ms - 2000)
        hi = bisect.bisect_right(self._starts, t_ms)
        best_pose = ""
        best_start = -1e9
        for it in self._items[lo:hi]:
            t_norm = (t_ms - it.start_ms) / it.gesture.duration_ms
            if 0.0 <= t_norm <= 1.0 and it.gesture.body_pose \
                    and it.scale >= self._POSE_SWAP_MIN_SCALE:
                if it.start_ms > best_start:
                    best_start = it.start_ms
                    best_pose = it.gesture.body_pose
        return best_pose
