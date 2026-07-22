"""
JEEVidya V5 — Professional Lip Sync
═══════════════════════════════════
Three-state amplitude flapping is THE anime-dub tell. Real mouths:

  1. open FAST and close SLOW (muscle asymmetry) — so the openness
     signal is an attack/release envelope over loudness, not a
     threshold ladder. No jitter, no strobing.
  2. change SHAPE with the phoneme, not just size. We derive viseme
     classes from the ACTUAL text of the word being spoken (edge-tts
     VTT gives us word boundaries; the word's letters give us its
     articulation): bilabials close the mouth mid-word, rounded vowels
     narrow it, open vowels widen it.
  3. blend continuously: mouth frames are float-alpha mixes of the
     neutral/talk_1/talk_2 art, quantized to 12 levels and cached, so
     the per-frame cost is a dict lookup.

Works on plain expression art (no rig needed); Tier 1 rigs consume the
same (openness, viseme) signal for true mesh visemes.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PIL import Image

# ═══════════════════════════════════════════
# VISEME CLASSES FROM TEXT (Devanagari + Latin)
# ═══════════════════════════════════════════

VISEME_REST = "rest"
VISEME_OPEN = "open"        # a, aa, e → wide open
VISEME_ROUND = "round"      # o, u, w → narrow/rounded
VISEME_CLOSED = "closed"    # p, b, m → lips together
VISEME_TEETH = "teeth"      # f, v, s → small opening

_LATIN = {
    **{c: VISEME_OPEN for c in "aáàâeéèêæ"},
    **{c: VISEME_ROUND for c in "oóòôuúùûw"},
    **{c: VISEME_CLOSED for c in "pbm"},
    **{c: VISEME_TEETH for c in "fvsz"},
}
_DEVANAGARI = {
    **{c: VISEME_OPEN for c in "अआएऐाे ैयऱ"},
    **{c: VISEME_ROUND for c in "ओऔउऊोौुूव"},
    **{c: VISEME_CLOSED for c in "पफबभम"},
    **{c: VISEME_TEETH for c in "सशषज़फ़"},
}


def word_visemes(word: str) -> List[str]:
    """Ordered viseme classes for a word's articulable characters."""
    out: List[str] = []
    for ch in word.lower():
        v = _LATIN.get(ch) or _DEVANAGARI.get(ch)
        if v:
            out.append(v)
        elif ch.isalpha():
            out.append(VISEME_REST)     # consonants: neutral opening
    return out or [VISEME_REST]


def viseme_at(word: str, progress: float) -> str:
    """Viseme class at fractional progress (0..1) through a word."""
    vs = word_visemes(word)
    idx = min(len(vs) - 1, int(progress * len(vs)))
    return vs[idx]


# ═══════════════════════════════════════════
# OPENNESS ENVELOPE (attack/release over loudness)
# ═══════════════════════════════════════════

ATTACK = 0.55     # per-frame approach when opening (fast)
RELEASE = 0.18    # per-frame approach when closing (slow)
DB_FLOOR = -42.0
DB_CEIL = -12.0


def openness_track(frame_audio: List[Dict]) -> List[float]:
    """Per-frame continuous mouth openness 0..1 from the amplitude
    analysis. One pass per segment, then O(1) per frame."""
    track: List[float] = []
    o = 0.0
    for fa in frame_audio:
        db = float(fa.get("db", -80.0))
        target = (db - DB_FLOOR) / (DB_CEIL - DB_FLOOR)
        target = max(0.0, min(1.0, target))
        k = ATTACK if target > o else RELEASE
        o += (target - o) * k
        track.append(o)
    return track


def shaped_openness(openness: float, viseme: str) -> Tuple[float, float]:
    """(openness, width_bias) after phoneme shaping.
    width_bias −1 = narrow/rounded … +1 = wide."""
    if viseme == VISEME_CLOSED:
        return openness * 0.15, 0.0          # lips nearly meet
    if viseme == VISEME_ROUND:
        return openness * 0.8, -1.0
    if viseme == VISEME_TEETH:
        return openness * 0.45, 0.3
    if viseme == VISEME_OPEN:
        return min(1.0, openness * 1.15), 1.0
    return openness, 0.0


# ═══════════════════════════════════════════
# CONTINUOUS MOUTH BLENDER (quantized + cached)
# ═══════════════════════════════════════════

N_LEVELS = 12


class MouthBlender:
    """Float-alpha blends between neutral → talk_1 → talk_2, quantized
    to N_LEVELS and cached per (base_expression, level, width_bucket).

    The result: mouth motion becomes a continuous signal instead of a
    3-frame flipbook — the difference is immediately visible."""

    def __init__(self, library):
        """library: pipeline.lipsync.ExpressionLibrary (or compatible)."""
        self.lib = library
        self._cache: Dict[Tuple, Optional[Image.Image]] = {}

    def frame(self, base_expr: str, openness: float,
              width_bias: float = 0.0) -> Optional[Image.Image]:
        level = max(0, min(N_LEVELS, round(openness * N_LEVELS)))
        wb = 0 if abs(width_bias) < 0.5 else (1 if width_bias > 0 else -1)
        key = (base_expr, level, wb)
        if key in self._cache:
            return self._cache[key]

        img = self._build(base_expr, level / N_LEVELS, wb)
        self._cache[key] = img
        return img

    def _build(self, base_expr: str, o: float, wb: int
               ) -> Optional[Image.Image]:
        neutral = self.lib.get(base_expr) or self.lib.get("neutral")
        talk1 = self.lib.get("talk_1")
        talk2 = self.lib.get("talk_2")
        if neutral is None:
            return None
        if talk1 is None or talk2 is None or o <= 0.02:
            return neutral

        # Rounded visemes peak at talk_1 (small mouth); wide go to talk_2
        if wb < 0:
            o = min(o, 0.55)

        def match(img, ref):
            return img if img.size == ref.size else img.resize(
                ref.size, Image.Resampling.LANCZOS)

        if o < 0.5:
            return Image.blend(neutral, match(talk1, neutral), o * 2.0)
        return Image.blend(match(talk1, neutral), match(talk2, neutral),
                           (o - 0.5) * 2.0)


# ═══════════════════════════════════════════
# THE PER-FRAME SIGNAL (what the compositor calls)
# ═══════════════════════════════════════════

class LipSyncTrack:
    """One speaking turn's complete lip-sync signal, built once."""

    def __init__(self, frame_audio: List[Dict], words, span_start_ms: int,
                 fps: int):
        self.openness = openness_track(frame_audio)
        self.words = words or []
        self.start_ms = span_start_ms
        self.fps = fps

    def at(self, local_frame: int, t_ms: float) -> Tuple[float, float]:
        """(openness, width_bias) for this frame — phoneme-shaped."""
        o = self.openness[local_frame] \
            if 0 <= local_frame < len(self.openness) else 0.0
        viseme = VISEME_REST
        for w in self.words:
            if w.start_ms <= t_ms < w.end_ms:
                progress = (t_ms - w.start_ms) / max(1, w.end_ms - w.start_ms)
                viseme = viseme_at(w.text, progress)
                break
        return shaped_openness(o, viseme)
