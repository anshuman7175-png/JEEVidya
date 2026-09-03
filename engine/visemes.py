"""10-class viseme engine: Devanagari + Latin G2P with Ohala schwa
deletion (C5), raised-cosine coarticulation with bilabial dominance,
anticipatory lip rounding, 15ms/110ms amplitude envelope, and
long-pause REST enforcement (C8)."""
from __future__ import annotations

import math
import random
from itertools import pairwise
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from engine.curves import clamp01, raised_cosine


class V(str, Enum):
    REST = "REST"
    BILABIAL = "BILABIAL"
    LABIODENTAL = "LABIODENTAL"
    DENTAL = "DENTAL"
    RETROFLEX = "RETROFLEX"
    OPEN_A = "OPEN_A"
    MID_E = "MID_E"
    CLOSED_I = "CLOSED_I"
    ROUNDED_TENSE = "ROUNDED_TENSE"
    ROUNDED_LAX = "ROUNDED_LAX"


VOWELS = {V.OPEN_A, V.MID_E, V.CLOSED_I, V.ROUNDED_TENSE, V.ROUNDED_LAX}
ROUNDED = {V.ROUNDED_TENSE, V.ROUNDED_LAX}

# jaw-drop factor per class (drives vertical mouth stretch in BoneEngine)
JAW: Dict[V, float] = {
    V.REST: 0.0, V.BILABIAL: 0.0, V.LABIODENTAL: 0.08, V.DENTAL: 0.18,
    V.RETROFLEX: 0.30, V.OPEN_A: 1.0, V.MID_E: 0.55, V.CLOSED_I: 0.25,
    V.ROUNDED_TENSE: 0.35, V.ROUNDED_LAX: 0.55,
}

# ---- Devanagari G2P mapping ----

_DEV_MAP: Dict[str, V] = {}
for _chars, _cls in [
    ("\u092e\u092c\u092a\u092d", V.BILABIAL),              # म ब प भ
    ("\u092b\u0935", V.LABIODENTAL),                        # फ व
    ("\u0924\u0925\u0926\u0927\u0928\u0938\u0932\u091c\u091d\u0936\u0937\u091a\u091b\u092f\u0939", V.DENTAL),
    ("\u091f\u0920\u0921\u0922\u0923\u0915\u0916\u0917\u0918\u0919\u091e\u0930", V.RETROFLEX),
    ("\u0905\u0906", V.OPEN_A), ("\u093e", V.OPEN_A),       # अ आ, aa matra
    ("\u090f\u0910", V.MID_E), ("\u0947\u0948", V.MID_E),   # ए ऐ, e/ai matra
    ("\u0907\u0908", V.CLOSED_I), ("\u093f\u0940", V.CLOSED_I),
    ("\u090a\u0909", V.ROUNDED_TENSE), ("\u0941\u0942", V.ROUNDED_TENSE),
    ("\u0913\u0914", V.ROUNDED_LAX), ("\u094b\u094c", V.ROUNDED_LAX),
]:
    for c in _chars:
        _DEV_MAP[c] = _cls

_LAT_MAP: Dict[str, V] = {}
for _chars, _cls in [
    ("mbp", V.BILABIAL), ("fvw", V.LABIODENTAL),
    ("tdnslzjc", V.DENTAL),
    ("kgrqxh", V.RETROFLEX),
    ("a", V.OPEN_A), ("e", V.MID_E),
    ("iy", V.CLOSED_I), ("u", V.ROUNDED_TENSE), ("o", V.ROUNDED_LAX),
]:
    for c in _chars:
        _LAT_MAP[c] = _cls

# Digits are spoken aloud, so a numeric token ("9.8") must still move the
# mouth. Map each digit to the viseme run of its spoken English name —
# a coarse but correct-order approximation for lip sync.
_DIGIT_NAMES = ("zero", "one", "two", "three", "four",
                "five", "six", "seven", "eight", "nine")
_DIGIT_G2P: Dict[int, List[V]] = {
    d: [_LAT_MAP[c] for c in name if c in _LAT_MAP]
    for d, name in enumerate(_DIGIT_NAMES)
}

# JEE content is saturated with exponents ("9.8 m/s²", "sin²θ"). These are
# *spoken words*, not digits — and `"²".isdigit()` is True while
# `int("²")` raises, so they must be handled before any digit branch.
_SUPERSCRIPT_WORDS: Dict[str, str] = {
    "\u00b2": "squared", "\u00b3": "cubed",
    "\u00b9": "", "\u2070": "", "\u2074": "fourth", "\u207f": "n",
}
_SUPERSCRIPT_G2P: Dict[str, List[V]] = {
    ch: [_LAT_MAP[c] for c in word if c in _LAT_MAP]
    for ch, word in _SUPERSCRIPT_WORDS.items()
}

_VIRAMA = "\u094d"
_NUKTA = "\u093c"
_SKIP = set(_VIRAMA + _NUKTA + "\u0902\u0901\u0903\u093d\u093c\u0970")

ATTACK_MS = 15.0
RELEASE_MS = 110.0
SHORT_TAIL_MS = 90.0     # short inter-word gaps relax toward REST
HARD_PAUSE_MS = 300.0    # gaps beyond this force full REST (C8)
BREATH_PAUSE_MS = 500.0  # gaps beyond this trigger a visible inhale


# ---- G2P ----

def _is_dev_consonant(ch: str) -> bool:
    return "\u0915" <= ch <= "\u0939"


def _lookahead(chars: List[str], i: int) -> str:
    """Next significant character, skipping nukta — which re-articulates
    the preceding consonant (क → क़) without supplying a vowel. Without
    the skip, क़ल reads as if क carried an inherent schwa."""
    j = i + 1
    while j < len(chars) and chars[j] == _NUKTA:
        j += 1
    return chars[j] if j < len(chars) else ""


def _delete_schwas(seq: List[V], inherent: List[bool]) -> List[V]:
    """Ohala's Hindi schwa-deletion rule (Terminal Plan C5).

    Devanagari *writes* the inherent schwa that modern Hindi does not
    *pronounce*: कमल is "kamal", not "kamala". Because OPEN_A carries the
    largest jaw drop in the inventory (JAW == 1.0), a retained final
    schwa is the single most visible lip-sync error available — the
    character ends nearly every Hindi word with a wide-open mouth.

    Two rules, applied right-to-left as Ohala specifies:
      1. word-final inherent schwa deletes;
      2. medial inherent schwa deletes in V C _ C V.

    Only schwas this module *inserted* are eligible (`inherent`), so a
    written matra can never be deleted — the invariant the plan demands.
    A word is never left without a vowel, which protects monosyllables
    like न ("na").
    """
    alive = [True] * len(seq)

    def prev_alive(k: int) -> int:
        j = k - 1
        while j >= 0 and not alive[j]:
            j -= 1
        return j

    def next_alive(k: int) -> int:
        j = k + 1
        while j < len(seq) and not alive[j]:
            j += 1
        return j if j < len(seq) else -1

    def vowel_count() -> int:
        return sum(1 for k, v in enumerate(seq) if alive[k] and v in VOWELS)

    # Rule 1 — word-final. Only the last surviving segment qualifies.
    for k in range(len(seq) - 1, -1, -1):
        if not alive[k]:
            continue
        if inherent[k] and vowel_count() > 1:
            alive[k] = False
        break

    # Rule 2 — medial, in V C _ C V (right-to-left, so already-deleted
    # schwas correctly stop further deletion: कमल keeps its first 'a').
    for k in range(len(seq) - 1, -1, -1):
        if not alive[k] or not inherent[k]:
            continue
        c1 = prev_alive(k)
        if c1 < 0 or seq[c1] in VOWELS:
            continue
        v0 = prev_alive(c1)
        c2 = next_alive(k)
        if v0 < 0 or seq[v0] not in VOWELS or c2 < 0 or seq[c2] in VOWELS:
            continue
        v2 = next_alive(c2)
        if v2 < 0 or seq[v2] not in VOWELS:
            continue
        if vowel_count() > 1:
            alive[k] = False

    # alive is allocated as a parallel flag array over seq; strict=True
    # keeps that invariant enforced rather than assumed.
    return [v for v, a in zip(seq, alive, strict=True) if a]


def _g2p_word(token: str) -> List[V]:
    """One whitespace-delimited token → viseme run, with the inherent
    schwa inserted and then deleted where Hindi drops it."""
    out: List[V] = []
    inherent: List[bool] = []          # parallel flags: unwritten schwa
    chars = list(token)
    for i, ch in enumerate(chars):
        if ch in _SKIP or ch.isspace():
            continue
        cls = _DEV_MAP.get(ch)
        if cls is not None:
            out.append(cls)
            inherent.append(False)
            # Inherent schwa: a bare consonant gets an 'a' unless followed
            # by a vowel sign or a virama.
            if _is_dev_consonant(ch) and cls not in VOWELS:
                nxt = _lookahead(chars, i)
                if _DEV_MAP.get(nxt) not in VOWELS and nxt != _VIRAMA:
                    out.append(V.OPEN_A)
                    inherent.append(True)
            continue
        cls = _LAT_MAP.get(ch.lower())
        if cls is not None:
            out.append(cls)
            inherent.append(False)
            continue
        sup = _SUPERSCRIPT_G2P.get(ch)
        if sup is not None:                          # exponent, spoken
            for v in sup:
                out.append(v)
                inherent.append(False)
            continue
        if "0" <= ch <= "9":                         # ASCII digits only:
            # `str.isdigit()` also accepts superscripts and other numeric
            # forms that `int()` rejects outright.
            for v in _DIGIT_G2P.get(int(ch), ()):    # spoken digit name
                out.append(v)
                inherent.append(False)
    return _delete_schwas(out, inherent)


def g2p(text: str) -> List[V]:
    """Grapheme to viseme. Tokenized per word so that Ohala's word-final
    schwa rule has a word boundary to anchor on."""
    out: List[V] = []
    for token in text.split():
        out.extend(_g2p_word(token))
    # Collapse immediate repeats
    dedup: List[V] = []
    for v in out:
        if not dedup or dedup[-1] != v:
            dedup.append(v)
    return dedup


# ---- Timed events ----

@dataclass
class VisemeEvent:
    viseme: V
    start_ms: float
    end_ms: float

    @property
    def dur(self) -> float:
        return self.end_ms - self.start_ms


# Sub-frame integration (Terminal Plan §7.4): a 20 ms plosive closure
# must contribute ~60% weight to the frame it lives in instead of
# vanishing between samples. 4 sub-samples per rendered frame,
# box-filtered; total viseme-weight mass is conserved (asserted in tests).
SUBFRAMES = 4

# Articulatory dominance rank for min-duration coalescing: when an event
# is too short to survive one sub-frame, it merges into whichever
# neighbour is MORE articulatorily dominant — bilabials win (a closure
# cannot be deleted), open vowels lose (they are the default backdrop).
COALESCE_RANK: Dict[V, int] = {
    V.BILABIAL: 9, V.LABIODENTAL: 8, V.ROUNDED_TENSE: 7, V.ROUNDED_LAX: 6,
    V.CLOSED_I: 5, V.RETROFLEX: 4, V.DENTAL: 3, V.MID_E: 2,
    V.OPEN_A: 1, V.REST: 0,
}


def coalesce_events(events: List[VisemeEvent],
                    min_dur_ms: float) -> List[VisemeEvent]:
    """Merge events shorter than min_dur_ms into the articulatorily
    dominant neighbour. Vowels with duration >= 30ms are protected from
    being swallowed by consonants to ensure crisp syllable articulation."""
    if not events:
        return events
    evs = [VisemeEvent(e.viseme, e.start_ms, e.end_ms) for e in events]
    changed = True
    while changed and len(evs) > 1:
        changed = False
        for i, e in enumerate(evs):
            # Protect vowels from being swallowed by plosive consonants
            if e.viseme in VOWELS and e.dur >= 30.0:
                continue
            if e.dur >= min_dur_ms:
                continue
            prev_e = evs[i - 1] if i > 0 else None
            next_e = evs[i + 1] if i + 1 < len(evs) else None
            # If same viseme adjacent, merge immediately
            if prev_e is not None and prev_e.viseme == e.viseme:
                prev_e.end_ms = e.end_ms
                del evs[i]
                changed = True
                break
            if next_e is not None and next_e.viseme == e.viseme:
                next_e.start_ms = e.start_ms
                del evs[i]
                changed = True
                break
            # pick the more dominant ADJACENT event
            cand = []
            if prev_e is not None and prev_e.end_ms >= e.start_ms - 1e-6:
                cand.append(("prev", COALESCE_RANK[prev_e.viseme]))
            if next_e is not None and next_e.start_ms <= e.end_ms + 1e-6:
                cand.append(("next", COALESCE_RANK[next_e.viseme]))
            if not cand:
                continue
            self_rank = COALESCE_RANK[e.viseme]
            weakest = min(cand, key=lambda c: c[1])
            if self_rank > max(r for _, r in cand):
                if weakest[0] == "prev" and prev_e is not None:
                    e.start_ms = prev_e.start_ms
                    del evs[i - 1]
                elif next_e is not None:
                    e.end_ms = next_e.end_ms
                    del evs[i + 1]
                changed = True
                break
            winner = max(cand, key=lambda c: c[1])
            if winner[0] == "prev" and prev_e is not None:
                prev_e.end_ms = e.end_ms
            elif next_e is not None:
                next_e.start_ms = e.start_ms
            del evs[i]
            changed = True
            break
    return evs


class VisemeTrack:
    """Timed viseme events with per-frame blended weights and jaw drop."""

    def __init__(self, events: List[VisemeEvent], turn_end_ms: float):
        self.events = events
        self.turn_end_ms = turn_end_ms

    @classmethod
    def from_aligned_events(cls, events: List[VisemeEvent],
                            turn_end_ms: float, fps: int) -> "VisemeTrack":
        """Build from engine/align.py phoneme-exact events (the Tier-1/2
        path). Coalesces events shorter than one SUB-frame at this fps
        into the articulatorily dominant neighbour (§7.4)."""
        min_dur = (1000.0 / max(1, fps)) / SUBFRAMES
        return cls(coalesce_events(events, min_dur), turn_end_ms)

    # Minimum event duration on the production VTT path (§7.4).
    # 40ms (~2.4 frames @ 60fps) preserves distinct consonants and vowels
    # while eliminating sub-frame fluttering.
    WORD_EVENT_MIN_DUR_MS = 40.0

    @classmethod
    def from_words(cls, words: Sequence,
                   turn_end_ms: Optional[float] = None) -> "VisemeTrack":
        """Build a viseme track from VTT word events. Events shorter
        than WORD_EVENT_MIN_DUR_MS are merged into the articulatorily
        dominant neighbour, mirroring from_aligned_events."""
        events: List[VisemeEvent] = []
        for w in words:
            phones = g2p(w.text)
            if not phones:
                continue
            # Vowels receive twice the time of consonants
            weights = [1.4 if p in VOWELS else 0.7 for p in phones]
            total = sum(weights)
            span = max(50.0, w.end_ms - w.start_ms)
            t = w.start_ms
            for p, wt in zip(phones, weights, strict=True):
                d = span * wt / total
                events.append(VisemeEvent(p, t, t + d))
                t += d
        end = turn_end_ms if turn_end_ms is not None else (
            events[-1].end_ms if events else 0.0)
        return cls(coalesce_events(events, cls.WORD_EVENT_MIN_DUR_MS), end)

    # ---- query ----

    @staticmethod
    def _blend_window(a: "VisemeEvent", b: "VisemeEvent") -> float:
        """Half-width of the cross-fade window centered on the a→b boundary.
        Scales with articulator travel: a full jaw excursion (BILABIAL→
        OPEN_A) physically takes longer than a small shape change, and a
        window shorter than ~2 frames at 30 fps reads as a mouth pop."""
        travel = abs(JAW[b.viseme] - JAW[a.viseme])
        return min(80.0, max(0.4 * min(a.dur, max(1.0, b.dur)),
                             60.0 * travel))

    def _find(self, t_ms: float) -> int:
        """Binary search for the last event starting <= t_ms."""
        lo, hi = 0, len(self.events) - 1
        idx = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.events[mid].start_ms <= t_ms:
                idx = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return idx

    def _envelope(self, t_ms: float) -> float:
        """Attack 15ms into speech, release 110ms into silence."""
        idx = self._find(t_ms)
        if idx < 0:
            return 0.0
        ev = self.events[idx]
        if t_ms <= ev.end_ms:
            seg_start = ev.start_ms
            j = idx
            while (j > 0 and self.events[j].start_ms
                    - self.events[j - 1].end_ms < SHORT_TAIL_MS):
                j -= 1
                seg_start = self.events[j].start_ms
            return clamp01((t_ms - seg_start) / ATTACK_MS)
        # In a gap after ev
        since = t_ms - ev.end_ms
        return clamp01(1.0 - since / RELEASE_MS)

    def weights_at(self, t_ms: float,
                   energy: float = 1.0) -> Tuple[Dict[V, float], float]:
        """Returns ({viseme: weight} summing to ~1, jaw_drop 0..1)."""
        if not self.events:
            return {V.REST: 1.0}, 0.0
        idx = self._find(t_ms)
        if idx < 0:
            return {V.REST: 1.0}, 0.0
        cur = self.events[idx]
        weights: Dict[V, float] = {}

        if t_ms > cur.end_ms:                              # inside a gap
            env = self._envelope(t_ms)
            if env <= 0.01:
                return {V.REST: 1.0}, 0.0
            weights = {cur.viseme: env, V.REST: 1.0 - env}
            jaw = JAW[cur.viseme] * env * (0.5 + 0.5 * energy)
            return weights, jaw

        nxt = (self.events[idx + 1]
               if idx + 1 < len(self.events) else None)
        blend = 0.0
        if nxt is not None and nxt.start_ms - cur.end_ms < SHORT_TAIL_MS:
            T = self._blend_window(cur, nxt)
            into = t_ms - (cur.end_ms - T)
            if into > 0:
                raw = clamp01(into / (2.0 * T))
                # bilabial dominance: lips stay pressed longer / close earlier
                if cur.viseme == V.BILABIAL:
                    raw = raw ** 1.8
                elif nxt is not None and nxt.viseme == V.BILABIAL:
                    raw = 1.0 - (1.0 - raw) ** 1.8
                blend = raised_cosine(raw)
        if blend > 0.0 and nxt is not None:
            weights[cur.viseme] = 1.0 - blend
            weights[nxt.viseme] = weights.get(nxt.viseme, 0.0) + blend
        else:
            weights[cur.viseme] = 1.0

        # Complete the incoming half of the previous boundary's cross-fade.
        # The blend window is centered on prv.end_ms; without this branch
        # the mix snaps to 100% cur the instant t enters cur (single-frame
        # jaw pop at every event boundary).
        prv = self.events[idx - 1] if idx > 0 else None
        if prv is not None and cur.start_ms - prv.end_ms < SHORT_TAIL_MS:
            T = self._blend_window(prv, cur)
            into = t_ms - (prv.end_ms - T)
            if 0.0 < into < 2.0 * T:
                raw = clamp01(into / (2.0 * T))
                if prv.viseme == V.BILABIAL:
                    raw = raw ** 1.8
                elif cur.viseme == V.BILABIAL:
                    raw = 1.0 - (1.0 - raw) ** 1.8
                blend_in = raised_cosine(raw)
                if blend_in < 1.0:
                    for k in list(weights):
                        weights[k] *= blend_in
                    weights[prv.viseme] = (weights.get(prv.viseme, 0.0)
                                           + (1.0 - blend_in))

        # anticipatory rounding in the last 60ms of a vowel before ROUNDED_*
        if (nxt is not None and nxt.viseme in ROUNDED
                and cur.viseme in VOWELS and cur.viseme not in ROUNDED):
            lead = clamp01((t_ms - (cur.end_ms - 60.0)) / 60.0)
            if lead > 0:
                r = 0.25 * lead
                for k in list(weights):
                    weights[k] *= (1.0 - r)
                weights[nxt.viseme] = weights.get(nxt.viseme, 0.0) + r

        env = self._envelope(t_ms)
        if env < 1.0:
            for k in list(weights):
                weights[k] *= env
            weights[V.REST] = weights.get(V.REST, 0.0) + (1.0 - env)

        jaw = sum(JAW[v] * w for v, w in weights.items()) * (
            0.55 + 0.45 * clamp01(energy))
        return weights, jaw

    def weights_at_frame(self, t_ms: float, fps: int,
                         energy: float = 1.0) -> Tuple[Dict[V, float], float]:
        """§7.4: box-filtered sampling at SUBFRAMES points across the
        rendered frame centered on t_ms. A 20 ms closure inside a
        16.7 ms frame contributes proportional weight instead of being
        missed by the single-instant sampler. Weight mass is conserved:
        the average of normalized distributions is normalized."""
        frame_ms = 1000.0 / max(1, fps)
        acc: Dict[V, float] = {}
        jaw_acc = 0.0
        for k in range(SUBFRAMES):
            # sub-sample centers, symmetric about t_ms
            ts = t_ms + frame_ms * ((k + 0.5) / SUBFRAMES - 0.5)
            w, jaw = self.weights_at(ts, energy)
            for v, wt in w.items():
                acc[v] = acc.get(v, 0.0) + wt / SUBFRAMES
            jaw_acc += jaw / SUBFRAMES
        return acc, jaw_acc

    def breath_pauses(self) -> List[float]:
        """Centers of pauses > 500ms -- puppet plays one slow inhale."""
        out: List[float] = []
        for a, b in pairwise(self.events):
            gap = b.start_ms - a.end_ms
            if gap >= BREATH_PAUSE_MS:
                out.append(a.end_ms + min(400.0, gap * 0.4))
        return out

    def word_started_within(self, t_ms: float,
                            window_ms: float = 130) -> bool:
        """True if any viseme event began within the last window_ms.
        Used for speech-synced micro head nods."""
        idx = self._find(t_ms)
        return idx >= 0 and (t_ms - self.events[idx].start_ms) <= window_ms


# ---- Legacy compatibility ----
# The old 6-class names are mapped to new 10-class for any code that
# still references them

VISEME_CEILING: Dict[str, float] = {v.value: JAW[v] for v in V}

# Per-viseme mouth-openness ceiling (for old code paths)
def mouth_openness_blend(v_from: str, v_to: str, blend_t: float,
                         weight: float, amp_level: float) -> float:
    """Legacy compatibility: openness for a coarticulated pair."""
    ceiling_from = VISEME_CEILING.get(v_from, 0.6)
    ceiling_to = VISEME_CEILING.get(v_to, 0.6)
    ceiling = ceiling_from + (ceiling_to - ceiling_from) * blend_t
    if ceiling <= 0.0:
        return 0.0
    amp = 0.22 + 0.78 * max(0.0, min(1.0, amp_level))
    return max(0.0, min(1.0, ceiling * weight * amp))


class AmplitudeEnvelope:
    """One-pole attack/release follower. Legacy compatibility."""

    def __init__(self, fps: int, attack_ms: float = 15.0,
                 release_ms: float = 110.0):
        dt = 1000.0 / max(1, fps)
        self._ka = 1.0 - math.exp(-dt / max(1e-3, attack_ms))
        self._kr = 1.0 - math.exp(-dt / max(1e-3, release_ms))
        self.level = 0.0

    def step(self, amplitude_db: float) -> float:
        x = max(0.0, min(1.0, (amplitude_db + 50.0) / 35.0))
        k = self._ka if x > self.level else self._kr
        self.level += (x - self.level) * k
        return self.level


def visemes_for_word(word: str) -> List[str]:
    """Legacy G2P returning string names. Never returns an empty list:
    unmappable tokens fall back to REST so callers can always index."""
    return [v.value for v in g2p(word)] or [V.REST.value]
