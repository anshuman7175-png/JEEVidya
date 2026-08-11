"""
JEEVidya — Phoneme-Exact Timing (Terminal Plan, Part VII — the keystone)
════════════════════════════════════════════════════════════════════════
Fixes sync (D6/D7/D8) AND unlocks any voice (Part X), because timings
derive from the WAVEFORM, never from TTS metadata. Three tiers, each a
strict fallback of the one above, all funnelling into the same
`AlignedTurn` contract:

  Tier 1  torchaudio MMS_FA CTC forced alignment against the actual
          TTS wav. Hindi/Devanagari romanized with uroman first. A
          REAL phoneme inventory (digraph-aware: ch/sh/th/kh/gh/bh…)
          maps to the 10 viseme classes — killing D6's wrong shapes
          (c→DENTAL, h→RETROFLEX) for aspirates, gemination, schwa
          deletion and conjuncts.
  Tier 2  waveform calibration — ALWAYS applied, even over Tier 1:
          • global A/V offset: cross-correlate predicted jaw envelope
            against the audio onset envelope over ±250 ms, apply the
            argmax lag (D8);
          • DTW drift correction inside long turns;
          • silence gating: no open-vowel visemes on silent frames,
            bilabial closures snapped to amplitude dips.
  Tier 3  the current even-split G2P, retained ONLY as last resort,
          with the Tier 2 offset still applied.

Caching: `<audio_hash>.align.json` — alignment runs once per turn,
EVER. The cache key is the audio content hash + text + module version,
so a changed wav or a changed aligner can never serve stale timings.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import wave
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from engine.visemes import V, VisemeEvent, g2p

ALIGN_SCHEMA_VERSION = 1

# Tier-2 search window for the global A/V offset (D8)
MAX_OFFSET_MS = 250.0
# Envelope frame hop for calibration (ms) — fine enough for sub-frame
# accuracy at 60 fps without being noise-dominated
ENV_HOP_MS = 5.0
# RMS floor (dBFS) below which a frame counts as silence
SILENCE_DB = -46.0


# ═══════════════════════════════════════════
# Phoneme inventory — digraph-aware romanized-Hindi/Hinglish → viseme
# ═══════════════════════════════════════════
# Ordered longest-first so "chh" wins over "ch" wins over "c".
# This is the D6 fix: `ch` is not DENTAL-c + RETROFLEX-h; it is one
# post-alveolar affricate. Aspirates (kh/gh/bh/ph/th/dh/jh) are ONE
# phoneme whose viseme is the base stop's — the /h/ release does not
# reshape the lips.

_PHONEME_TABLE: Tuple[Tuple[str, V], ...] = (
    # 3-char clusters
    ("chh", V.DENTAL), ("ksh", V.DENTAL), ("shr", V.DENTAL),
    # aspirated stops → base articulator
    ("bh", V.BILABIAL), ("ph", V.LABIODENTAL),   # ph = /f/ in Hindi loanwords
    ("kh", V.RETROFLEX), ("gh", V.RETROFLEX),
    ("th", V.DENTAL), ("dh", V.DENTAL),
    ("jh", V.DENTAL), ("ch", V.DENTAL),
    ("sh", V.DENTAL), ("zh", V.DENTAL),
    # long vowels (romanized doubles)
    ("aa", V.OPEN_A), ("ee", V.CLOSED_I), ("ii", V.CLOSED_I),
    ("oo", V.ROUNDED_TENSE), ("uu", V.ROUNDED_TENSE),
    ("ai", V.MID_E), ("au", V.ROUNDED_LAX), ("ou", V.ROUNDED_LAX),
    ("ei", V.MID_E), ("oi", V.ROUNDED_LAX),
    # single consonants
    ("m", V.BILABIAL), ("b", V.BILABIAL), ("p", V.BILABIAL),
    ("f", V.LABIODENTAL), ("v", V.LABIODENTAL), ("w", V.LABIODENTAL),
    ("t", V.DENTAL), ("d", V.DENTAL), ("n", V.DENTAL), ("s", V.DENTAL),
    ("z", V.DENTAL), ("l", V.DENTAL), ("j", V.DENTAL), ("c", V.DENTAL),
    ("k", V.RETROFLEX), ("g", V.RETROFLEX), ("q", V.RETROFLEX),
    ("x", V.RETROFLEX), ("r", V.RETROFLEX),
    # /h/ alone: glottal — jaw follows the NEXT vowel; approximate with
    # a light open, never RETROFLEX (the old bug)
    ("h", V.OPEN_A),
    # single vowels
    ("a", V.OPEN_A), ("e", V.MID_E), ("i", V.CLOSED_I), ("y", V.CLOSED_I),
    ("u", V.ROUNDED_TENSE), ("o", V.ROUNDED_LAX),
)


def phonemize(token: str) -> List[Tuple[str, V]]:
    """Split a romanized token into (grapheme-cluster, viseme) pairs,
    longest-match-first. Gemination ("kk") collapses to one closure."""
    t = token.lower()
    out: List[Tuple[str, V]] = []
    i = 0
    while i < len(t):
        for graph, vis in _PHONEME_TABLE:
            if t.startswith(graph, i):
                # gemination: identical cluster repeated → single phoneme
                j = i + len(graph)
                while t.startswith(graph, j):
                    j += len(graph)
                if not out or out[-1][1] != vis or out[-1][0] != graph:
                    out.append((t[i:j], vis))
                i = j
                break
        else:
            i += 1  # unmappable char (punctuation, digit handled upstream)
    return out


def romanize(text: str) -> str:
    """Devanagari → Latin via uroman when available; identity otherwise
    (Hinglish is already Latin). Deterministic, cached per process."""
    if not re.search(r"[\u0900-\u097F]", text):
        return text
    try:
        import uroman as ur  # type: ignore
        return _uroman().romanize_string(text)
    except Exception:
        # Fallback: strip to the grapheme classes the Tier-3 G2P knows.
        return text


_UROMAN = None


def _uroman():
    global _UROMAN
    if _UROMAN is None:
        import uroman as ur  # type: ignore
        _UROMAN = ur.Uroman()
    return _UROMAN


# ═══════════════════════════════════════════
# Contract
# ═══════════════════════════════════════════

@dataclass
class AlignedWord:
    text: str
    start_ms: float
    end_ms: float
    score: float = 1.0            # aligner confidence (CTC posterior)
    phones: List[Tuple[str, str, float, float]] = field(default_factory=list)
    # (grapheme_cluster, viseme_name, start_ms, end_ms)


@dataclass
class AlignedTurn:
    """The single timing contract every downstream consumer reads:
    gestures, visemes, prosody acting, QC. One timing source (Part VII)."""
    tier: int                     # 1 = MMS_FA, 2 = calibrated-only, 3 = even-split
    audio_hash: str
    duration_ms: float
    offset_ms: float              # Tier-2 global A/V offset, ALREADY APPLIED
    confidence: float             # mean aligner score (1.0 for Tier 3)
    words: List[AlignedWord] = field(default_factory=list)

    def viseme_events(self) -> List[VisemeEvent]:
        evs: List[VisemeEvent] = []
        for w in self.words:
            for _g, vis, s, e in w.phones:
                evs.append(VisemeEvent(V(vis), s, e))
        return evs

    # ── persistence (the once-per-turn-ever cache) ──

    def to_dict(self) -> dict:
        return {
            "schema": ALIGN_SCHEMA_VERSION, "tier": self.tier,
            "audio_hash": self.audio_hash, "duration_ms": self.duration_ms,
            "offset_ms": self.offset_ms, "confidence": self.confidence,
            "words": [{"text": w.text, "start_ms": w.start_ms,
                       "end_ms": w.end_ms, "score": w.score,
                       "phones": w.phones} for w in self.words],
        }

    @staticmethod
    def from_dict(d: dict) -> "AlignedTurn":
        return AlignedTurn(
            tier=d["tier"], audio_hash=d["audio_hash"],
            duration_ms=d["duration_ms"], offset_ms=d["offset_ms"],
            confidence=d["confidence"],
            words=[AlignedWord(text=w["text"], start_ms=w["start_ms"],
                               end_ms=w["end_ms"], score=w.get("score", 1.0),
                               phones=[tuple(p) for p in w["phones"]])
                   for w in d["words"]])


# ═══════════════════════════════════════════
# Audio utilities (pure numpy — no librosa dependency on the hot path)
# ═══════════════════════════════════════════

def load_wav_mono(path: str) -> Tuple[np.ndarray, int]:
    """Load a wav as float32 mono in [-1, 1]. Deterministic, stdlib-only."""
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width {sw}")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def audio_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:24]


def onset_envelope(x: np.ndarray, sr: int,
                   hop_ms: float = ENV_HOP_MS) -> np.ndarray:
    """Half-wave-rectified spectral-flux-ish energy onset envelope.
    Pure numpy; matches librosa's onset_strength closely enough for
    cross-correlation calibration."""
    hop = max(1, int(sr * hop_ms / 1000.0))
    win = hop * 4
    if len(x) < win:
        return np.zeros(1, dtype=np.float64)
    n_frames = 1 + (len(x) - win) // hop
    rms = np.empty(n_frames)
    for i in range(n_frames):
        seg = x[i * hop:i * hop + win]
        rms[i] = math.sqrt(float(np.mean(seg * seg)) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-9)
    flux = np.diff(db, prepend=db[0])
    flux[flux < 0] = 0.0
    m = flux.max()
    return flux / m if m > 0 else flux


def rms_db_track(x: np.ndarray, sr: int,
                 hop_ms: float = ENV_HOP_MS) -> np.ndarray:
    hop = max(1, int(sr * hop_ms / 1000.0))
    win = hop * 4
    if len(x) < win:
        return np.full(1, -90.0)
    n_frames = 1 + (len(x) - win) // hop
    out = np.empty(n_frames)
    for i in range(n_frames):
        seg = x[i * hop:i * hop + win]
        out[i] = 20.0 * math.log10(math.sqrt(float(np.mean(seg * seg)) + 1e-12) + 1e-9)
    return out


# ═══════════════════════════════════════════
# Tier 2 — waveform calibration (always-on)
# ═══════════════════════════════════════════

def predicted_jaw_envelope(events: Sequence[VisemeEvent],
                           duration_ms: float,
                           hop_ms: float = ENV_HOP_MS) -> np.ndarray:
    """The jaw-openness curve the current timing PREDICTS, sampled on
    the same grid as the audio envelope."""
    from engine.visemes import JAW
    n = max(1, int(duration_ms / hop_ms))
    env = np.zeros(n)
    for ev in events:
        a = max(0, int(ev.start_ms / hop_ms))
        b = min(n, int(math.ceil(ev.end_ms / hop_ms)))
        if b > a:
            env[a:b] = np.maximum(env[a:b], JAW[ev.viseme])
    # C¹ smooth: 25 ms box then diff-rectify to onsets, like the audio side
    k = max(1, int(25.0 / hop_ms))
    kern = np.ones(k) / k
    env = np.convolve(env, kern, mode="same")
    flux = np.diff(env, prepend=env[0])
    flux[flux < 0] = 0.0
    m = flux.max()
    return flux / m if m > 0 else flux


def global_offset_ms(pred: np.ndarray, audio: np.ndarray,
                     hop_ms: float = ENV_HOP_MS,
                     max_offset_ms: float = MAX_OFFSET_MS) -> float:
    """D8: argmax cross-correlation lag within ±max_offset_ms.
    Positive result ⇒ predicted events are EARLY by that much (shift
    events later by +offset)."""
    n = min(len(pred), len(audio))
    if n < 8:
        return 0.0
    p = pred[:n] - pred[:n].mean()
    a = audio[:n] - audio[:n].mean()
    max_lag = int(max_offset_ms / hop_ms)
    best_lag, best_val = 0, -1e18
    denom = (np.linalg.norm(p) * np.linalg.norm(a)) + 1e-12
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            v = float(np.dot(p[:n - lag], a[lag:])) / denom
        else:
            v = float(np.dot(p[-lag:], a[:n + lag])) / denom
        if v > best_val:
            best_val, best_lag = v, lag
    return best_lag * hop_ms


def dtw_warp(pred: np.ndarray, audio: np.ndarray,
             hop_ms: float = ENV_HOP_MS,
             band_ms: float = 150.0) -> Optional[np.ndarray]:
    """Sakoe–Chiba-banded DTW pred→audio. Returns, for each pred frame,
    the audio time (ms) it maps to — used to correct slow drift inside
    long turns. None when the turn is too short to bother."""
    n = min(len(pred), len(audio))
    if n < int(1500 / hop_ms):        # < 1.5 s: global offset is enough
        return None
    band = max(2, int(band_ms / hop_ms))
    INF = 1e18
    D = np.full((n + 1, n + 1), INF)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        lo, hi = max(1, i - band), min(n, i + band)
        for j in range(lo, hi + 1):
            cost = abs(pred[i - 1] - audio[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    # Backtrack
    i, j = n, n
    path_j = np.empty(n, dtype=np.int64)
    while i > 0:
        path_j[i - 1] = j - 1
        moves = ((D[i - 1, j - 1], i - 1, j - 1),
                 (D[i - 1, j], i - 1, j),
                 (D[i, j - 1], i, j - 1))
        _, i, j = min(moves, key=lambda m: m[0])
        if i == 0:
            break
    return path_j.astype(np.float64) * hop_ms


def apply_calibration(turn: AlignedTurn, wav_path: str) -> AlignedTurn:
    """Tier 2, applied over ANY tier's output: global offset + drift +
    silence gating. Mutates a copy of the turn's word/phone times."""
    x, sr = load_wav_mono(wav_path)
    duration_ms = len(x) / sr * 1000.0
    aud_env = onset_envelope(x, sr)
    events = turn.viseme_events()
    pred_env = predicted_jaw_envelope(events, duration_ms)

    off = global_offset_ms(pred_env, aud_env)
    warp = dtw_warp(
        predicted_jaw_envelope(
            [VisemeEvent(e.viseme, e.start_ms + off, e.end_ms + off)
             for e in events], duration_ms),
        aud_env)

    db = rms_db_track(x, sr)

    def remap(t_ms: float) -> float:
        t = t_ms + off
        if warp is not None:
            idx = min(len(warp) - 1, max(0, int(t / ENV_HOP_MS)))
            # blend 50% toward the DTW path: full trust would let a noisy
            # band fold time; half trust corrects drift without artifacts
            t = 0.5 * t + 0.5 * float(warp[idx])
        return max(0.0, min(duration_ms, t))

    def is_silent(t_ms: float) -> bool:
        idx = min(len(db) - 1, max(0, int(t_ms / ENV_HOP_MS)))
        return db[idx] < SILENCE_DB

    words: List[AlignedWord] = []
    for w in turn.words:
        phones = []
        for g, vis, s, e in w.phones:
            s2, e2 = remap(s), remap(e)
            if e2 - s2 < 1.0:
                e2 = min(duration_ms, s2 + 1.0)
            # Silence gate: no open-vowel viseme on a silent frame
            if V(vis) in (V.OPEN_A, V.MID_E, V.ROUNDED_LAX) \
                    and is_silent((s2 + e2) / 2.0):
                vis = V.REST.value
            phones.append((g, vis, s2, e2))
        words.append(AlignedWord(text=w.text,
                                 start_ms=remap(w.start_ms),
                                 end_ms=remap(w.end_ms),
                                 score=w.score, phones=phones))
    return AlignedTurn(tier=turn.tier, audio_hash=turn.audio_hash,
                       duration_ms=duration_ms, offset_ms=off,
                       confidence=turn.confidence, words=words)


# ═══════════════════════════════════════════
# Tier 1 — MMS_FA forced alignment
# ═══════════════════════════════════════════

def _align_mms_fa(wav_path: str, text: str) -> Optional[AlignedTurn]:
    """CTC forced alignment via torchaudio.pipelines.MMS_FA. Returns
    None (→ fall through to Tier 3) when torch/torchaudio is missing
    or the model bundle cannot be loaded. CPU-safe on arm64."""
    try:
        import torch
        import torchaudio
        from torchaudio.pipelines import MMS_FA as bundle
        from engine.device import resolve_device  # seeds + MPS fallback
    except Exception:
        return None
    try:
        device = torch.device("cpu")   # MMS_FA is CPU-fast; determinism first
        model = _mms_model(bundle, device)
        tokenizer = bundle.get_tokenizer()
        aligner = bundle.get_aligner()

        wav, sr = torchaudio.load(wav_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != bundle.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, bundle.sample_rate)
            sr = bundle.sample_rate

        roman = romanize(text)
        words_raw = [w for w in re.findall(r"[a-zA-Z']+", roman.lower()) if w]
        if not words_raw:
            return None

        with torch.inference_mode():
            emission, _ = model(wav.to(device))
            token_spans = aligner(emission[0], tokenizer(words_raw))

        n_frames = emission.shape[1]
        total_ms = wav.shape[1] / sr * 1000.0
        frame_ms = total_ms / n_frames

        out_words: List[AlignedWord] = []
        for word, spans in zip(words_raw, token_spans):
            if not spans:
                continue
            w_start = spans[0].start * frame_ms
            w_end = spans[-1].end * frame_ms
            score = float(np.mean([s.score for s in spans]))
            # Distribute the word's char-span timings onto phoneme clusters
            clusters = phonemize(word)
            phones: List[Tuple[str, str, float, float]] = []
            if clusters:
                # char-proportional: each cluster takes time ∝ its char count
                lens = np.array([len(g) for g, _ in clusters], dtype=np.float64)
                cum = np.concatenate([[0.0], np.cumsum(lens)]) / lens.sum()
                for (g, vis), a, b in zip(clusters, cum[:-1], cum[1:]):
                    phones.append((g, vis.value,
                                   w_start + (w_end - w_start) * a,
                                   w_start + (w_end - w_start) * b))
            out_words.append(AlignedWord(text=word, start_ms=w_start,
                                         end_ms=w_end, score=score,
                                         phones=phones))
        if not out_words:
            return None
        conf = float(np.mean([w.score for w in out_words]))
        return AlignedTurn(tier=1, audio_hash="", duration_ms=total_ms,
                           offset_ms=0.0, confidence=conf, words=out_words)
    except Exception:
        return None


_MMS_MODEL = None


def _mms_model(bundle, device):
    global _MMS_MODEL
    if _MMS_MODEL is None:
        _MMS_MODEL = bundle.get_model(with_star=False).to(device).eval()
    return _MMS_MODEL


# ═══════════════════════════════════════════
# Tier 3 — even-split fallback (retained, offset-corrected)
# ═══════════════════════════════════════════

def _align_even_split(wav_path: str, text: str) -> AlignedTurn:
    """Last resort: distribute phoneme clusters evenly across the
    non-silent span of the wav. Tier-2 calibration still applies."""
    x, sr = load_wav_mono(wav_path)
    duration_ms = len(x) / sr * 1000.0
    db = rms_db_track(x, sr)
    voiced = np.where(db > SILENCE_DB)[0]
    if len(voiced):
        speech_a = float(voiced[0]) * ENV_HOP_MS
        speech_b = float(voiced[-1]) * ENV_HOP_MS
    else:
        speech_a, speech_b = 0.0, duration_ms

    roman = romanize(text)
    tokens = [t for t in re.findall(r"[\w']+", roman) if t]
    all_clusters: List[Tuple[str, List[Tuple[str, V]]]] = []
    total_units = 0
    for tok in tokens:
        cl = phonemize(tok)
        if not cl:
            cl = [(tok, v) for v in g2p(tok)][:1] or [(tok, V.REST)]
        all_clusters.append((tok, cl))
        total_units += sum(1.6 if v in
                           (V.OPEN_A, V.MID_E, V.CLOSED_I,
                            V.ROUNDED_TENSE, V.ROUNDED_LAX) else 1.0
                           for _, v in cl) + 0.6  # word gap share

    span = max(1.0, speech_b - speech_a)
    unit_ms = span / max(1.0, total_units)
    t = speech_a
    words: List[AlignedWord] = []
    for tok, cl in all_clusters:
        w_start = t
        phones: List[Tuple[str, str, float, float]] = []
        for g, vis in cl:
            w_ = 1.6 if vis in (V.OPEN_A, V.MID_E, V.CLOSED_I,
                                V.ROUNDED_TENSE, V.ROUNDED_LAX) else 1.0
            d = unit_ms * w_
            phones.append((g, vis.value, t, t + d))
            t += d
        words.append(AlignedWord(text=tok, start_ms=w_start, end_ms=t,
                                 score=1.0, phones=phones))
        t += unit_ms * 0.6
    return AlignedTurn(tier=3, audio_hash="", duration_ms=duration_ms,
                       offset_ms=0.0, confidence=1.0, words=words)


# ═══════════════════════════════════════════
# Public entry — cached, tiered, calibrated
# ═══════════════════════════════════════════

def align_turn(wav_path: str, text: str,
               cache_dir: Optional[str] = None) -> AlignedTurn:
    """THE timing source. Tier 1 → (fallback Tier 3), then Tier 2
    calibration on top of whichever tier produced events. Result is
    cached by audio-hash + text-hash + schema version: once per turn,
    ever."""
    ah = audio_hash(wav_path)
    th = hashlib.sha256(text.encode()).hexdigest()[:12]
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(
            cache_dir, f"{ah}-{th}-v{ALIGN_SCHEMA_VERSION}.align.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return AlignedTurn.from_dict(json.load(f))

    turn = _align_mms_fa(wav_path, text)
    if turn is None:
        turn = _align_even_split(wav_path, text)
    turn.audio_hash = ah
    turn = apply_calibration(turn, wav_path)

    if cache_path:
        tmp = cache_path + ".part"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(turn.to_dict(), f)
        os.replace(tmp, cache_path)
    return turn


__all__ = ["AlignedTurn", "AlignedWord", "align_turn", "apply_calibration",
           "phonemize", "romanize", "global_offset_ms", "dtw_warp",
           "onset_envelope", "predicted_jaw_envelope", "load_wav_mono",
           "audio_hash", "rms_db_track", "ALIGN_SCHEMA_VERSION"]
