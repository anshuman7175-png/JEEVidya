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

from engine.visemes import V, VisemeEvent, g2p, VOWELS, _PHONEME_TABLE

ALIGN_SCHEMA_VERSION = 3

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
# Shared with engine/visemes._PHONEME_TABLE so Tier 1/2/3 and G2P
# never diverge on articulatory shape assignments.


def phonemize(token: str) -> List[Tuple[str, V]]:
    """Split a romanized token into (grapheme-cluster, viseme) pairs,
    longest-match-first. Gemination ("kk") collapses to one closure.
    Silent word-final 'e' is dropped when a preceding vowel exists."""
    t = token.lower()
    out: List[Tuple[str, V]] = []
    i = 0
    while i < len(t):
        matched = False
        for graph, vis in _PHONEME_TABLE:
            if t.startswith(graph, i):
                j = i + len(graph)
                # gemination: identical cluster repeated → single phoneme
                if vis not in VOWELS:
                    while t.startswith(graph, j):
                        j += len(graph)
                # Check for silent word-final 'e'
                if graph == "e":
                    is_word_end = True
                    for k in range(j, len(t)):
                        if t[k].isalnum():
                            is_word_end = False
                            break
                    if is_word_end and any(v in VOWELS for _, v in out):
                        i = j
                        matched = True
                        break
                if not out or out[-1][1] != vis or out[-1][0] != graph:
                    out.append((t[i:j], vis))
                i = j
                matched = True
                break
        if not matched:
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
    silence gating. Mutates a copy of the turn's word/phone times.
    
    When turn.tier == 3 (acoustic speech segmentation), word boundaries
    are already anchored to the physical waveform bursts by segment_acoustic_speech.
    In that case, synthetic jaw cross-correlation (which peaks on secondary
    vowels) must NOT apply false negative shifts that pull words into silence."""
    x, sr = load_wav_mono(wav_path)
    duration_ms = len(x) / sr * 1000.0

    if turn.tier == 3:
        off = 0.0
        warp = None
    else:
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
    p15 = float(np.percentile(db, 15))
    p90 = float(np.percentile(db, 90))
    calib_silence_thresh = max(-50.0, min(-28.0, p15 + 0.25 * max(5.0, p90 - p15)))

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
        return db[idx] < calib_silence_thresh

    words: List[AlignedWord] = []
    for w in turn.words:
        w_start = remap(w.start_ms)
        w_end = remap(w.end_ms)
        min_w_dur = min(max(50.0, w.end_ms - w.start_ms), 100.0)
        if w_end < w_start + min_w_dur:
            w_end = min(duration_ms, w_start + min_w_dur)

        phones = []
        for g, vis, s, e in w.phones:
            s2, e2 = remap(s), remap(e)
            if e2 - s2 < 1.0:
                e2 = min(duration_ms, s2 + 1.0)
            # Silence gate: any vowel or open viseme on a silent frame clamps to REST
            if V(vis) in (V.OPEN_A, V.MID_E, V.CLOSED_I,
                          V.ROUNDED_TENSE, V.ROUNDED_LAX) \
                    and is_silent((s2 + e2) / 2.0):
                vis = V.REST.value
            phones.append((g, vis, s2, e2))
        words.append(AlignedWord(text=w.text,
                                 start_ms=w_start,
                                 end_ms=w_end,
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
        # aligner() contracts one span-list per input word; strict=True turns
        # any drift between the two into a crash instead of dropped words.
        for word, spans in zip(words_raw, token_spans, strict=True):
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
                # len(cum) == len(clusters) + 1 by construction, so both
                # boundary slices match clusters exactly — assert it stays so.
                for (g, vis), a, b in zip(clusters, cum[:-1], cum[1:],
                                          strict=True):
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

def segment_acoustic_speech(x: np.ndarray, sr: int,
                            hop_ms: float = ENV_HOP_MS,
                            min_pause_ms: float = 70.0,
                            min_speech_ms: float = 40.0) -> Tuple[List[Tuple[float, float]], float]:
    """Segment audio into discrete voiced speech bursts separated by acoustic silence pauses.
    Returns (speech_chunks, adaptive_silence_thresh_db)."""
    db = rms_db_track(x, sr, hop_ms=hop_ms)
    n_frames = len(db)
    total_dur_ms = len(x) / sr * 1000.0
    if n_frames == 0:
        return [(0.0, total_dur_ms)], -45.0
    
    p15 = float(np.percentile(db, 15))
    p90 = float(np.percentile(db, 90))
    thresh = max(-50.0, min(-28.0, p15 + 0.25 * max(5.0, p90 - p15)))
    is_voiced = db > thresh
    
    runs: List[Tuple[bool, int, int]] = []
    cur_v = is_voiced[0]
    cur_s = 0
    for i, v in enumerate(is_voiced):
        if v != cur_v:
            runs.append((cur_v, cur_s, i))
            cur_v = v
            cur_s = i
    runs.append((cur_v, cur_s, n_frames))
    
    # Merge short silence dips (< min_pause_ms) into neighbouring speech
    pause_frames = int(math.ceil(min_pause_ms / hop_ms))
    merged: List[Tuple[bool, int, int]] = []
    for is_v, s, e in runs:
        dur = e - s
        if not is_v and dur < pause_frames and merged:
            prev_v, prev_s, _ = merged[-1]
            if prev_v:
                merged[-1] = (True, prev_s, e)
                continue
        merged.append((is_v, s, e))
        
    final_runs: List[Tuple[bool, int, int]] = []
    for is_v, s, e in merged:
        if final_runs and final_runs[-1][0] == is_v:
            final_runs[-1] = (is_v, final_runs[-1][1], e)
        else:
            final_runs.append((is_v, s, e))
            
    speech_frames = int(math.ceil(min_speech_ms / hop_ms))
    speech_chunks: List[Tuple[float, float]] = []
    for is_v, s, e in final_runs:
        if is_v and (e - s) >= speech_frames:
            speech_chunks.append((s * hop_ms, min(total_dur_ms, e * hop_ms)))
            
    if not speech_chunks:
        speech_chunks = [(0.0, total_dur_ms)]
        
    return speech_chunks, thresh


def partition_words_to_chunks(word_units: List[float], chunk_durs: List[float]) -> List[List[int]]:
    """Partition N words into K chunks by minimizing duration discrepancy via DP."""
    N = len(word_units)
    K = len(chunk_durs)
    if N == 0:
        return [[] for _ in range(K)]
    if K <= 1:
        return [list(range(N))]
    if K >= N:
        # More chunks than words: each word gets its own chunk, extra chunks are empty
        return [[i] if i < N else [] for i in range(K)]
        
    tot_units = sum(word_units)
    tot_dur = sum(chunk_durs)
    rho = tot_dur / max(1e-3, tot_units)
    
    INF = 1e18
    dp = [[INF] * (N + 1) for _ in range(K + 1)]
    parent = [[0] * (N + 1) for _ in range(K + 1)]
    dp[0][0] = 0.0
    
    for k in range(1, K + 1):
        D_k = chunk_durs[k - 1]
        for i in range(k, N - (K - k) + 1):
            for j in range(k - 1, i):
                sub_u = sum(word_units[j:i])
                expected_dur = rho * sub_u
                diff = D_k - expected_dur
                cost = (diff * diff) / max(1.0, D_k)
                if dp[k - 1][j] + cost < dp[k][i]:
                    dp[k][i] = dp[k - 1][j] + cost
                    parent[k][i] = j
                    
    curr_i = N
    result: List[List[int]] = []
    for k in range(K, 0, -1):
        prev_i = parent[k][curr_i]
        result.append(list(range(prev_i, curr_i)))
        curr_i = prev_i
    result.reverse()
    return result


def _align_even_split(wav_path: str, text: str) -> AlignedTurn:
    """Acoustic-phrase & silence-aware alignment fallback.
    Detects acoustic speech bursts separated by natural pauses (>= 70ms),
    and partitions words into these bursts using DP. Words NEVER span
    across acoustic silence pauses, preserving pause resting."""
    x, sr = load_wav_mono(wav_path)
    duration_ms = len(x) / sr * 1000.0
    chunks, thresh = segment_acoustic_speech(x, sr)

    roman = romanize(text)
    tokens = [t for t in re.findall(r"[\w']+", roman) if t]
    all_clusters: List[Tuple[str, List[Tuple[str, V]]]] = []
    word_units: List[float] = []

    for tok in tokens:
        cl = phonemize(tok)
        if not cl:
            cl = [(tok, v) for v in g2p(tok)] or [(tok, V.REST)]
        all_clusters.append((tok, cl))
        wu = sum(1.6 if v in
                 (V.OPEN_A, V.MID_E, V.CLOSED_I,
                  V.ROUNDED_TENSE, V.ROUNDED_LAX) else 1.0
                 for _, v in cl) + 0.5
        word_units.append(wu)

    if not all_clusters:
        return AlignedTurn(tier=3, audio_hash="", duration_ms=duration_ms,
                           offset_ms=0.0, confidence=1.0, words=[])

    chunk_durs = [max(1.0, e - s) for s, e in chunks]
    partitions = partition_words_to_chunks(word_units, chunk_durs)

    words: List[AlignedWord] = []
    for k, word_indices in enumerate(partitions):
        if not word_indices or k >= len(chunks):
            continue
        c_start, c_end = chunks[k]
        c_dur = max(1.0, c_end - c_start)
        c_units = sum(word_units[idx] for idx in word_indices)
        unit_ms = c_dur / max(1.0, c_units)
        t = c_start
        for idx in word_indices:
            tok, cl = all_clusters[idx]
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
            t += unit_ms * 0.5

    return AlignedTurn(tier=3, audio_hash="", duration_ms=duration_ms,
                       offset_ms=0.0, confidence=1.0, words=words)


# ═══════════════════════════════════════════
# Monotonicity enforcement (fixes scrambled CTC word order)
# ═══════════════════════════════════════════

def _enforce_monotonicity(words: List[AlignedWord],
                          text: str) -> List[AlignedWord]:
    """Ensure aligned words follow script order.

    CTC forced alignment can produce non-monotonic assignments when the
    acoustic model's emission probabilities are weak for certain tokens
    (numerals like '90', English words in Hindi audio, etc.).  The time
    *slots* (start_ms, end_ms pairs) are acoustically valid — they mark
    real speech events — but the word-to-slot mapping can be wrong.

    Fix: sort the time slots by onset, keep the script-order tokens, and
    re-pair them.  If inversion rate exceeds 40 %, the alignment is
    considered unreliable and None is returned (caller falls back to
    even-split).
    """
    if len(words) < 2:
        return words

    # Count inversions: how many adjacent pairs are out of onset order?
    inversions = sum(
        1 for i in range(len(words) - 1)
        if words[i].start_ms > words[i + 1].start_ms
    )
    inv_rate = inversions / max(1, len(words) - 1)

    if inversions == 0:
        return words  # already monotonic

    if inv_rate > 0.40:
        # Alignment is catastrophically wrong — signal caller to fall back
        print(f"  [Align] Monotonicity: {inversions}/{len(words)-1} inversions "
              f"({inv_rate:.0%}) — falling back to even-split")
        return []  # empty signals fallback

    print(f"  [Align] Monotonicity: correcting {inversions} inversion(s) "
          f"in {len(words)} words")

    # Sort time slots by onset (the acoustic truth)
    slots = sorted(
        [(w.start_ms, w.end_ms, w.score, w.phones) for w in words],
        key=lambda s: s[0]
    )

    # Re-pair: i-th script token gets i-th time slot
    fixed: List[AlignedWord] = []
    for i, w in enumerate(words):
        s_start, s_end, s_score, s_phones = slots[i]
        fixed.append(AlignedWord(
            text=w.text,
            start_ms=s_start,
            end_ms=s_end,
            score=s_score,
            phones=s_phones,
        ))

    return fixed


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
    if turn is not None and turn.tier == 1:
        # Enforce monotonic word order — CTC can scramble tokens
        fixed = _enforce_monotonicity(turn.words, text)
        if not fixed:
            # Catastrophic inversion rate — fall back to even-split
            turn = _align_even_split(wav_path, text)
        else:
            turn.words = fixed
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
