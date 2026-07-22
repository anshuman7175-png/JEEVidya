"""Prosody extraction (C7): F0 + RMS energy from audio, cached by content hash.

Tries parselmouth (Praat) when installed; otherwise a numpy fallback
(autocorrelation pitch + windowed RMS) that is fully adequate for driving
head motion, breathing modulation and pause detection."""
from __future__ import annotations

import hashlib
import os
import wave
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

HOP_MS = 10.0
WIN_MS = 25.0
F0_MIN, F0_MAX = 75.0, 400.0
SILENCE_RMS = 0.06


@dataclass
class ProsodyTrack:
    hop_ms: float
    rms: np.ndarray            # normalized 0..1
    f0: np.ndarray             # Hz, 0 where unvoiced
    duration_ms: float

    def _idx(self, t_ms: float) -> int:
        return int(min(max(t_ms, 0.0) / self.hop_ms, len(self.rms) - 1))

    def energy_at(self, t_ms: float) -> float:
        return float(self.rms[self._idx(t_ms)]) if len(self.rms) else 0.0

    def pitch_at(self, t_ms: float) -> float:
        return float(self.f0[self._idx(t_ms)]) if len(self.f0) else 0.0

    def is_silent(self, t_ms: float, thresh: float = SILENCE_RMS) -> bool:
        return self.energy_at(t_ms) < thresh

    def silence_gaps(self, min_ms: float = 300.0,
                     thresh: float = SILENCE_RMS) -> List[Tuple[float, float]]:
        """Find all contiguous silence regions >= min_ms."""
        gaps: List[Tuple[float, float]] = []
        start: Optional[float] = None
        for i, v in enumerate(self.rms):
            t = i * self.hop_ms
            if v < thresh:
                if start is None:
                    start = t
            else:
                if start is not None and t - start >= min_ms:
                    gaps.append((start, t))
                start = None
        if start is not None and self.duration_ms - start >= min_ms:
            gaps.append((start, self.duration_ms))
        return gaps

    def emphasis_peaks(self, threshold: float = 0.7,
                       min_gap_ms: float = 260.0) -> List[float]:
        """RMS local maxima above threshold, spaced >= min_gap_ms apart."""
        peaks: List[float] = []
        last = -1e9
        r = self.rms
        for i in range(1, len(r) - 1):
            t = i * self.hop_ms
            if (r[i] >= threshold and r[i] >= r[i - 1]
                    and r[i] >= r[i + 1] and t - last >= min_gap_ms):
                peaks.append(t)
                last = t
        return peaks


def _read_wav(path: str) -> Tuple[np.ndarray, int]:
    """Read a WAV file into a mono float32 array + sample rate."""
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
        width = w.getsampwidth()
        ch = w.getnchannels()
    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


def _fallback_prosody(samples: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pure-numpy F0 + RMS extraction (no external deps)."""
    hop = int(sr * HOP_MS / 1000.0)
    win = int(sr * WIN_MS / 1000.0)
    n_frames = max(1, (len(samples) - win) // hop + 1)
    rms = np.zeros(n_frames, dtype=np.float32)
    f0 = np.zeros(n_frames, dtype=np.float32)
    lag_min = int(sr / F0_MAX)
    lag_max = min(int(sr / F0_MIN), win - 1)
    hann = np.hanning(win)
    for i in range(n_frames):
        frame = samples[i * hop: i * hop + win]
        if len(frame) < win:
            frame = np.pad(frame, (0, win - len(frame)))
        rms[i] = float(np.sqrt(np.mean(frame ** 2)))
        fw = frame * hann
        if rms[i] > 0.01 and lag_max > lag_min:
            spec = np.fft.rfft(fw, n=2 * win)
            ac = np.fft.irfft(spec * np.conj(spec))[:win]
            if ac[0] > 1e-9:
                seg = ac[lag_min:lag_max]
                peak = int(np.argmax(seg)) + lag_min
                if ac[peak] / ac[0] > 0.30:          # voicing clarity gate
                    f0[i] = sr / peak
    peak_rms = float(rms.max()) or 1.0
    return rms / peak_rms, f0


def extract_prosody(wav_path: str,
                    cache_dir: Optional[str] = None) -> ProsodyTrack:
    """Extract prosody from a WAV file. Cached by content hash."""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        with open(wav_path, "rb") as f:
            digest = hashlib.sha1(f.read()).hexdigest()[:16]
        cache_path = os.path.join(cache_dir, f"prosody_{digest}.npz")
        if os.path.exists(cache_path):
            z = np.load(cache_path)
            return ProsodyTrack(HOP_MS, z["rms"], z["f0"], float(z["dur"]))

    samples, sr = _read_wav(wav_path)
    try:
        import parselmouth  # type: ignore
        snd = parselmouth.Sound(samples, sampling_frequency=sr)
        pitch = snd.to_pitch(time_step=HOP_MS / 1000.0,
                             pitch_floor=F0_MIN, pitch_ceiling=F0_MAX)
        f0 = np.nan_to_num(
            pitch.selected_array["frequency"]).astype(np.float32)
        intensity = snd.to_intensity(time_step=HOP_MS / 1000.0)
        rms = np.asarray(intensity.values[0], dtype=np.float32)
        rms = np.clip((rms - rms.min()) / max(1e-6, rms.max() - rms.min()),
                      0, 1)
        n = min(len(rms), len(f0))
        rms, f0 = rms[:n], f0[:n]
    except Exception:
        rms, f0 = _fallback_prosody(samples, sr)

    dur = len(samples) / sr * 1000.0
    if cache_dir:
        np.savez_compressed(cache_path, rms=rms, f0=f0, dur=dur)
    return ProsodyTrack(HOP_MS, rms, f0, dur)


def synthetic_prosody(word_timings, duration_ms: float,
                      seed: str = "prosody") -> ProsodyTrack:
    """When no audio exists (silent preview renders) -- build a plausible RMS
    track from word timings so all downstream systems still get life signals."""
    import random
    rng = random.Random(seed)
    n = int(duration_ms / HOP_MS) + 1
    rms = np.zeros(n, dtype=np.float32)
    f0 = np.zeros(n, dtype=np.float32)
    for w in word_timings:
        amp = rng.uniform(0.45, 1.0)
        i0, i1 = int(w.start_ms / HOP_MS), int(w.end_ms / HOP_MS)
        for i in range(max(0, i0), min(n, i1)):
            p = (i - i0) / max(1, i1 - i0)
            rms[i] = max(rms[i], amp * float(np.sin(np.pi * p) ** 0.6))
            f0[i] = 140 + 60 * amp
    return ProsodyTrack(HOP_MS, rms, f0, duration_ms)
