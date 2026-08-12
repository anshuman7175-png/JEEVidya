"""
JEEVidya V5 — Audio Forge (Tier 2)
══════════════════════════════════
Synthesizes the channel's ENTIRE sound from numpy — no downloads:

  SFX   whoosh, pop, riser, bass_drop, achievement (the exact filenames
        pipeline/sfx.py SFXManager already expects)
  BGM   generative per-DNA lofi bed: chord pads + soft plucks, key/mode/
        tempo from the genes, loopable, seeded → deterministic
  Master approximate-LUFS normalization to −14 (YouTube's playback
        normalization target) with a soft-knee peak safety.

Run once:  python3 jvmake.py forge      → assets/sfx/*.mp3
Per-video: forge_bgm(dna, seconds)      → the factory's unique bed
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from config import settings

if TYPE_CHECKING:
    from engine.visual_dna import VisualDNA

SR = 44100
TARGET_LUFS = -14.0


# ═══════════════════════════════════════════
# PRIMITIVES
# ═══════════════════════════════════════════

def _t(seconds: float) -> np.ndarray:
    return np.arange(int(SR * seconds), dtype=np.float32) / SR


def _t_n(n: int) -> np.ndarray:
    """Time axis with EXACTLY n samples. Use whenever the buffer length is
    already known in samples — round-tripping through seconds
    (`_t(n / SR)`) can lose a sample to float rounding and crash a
    broadcast (`operands could not be broadcast together`)."""
    return np.arange(int(n), dtype=np.float32) / SR


def _env(n: int, attack: float, release: float, curve: float = 2.0) -> np.ndarray:
    """Attack/release envelope over n samples (fractions of total)."""
    env = np.ones(n, dtype=np.float32)
    a = max(1, int(n * attack))
    r = max(1, int(n * release))
    env[:a] = np.linspace(0, 1, a) ** curve
    env[-r:] = np.linspace(1, 0, r) ** curve
    return env


def _midi_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def _lowpass(x: np.ndarray, alpha: float) -> np.ndarray:
    """One-pole lowpass (cheap, musical)."""
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):          # small buffers only — SFX are short
        acc += alpha * (x[i] - acc)
        y[i] = acc
    return y


def _lowpass_fft(x: np.ndarray, cutoff_hz: float) -> np.ndarray:
    """FFT brick-wall-ish lowpass for long BGM buffers (fast)."""
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    spec *= 1.0 / (1.0 + (freqs / max(40.0, cutoff_hz)) ** 4)
    return np.fft.irfft(spec, n=len(x)).astype(np.float32)


def _highpass_fft(x: np.ndarray, cutoff_hz: float) -> np.ndarray:
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    spec *= 1.0 - 1.0 / (1.0 + (freqs / max(1.0, cutoff_hz)) ** 4)
    return np.fft.irfft(spec, n=len(x)).astype(np.float32)


def _noise(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1, 1, n).astype(np.float32)


# ═══════════════════════════════════════════
# MASTERING (approximate LUFS)
# ═══════════════════════════════════════════

def loudness_lufs(x: np.ndarray) -> float:
    """Approximate integrated loudness: K-weighting reduced to a highpass
    at ~100 Hz + mean-square over the program. Close enough to steer a
    mix to YouTube's −14 target without a full BS.1770 stack."""
    if len(x) == 0:
        return -70.0
    w = _highpass_fft(x, 100.0)
    ms = float(np.mean(w * w))
    return -0.691 + 10 * math.log10(max(ms, 1e-12))


def master(x: np.ndarray, target: float = TARGET_LUFS) -> np.ndarray:
    """Gain to target loudness, then soft-clip stray peaks (tanh knee)."""
    gain = 10 ** ((target - loudness_lufs(x)) / 20)
    y = x * gain
    # tanh has unity slope at 0, so small signals pass at target loudness
    # while peaks compress toward the 0.98 ceiling (never clips).
    return np.tanh(y) * 0.98


def _to_segment(x: np.ndarray):
    """numpy float [-1,1] → pydub AudioSegment (mono 16-bit, SR)."""
    from pydub import AudioSegment
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    return AudioSegment(data=pcm.tobytes(), sample_width=2,
                        frame_rate=SR, channels=1)


# ═══════════════════════════════════════════
# THE SFX LIBRARY
# ═══════════════════════════════════════════

def sfx_whoosh(seconds: float = 0.55, seed: int = 11) -> np.ndarray:
    """Filtered noise with a rising sweep — scene cuts, camera whips."""
    n = int(SR * seconds)
    noise = _noise(n, seed)
    sweep = np.linspace(300, 4200, n)
    # Ring-modulate noise into the sweep band, then band-shape
    carrier = np.sin(2 * np.pi * np.cumsum(sweep) / SR).astype(np.float32)
    x = _lowpass_fft(noise * carrier, 5000.0)
    return x * _env(n, 0.25, 0.45) * 0.9


def sfx_pop(seconds: float = 0.16, seed: int = 12) -> np.ndarray:
    """Sine blip with a pitch drop + click transient — UI/text pops."""
    n = int(SR * seconds)
    freq = np.linspace(900, 420, n)
    x = np.sin(2 * np.pi * np.cumsum(freq) / SR).astype(np.float32)
    x[:64] += _noise(64, seed) * 0.5                       # click
    return x * _env(n, 0.01, 0.7, curve=1.5)


def sfx_riser(seconds: float = 1.6, seed: int = 13) -> np.ndarray:
    """Tension riser into reveals: detuned saws sweeping up + noise swell."""
    n = int(SR * seconds)
    x = np.zeros(n, dtype=np.float32)
    for det in (0.0, 0.7, -0.7):
        freq = np.linspace(70, 460 + det * 8, n) * (1 + det * 0.004)
        phase = 2 * np.pi * np.cumsum(freq) / SR
        x += ((phase / np.pi) % 2 - 1) * 0.28              # saw
    x += _lowpass_fft(_noise(n, seed), 3000) * np.linspace(0, 0.7, n,
                                                           dtype=np.float32)
    x = _lowpass_fft(x, 2600)
    ramp = np.linspace(0.15, 1.0, n, dtype=np.float32) ** 1.4
    return x * ramp * _env(n, 0.02, 0.06)


def sfx_bass_drop(seconds: float = 0.9, seed: int = 14) -> np.ndarray:
    """80→32 Hz sine drop with a sub thump — the reveal landing."""
    n = int(SR * seconds)
    freq = np.linspace(82, 32, n)
    x = np.sin(2 * np.pi * np.cumsum(freq) / SR).astype(np.float32)
    x[: n // 8] *= np.linspace(1.6, 1.0, n // 8)           # punch
    return x * _env(n, 0.005, 0.5)


def sfx_achievement(seed: int = 15) -> np.ndarray:
    """Rising major arpeggio with sparkle — correct-answer moments."""
    notes = [76, 80, 83, 88]                                # E5 maj arp
    step = 0.09
    total = step * len(notes) + 0.5
    n = int(SR * total)
    x = np.zeros(n, dtype=np.float32)
    for i, m in enumerate(notes):
        start = int(SR * step * i)
        dur = int(SR * 0.5)
        t = _t(0.5)
        tone = (np.sin(2 * np.pi * _midi_hz(m) * t)
                + 0.4 * np.sin(2 * np.pi * _midi_hz(m + 12) * t))
        tone *= np.exp(-t * 6)
        x[start:start + dur] += tone[: min(dur, n - start)] * 0.5
    return x * _env(n, 0.005, 0.2)


# ═══════════════════════════════════════════
# GENERATIVE BGM (per-DNA lofi bed)
# ═══════════════════════════════════════════

_MODE_STEPS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
}
_PROGRESSIONS = [[0, 5, 3, 4], [0, 3, 4, 4], [5, 3, 0, 4], [0, 4, 5, 3]]


def _chord(root_midi: int, scale: List[int], degree: int) -> List[int]:
    """Diatonic triad + added 7th on a scale degree."""
    def note(d):
        octave, idx = divmod(degree + d, 7)
        return root_midi + scale[idx] + 12 * octave
    return [note(0), note(2), note(4), note(6)]


def forge_bgm(dna: Optional["VisualDNA"] = None,
              seconds: float = 60.0) -> np.ndarray:
    """Loopable chord-pad bed: key/mode/tempo from the DNA genes."""
    if dna is not None:
        root = int(dna.genes["bgm_root"])
        mode = dna.genes["bgm_mode"]
        tempo = dna.bgm_tempo
        seed = dna.seed & 0x7FFFFFFF
    else:
        root, mode, tempo, seed = 48, "minor", 84, 42

    rng = np.random.default_rng(seed)
    scale = _MODE_STEPS.get(mode, _MODE_STEPS["minor"])
    prog = _PROGRESSIONS[seed % len(_PROGRESSIONS)]
    bar_s = 4 * 60.0 / tempo
    n_total = int(SR * seconds)
    x = np.zeros(n_total, dtype=np.float32)

    bar = 0
    pos = 0
    while pos < n_total:
        chord = _chord(root, scale, prog[bar % len(prog)])
        n_bar = min(int(SR * bar_s), n_total - pos)
        if n_bar <= 0:
            break
        t = _t_n(n_bar)

        # Pad: detuned sines per chord tone, slow attack
        pad = np.zeros(n_bar, dtype=np.float32)
        for m in chord:
            f = _midi_hz(m)
            pad += (np.sin(2 * np.pi * f * t)
                    + 0.5 * np.sin(2 * np.pi * f * 1.003 * t)) * 0.11
        pad *= _env(n_bar, 0.25, 0.25, curve=1.5)

        # Pluck: sparse melodic dots from the chord, eighth-note grid
        pluck = np.zeros(n_bar, dtype=np.float32)
        eighth = int(SR * bar_s / 8)
        for k in range(8):
            if rng.random() < 0.35:
                m = int(rng.choice(chord)) + 12
                dur = min(eighth * 2, n_bar - k * eighth)
                if dur <= 0:
                    continue
                tt = _t_n(dur)
                tone = np.sin(2 * np.pi * _midi_hz(m) * tt) * np.exp(-tt * 5)
                pluck[k * eighth:k * eighth + dur] += tone * 0.16

        x[pos:pos + n_bar] += pad + pluck
        pos += n_bar
        bar += 1

    x = _lowpass_fft(x, 2400.0)                    # lofi warmth
    fade = int(SR * 1.5)
    x[:fade] *= np.linspace(0, 1, fade)
    x[-fade:] *= np.linspace(1, 0, fade)
    return master(x, target=-24.0)                 # bed sits under voice


def sidechain_duck(bgm, voice, threshold_db: float = -34.0,
                   duck_db: float = -10.0, chunk_ms: int = 120):
    """pydub-level sidechain: duck the bed wherever the voice speaks."""
    out = bgm[:0]
    pos = 0
    n = len(bgm)
    while pos < n:
        b = bgm[pos:pos + chunk_ms]
        v = voice[pos:pos + chunk_ms] if pos < len(voice) else None
        if v is not None and v.dBFS > threshold_db:
            b = b + duck_db
        out += b
        pos += chunk_ms
    return out.fade_in(60).fade_out(60)


# ═══════════════════════════════════════════
# LIBRARY FORGE (writes what SFXManager expects)
# ═══════════════════════════════════════════

def forge_library(out_dir: Optional[str] = None,
                  force: bool = False) -> Dict[str, str]:
    """Synthesize the whole SFX library + a default BGM into assets/sfx."""
    out_dir = out_dir or settings.SFX_DIR
    os.makedirs(out_dir, exist_ok=True)

    recipes = {
        "whoosh.mp3": lambda: master(sfx_whoosh()),
        "pop.mp3": lambda: master(sfx_pop()),
        "riser.mp3": lambda: master(sfx_riser()),
        "bass_drop.mp3": lambda: master(sfx_bass_drop()),
        "achievement.mp3": lambda: master(sfx_achievement()),
        "bgm_lofi.mp3": lambda: forge_bgm(None, seconds=64.0),
    }

    written: Dict[str, str] = {}
    for name, make in recipes.items():
        path = os.path.join(out_dir, name)
        if os.path.exists(path) and not force:
            written[name] = "kept"
            continue
        _to_segment(make()).export(path, format="mp3", bitrate="192k")
        written[name] = "forged"
        print(f"  [Forge] {name} ✓")
    return written
