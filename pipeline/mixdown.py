"""
JEEVidya V5 — The Mixing Console
════════════════════════════════
Raw edge-tts sounds like a phone call. Broadcast audio is a CHAIN, and
this module is that chain, sample-accurate on the Tier 0 timeline grid:

  VOICE BUS   high-pass 75 Hz → presence bell +3.5 dB @ 3 kHz → air
              shelf → frame-based soft-knee compression (12 ms attack /
              140 ms release) → convolution room reverb (synthesized
              seeded IR, 8% wet). The voice suddenly has a BODY and a
              ROOM.
  SOUND DESIGN  events planned from the timeline itself:
              whoosh on every shot change (PANNED in the cut direction),
              riser landing exactly at the explanation reveal, bass
              drop ON the reveal, pops on emphasis-word timestamps,
              achievement arp on the post-reveal beat.
  MUSIC BUS   per-DNA generative bed, side-chain ducked by the voice
              envelope, Haas-widened (9 ms L/R) into real stereo width
              while the voice stays dead-center mono-compatible.
  MASTER      approximate-LUFS normalize to −14 (YouTube target) +
              tanh soft ceiling.

Voice is PLACED at each turn's exact start_ms offset in a preallocated
buffer — not concatenated — so mix drift is structurally zero, matching
the video's frame grid. Everything seeded → bit-identical remixes.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import settings

MIX_VERSION = "mix-v5-console"
SR = 44100


# ═══════════════════════════════════════════
# DECODE / ENCODE
# ═══════════════════════════════════════════

def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def _decode(path: str) -> np.ndarray:
    """Any audio file → float32 mono @ SR.

    Decodes via a direct ffmpeg subprocess (raw s16le pipe) so pydub's
    ffprobe dependency is never needed — imageio_ffmpeg bundles only
    ffmpeg, not ffprobe.
    """
    import subprocess
    try:
        proc = subprocess.run(
            [_ffmpeg_exe(), "-v", "error", "-i", path,
             "-f", "s16le", "-acodec", "pcm_s16le",
             "-ar", str(SR), "-ac", "1", "pipe:1"],
            capture_output=True, check=True,
        )
        arr = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32)
        return arr / 32768.0
    except (subprocess.CalledProcessError, OSError):
        # Fallback: pydub direct load (works when a real ffprobe exists)
        from pydub import AudioSegment
        seg = AudioSegment.from_file(path).set_frame_rate(SR).set_channels(1)
        arr = np.array(seg.get_array_of_samples(), dtype=np.float32)
        return arr / float(1 << (8 * seg.sample_width - 1))


def _encode_stereo(left: np.ndarray, right: np.ndarray, out_path: str) -> str:
    from pydub import AudioSegment
    inter = np.empty(len(left) * 2, dtype=np.int16)
    inter[0::2] = (np.clip(left, -1, 1) * 32767).astype(np.int16)
    inter[1::2] = (np.clip(right, -1, 1) * 32767).astype(np.int16)
    seg = AudioSegment(data=inter.tobytes(), sample_width=2,
                       frame_rate=SR, channels=2)
    seg.export(out_path, format="mp3", bitrate="256k")
    return out_path


# ═══════════════════════════════════════════
# VOICE BUS
# ═══════════════════════════════════════════

def eq_voice(x: np.ndarray) -> np.ndarray:
    """Broadcast voice EQ as one FFT gain curve."""
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    gain = np.ones_like(f)
    gain *= 1.0 / (1.0 + (75.0 / np.maximum(f, 1.0)) ** 4)          # HP 75Hz
    gain *= 1.0 + 0.5 * np.exp(-((np.log2(np.maximum(f, 1) / 3000)) ** 2))  # presence
    gain *= 1.0 + 0.18 / (1.0 + (9000.0 / np.maximum(f, 1.0)) ** 4)  # air
    return np.fft.irfft(spec * gain, n=len(x)).astype(np.float32)


def compress_voice(x: np.ndarray, threshold_db: float = -21.0,
                   ratio: float = 2.6, attack_ms: float = 12.0,
                   release_ms: float = 140.0) -> np.ndarray:
    """Frame-based soft-knee compressor (10 ms hops, smoothed gains)."""
    hop = SR // 100
    n_frames = max(1, len(x) // hop)
    rms = np.sqrt(np.mean(
        x[:n_frames * hop].reshape(n_frames, hop) ** 2, axis=1) + 1e-12)
    level_db = 20 * np.log10(rms + 1e-9)

    over = np.maximum(0.0, level_db - threshold_db)
    knee = 6.0
    soft = np.where(over < knee, over * over / (2 * knee), over - knee / 2)
    gain_db = -soft * (1.0 - 1.0 / ratio)

    # Attack/release smoothing across frames (10 ms per frame)
    a_coef = math.exp(-10.0 / attack_ms)
    r_coef = math.exp(-10.0 / release_ms)
    smoothed = np.empty_like(gain_db)
    g = 0.0
    for i, target in enumerate(gain_db):        # ~6k iterations: cheap
        c = a_coef if target < g else r_coef
        g = c * g + (1 - c) * target
        smoothed[i] = g

    gains = np.repeat(10 ** (smoothed / 20), hop)
    gains = np.pad(gains, (0, len(x) - len(gains)), mode="edge")
    makeup = 10 ** (min(8.0, -np.median(smoothed) + 3.0) / 20)
    return (x * gains * makeup).astype(np.float32)


def room_reverb(x: np.ndarray, wet: float = 0.08, seed: int = 17,
                decay_s: float = 0.22, predelay_ms: float = 16.0
                ) -> np.ndarray:
    """Convolution reverb with a synthesized impulse response —
    exponentially-decaying seeded noise, low-cut so the room never muds
    the voice. FFT convolution: fast on minute-long buffers."""
    n_ir = int(SR * decay_s)
    rng = np.random.default_rng(seed)
    ir = rng.standard_normal(n_ir).astype(np.float32) \
        * np.exp(-np.arange(n_ir) / (SR * decay_s / 5.5))
    # Low-cut the IR (rooms flatter voices below ~300 Hz)
    spec = np.fft.rfft(ir)
    f = np.fft.rfftfreq(n_ir, 1 / SR)
    spec *= 1.0 / (1.0 + (300.0 / np.maximum(f, 1.0)) ** 2)
    ir = np.fft.irfft(spec, n=n_ir).astype(np.float32)
    ir /= (np.abs(ir).sum() + 1e-9)
    ir = np.concatenate([np.zeros(int(SR * predelay_ms / 1000),
                                  dtype=np.float32), ir])

    n = len(x) + len(ir) - 1
    nfft = 1 << (n - 1).bit_length()
    tail = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(ir, nfft),
                        n=nfft)[:len(x)].astype(np.float32)
    return x * (1 - wet * 0.4) + tail * wet


def master_voice_bus(x: np.ndarray) -> np.ndarray:
    return room_reverb(compress_voice(eq_voice(x)))


# ═══════════════════════════════════════════
# SOUND DESIGN (planned from the timeline itself)
# ═══════════════════════════════════════════

def plan_events(timeline, turn_data: List[Dict[str, Any]]
                ) -> List[Tuple[int, str, float]]:
    """(ms, sfx_kind, pan −1..1) — every event earns its place."""
    from engine.render_fast import _is_emphasis_word

    events: List[Tuple[int, str, float]] = []
    prev_shot: Optional[str] = None
    explain_idx = None
    for i, span in enumerate(timeline.spans):
        turn = span.turn
        shot = turn.get("shot_type") or "two_shot"
        if prev_shot is not None and shot != prev_shot and span.start_ms > 0:
            pan = 0.5 if (i % 2 == 0) else -0.5          # alternate sweep
            events.append((span.start_ms, "whoosh", pan))
        prev_shot = shot
        if turn.get("speaker") == "explanation" and explain_idx is None:
            explain_idx = i
            events.append((max(0, span.start_ms - 1500), "riser", 0.0))
            events.append((span.start_ms, "bass_drop", 0.0))
        if explain_idx is not None and i == explain_idx + 1:
            events.append((span.start_ms + 120, "achievement", 0.0))

    # Emphasis pops on number/unit words (capped so it never gets busy)
    pops = 0
    for span in timeline.spans:
        for w in span.words:
            if pops >= 8:
                break
            if _is_emphasis_word(w.text):
                events.append((w.start_ms, "pop", 0.2 if pops % 2 else -0.2))
                pops += 1
    events.sort(key=lambda e: e[0])
    return events


def render_events(events: List[Tuple[int, str, float]], total_ms: int
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Synthesize + place every event into a stereo pair (constant-power
    panning). Deterministic — audio_forge synths are seeded."""
    from tools.audio_forge import (master, sfx_achievement, sfx_bass_drop,
                                   sfx_pop, sfx_riser, sfx_whoosh)
    synth = {"whoosh": sfx_whoosh, "pop": sfx_pop, "riser": sfx_riser,
             "bass_drop": sfx_bass_drop, "achievement": sfx_achievement}
    gain = {"whoosh": 0.30, "pop": 0.26, "riser": 0.30,
            "bass_drop": 0.42, "achievement": 0.30}

    n = int(total_ms * SR / 1000)
    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)
    cache: Dict[str, np.ndarray] = {}
    for ms, kind, pan in events:
        x = cache.get(kind)
        if x is None:
            x = cache[kind] = master(synth[kind]()) if kind in synth else None
        if x is None:
            continue
        start = int(ms * SR / 1000)
        end = min(n, start + len(x))
        if end <= start:
            continue
        theta = (pan + 1) * math.pi / 4                  # constant power
        chunk = x[:end - start] * gain[kind]
        left[start:end] += chunk * math.cos(theta)
        right[start:end] += chunk * math.sin(theta)
    return left, right


# ═══════════════════════════════════════════
# THE MIXDOWN
# ═══════════════════════════════════════════

def mixdown(turn_data: List[Dict[str, Any]], out_path: str,
            dna=None) -> str:
    """Voice bus + sound design + ducked stereo bed → mastered stereo mix.
    Sample-accurate: every turn lands at its exact timeline offset."""
    from pipeline.timeline import Timeline
    from tools.audio_forge import forge_bgm, loudness_lufs

    timeline = Timeline(turn_data, fps=settings.FPS)
    total_ms = max(1000, timeline.total_ms)
    n = int(total_ms * SR / 1000)

    # ── VOICE: placed at exact offsets (zero structural drift) ──
    voice = np.zeros(n, dtype=np.float32)
    for span in timeline.spans:
        audio = span.turn.get("audio")
        if audio and os.path.exists(audio):
            x = _decode(audio)
            start = int(span.start_ms * SR / 1000)
            end = min(n, start + len(x))
            voice[start:end] += x[:end - start]
    voice = master_voice_bus(voice)

    # ── SOUND DESIGN ──
    sfx_l, sfx_r = render_events(plan_events(timeline, turn_data), total_ms)

    # ── MUSIC: per-DNA bed, envelope-ducked, Haas-widened ──
    bed = forge_bgm(dna, seconds=total_ms / 1000 + 1)[:n]
    if len(bed) < n:
        bed = np.pad(bed, (0, n - len(bed)))
    hop = SR // 20                                        # 50 ms envelope
    n_frames = max(1, n // hop)
    env = np.sqrt(np.mean(
        voice[:n_frames * hop].reshape(n_frames, hop) ** 2, axis=1))
    duck = 1.0 / (1.0 + (env / 0.02) ** 2) * 0.55 + 0.45  # −7 dB under speech
    duck = np.repeat(duck, hop)
    duck = np.pad(duck, (0, n - len(duck)), mode="edge")
    # smooth the duck (30 ms) so it breathes instead of pumping
    k = int(SR * 0.03)
    kernel = np.ones(k, dtype=np.float32) / k
    duck = np.convolve(duck, kernel, mode="same")
    bed *= duck * 0.9

    haas = int(SR * 0.009)
    bed_r = np.concatenate([np.zeros(haas, dtype=np.float32), bed])[:n]

    # ── SUM + MASTER (−14 LUFS on the mid channel) ──
    left = voice + bed + sfx_l
    right = voice + bed_r + sfx_r
    mid = (left + right) * 0.5
    g = 10 ** ((-14.0 - loudness_lufs(mid)) / 20)
    left = np.tanh(left * g * 0.92) / 0.92
    right = np.tanh(right * g * 0.92) / 0.92

    return _encode_stereo(left, right, out_path)
