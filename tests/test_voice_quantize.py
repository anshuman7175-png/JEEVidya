"""Tier 0 — Frame quantization: the property that makes segments cacheable.

Every turn's total duration (audio + padding) must be an exact whole
number of frames. This is what guarantees:
  1. drift = 0 ms forever (all boundaries are exact frames), and
  2. a turn's frame count depends only on its OWN audio — editing one
     dialogue line never shifts any neighbouring segment's cache key.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from pipeline.voice import (FRAME_QUANTUM_MS, quantize_padding, quantize_up,
                            tts_cache_key)


def test_quantum_is_an_exact_number_of_frames():
    # quantum ms × fps must be a whole number of frames × 1000
    assert (FRAME_QUANTUM_MS * settings.FPS) % 1000 == 0


def test_padding_lands_on_frame_boundary():
    base = settings.INTER_TURN_PADDING_MS
    for audio_ms in (0, 1, 99, 100, 999, 1234, 4231, 59_999, 3_600_001):
        pad = quantize_padding(audio_ms, base)
        assert pad >= base                                # never shorter
        assert pad < base + FRAME_QUANTUM_MS              # never a full extra quantum
        assert (audio_ms + pad) % FRAME_QUANTUM_MS == 0   # exact frame multiple


def test_quantize_up():
    assert quantize_up(0) == 0
    assert quantize_up(1) == FRAME_QUANTUM_MS
    assert quantize_up(FRAME_QUANTUM_MS) == FRAME_QUANTUM_MS
    assert quantize_up(FRAME_QUANTUM_MS + 1) == 2 * FRAME_QUANTUM_MS


def test_sum_of_quantized_turns_is_exact_frames():
    """Cumulative boundaries of quantized turns are always exact frames —
    the structural proof that per-turn segments reassemble drift-free."""
    durations = [quantize_up(d) for d in (1234, 5678, 999, 41, 30_000)]
    cursor = 0
    for d in durations:
        cursor += d
        assert (cursor * settings.FPS) % 1000 == 0        # exact frame boundary


def test_tts_key_stability_and_sensitivity():
    a = tts_cache_key("gravity kya hai?", "boy")
    assert a == tts_cache_key("gravity kya hai?", "boy")       # deterministic
    assert a == tts_cache_key("  gravity kya hai?  ", "boy")   # whitespace-stable
    assert a != tts_cache_key("gravity kya hai!", "boy")       # text-sensitive
    assert a != tts_cache_key("gravity kya hai?", "girl")      # voice-sensitive
