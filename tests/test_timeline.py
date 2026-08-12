"""Tier 0 — Timeline: drift must be structurally zero, words speech-locked."""
import os
import random
from itertools import pairwise
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.timeline import Timeline, parse_vtt_words


def _turn(tid, ms, speaker="boy", vtt=None):
    return {"turn_id": tid, "speaker": speaker, "duration_ms": ms, "vtt": vtt}


# ─── Drift ──────────────────────────────────────────────────

def test_zero_drift_over_200_random_turns():
    """Property: with frame-quantized durations (voice.py guarantees
    multiples of FRAME_QUANTUM_MS), every turn boundary is an EXACT frame
    and cumulative A/V drift is 0 ms — for any number of turns."""
    random.seed(7)
    fps = 30
    turns = [_turn(i, random.randrange(100, 8000, 100)) for i in range(200)]
    tl = Timeline(turns, fps=fps)

    total_ms = sum(t["duration_ms"] for t in turns)
    assert tl.total_ms == total_ms
    assert tl.total_frames == round(total_ms * fps / 1000)

    for span in tl.spans:
        # boundary lands on an exact frame: frame*1000/fps == start_ms
        assert abs(span.start_frame * 1000 / fps - span.start_ms) < 1e-6
        assert abs(span.end_frame * 1000 / fps - span.end_ms) < 1e-6


def test_segments_tile_perfectly():
    """No gaps, no overlaps: per-turn segments must reassemble the video."""
    random.seed(21)
    turns = [_turn(i, random.randrange(100, 5000, 100)) for i in range(50)]
    tl = Timeline(turns, fps=30)
    assert sum(s.end_frame - s.start_frame for s in tl.spans) == tl.total_frames
    for prev, cur in pairwise(tl.spans):
        assert prev.end_frame == cur.start_frame


def test_span_at_frame_boundaries():
    tl = Timeline([_turn(1, 1000), _turn(2, 2000)], fps=30)
    assert tl.span_at_frame(0).turn["turn_id"] == 1
    assert tl.span_at_frame(29).turn["turn_id"] == 1
    assert tl.span_at_frame(30).turn["turn_id"] == 2
    assert tl.span_at_frame(89).turn["turn_id"] == 2


# ─── VTT → karaoke ──────────────────────────────────────────

_VTT = """WEBVTT

00:00:00.000 --> 00:00:00.400
hello

00:00:00.400 --> 00:00:00.900
world
"""


def test_parse_vtt_words(tmp_path):
    vtt = tmp_path / "t.vtt"
    vtt.write_text(_VTT, encoding="utf-8")
    words = parse_vtt_words(str(vtt))
    assert [w.text for w in words] == ["hello", "world"]
    assert words[0].start_ms == 0 and words[0].end_ms == 400
    assert words[1].start_ms == 400 and words[1].end_ms == 900


def test_parse_vtt_missing_file_is_empty():
    assert parse_vtt_words("/nonexistent/file.vtt") == []
    assert parse_vtt_words(None) == []


def test_words_promoted_to_global_time(tmp_path):
    vtt = tmp_path / "t.vtt"
    vtt.write_text(_VTT, encoding="utf-8")
    # Second turn carries the words → global offset = 1500 ms
    tl = Timeline([_turn(1, 1500), _turn(2, 1500, vtt=str(vtt))], fps=30)
    span = tl.spans[1]
    assert span.words[0].start_ms == 1500
    assert span.words[1].end_ms == 2400


def test_caption_chunks_stay_inside_span(tmp_path):
    vtt = tmp_path / "t.vtt"
    vtt.write_text(_VTT, encoding="utf-8")
    tl = Timeline([_turn(1, 1500, vtt=str(vtt))], fps=30)
    span = tl.spans[0]
    assert span.chunks, "words must produce caption chunks"
    for c in span.chunks:
        assert span.start_ms <= c.start_ms < c.end_ms <= span.end_ms


def test_active_caption_highlights_spoken_word(tmp_path):
    vtt = tmp_path / "t.vtt"
    vtt.write_text(_VTT, encoding="utf-8")
    tl = Timeline([_turn(1, 1500, vtt=str(vtt))], fps=30)
    span = tl.spans[0]

    chunk, active = tl.active_caption(span, 100)     # inside "hello"
    assert chunk is not None and chunk.words[active].text == "hello"

    chunk, active = tl.active_caption(span, 450)     # inside "world"
    assert chunk is not None and chunk.words[active].text == "world"


def test_word_at_global_lookup(tmp_path):
    vtt = tmp_path / "t.vtt"
    vtt.write_text(_VTT, encoding="utf-8")
    tl = Timeline([_turn(1, 1500), _turn(2, 1500, vtt=str(vtt))], fps=30)
    w = tl.word_at(1600)
    assert w is not None and w.text == "hello"
    assert tl.word_at(100) is None                   # silent first turn
