"""
Naturalism suite — coarticulation, amplitude envelope, beat gestures,
and safe character framing (head can never leave the frame).
"""
import random
from dataclasses import dataclass

import pytest

from engine.cinematics import CameraDynamics
from engine.gestures import GestureTrack
from engine.visemes import (AmplitudeEnvelope, JAW, V, VisemeTrack,
                            visemes_for_word)
from pipeline.compositor import CinematicCompositor


@dataclass
class _Word:
    text: str
    start_ms: int
    end_ms: int


class _Frame(CinematicCompositor):
    """Framing math only — skips the heavy asset-loading __init__."""

    def __init__(self, width=1080, height=1920):
        self.width, self.height = width, height


# ─── Coarticulation ─────────────────────────────────────────

def test_weights_never_emit_unknown_viseme():
    track = VisemeTrack.from_words(
        [_Word("namaste", 0, 600), _Word("bhaiyo", 650, 1200)])
    legal = set(V)
    for t in range(-100, 1400, 7):
        weights, jaw = track.weights_at(float(t))
        assert weights, f"empty weight dict at t={t}ms"
        for v, w in weights.items():
            assert v in legal, f"unknown viseme {v!r} at t={t}ms"
            assert 0.0 <= w <= 1.0 + 1e-6
        assert sum(weights.values()) <= 1.0 + 1e-6
        assert jaw >= 0.0


def test_jaw_is_continuous_across_boundaries():
    """Jaw drop must glide at 30 fps. Openings are slow (jaw drops with
    inertia); closings may be fast — bilabial closure IS a fast event —
    but never a full-range single-frame snap."""
    track = VisemeTrack.from_words(
        [_Word("dekho", 0, 500), _Word("magnet", 520, 1100)])
    prev = None
    for t in range(0, 1300, 33):          # 30 fps sampling
        _, jaw = track.weights_at(float(t))
        if prev is not None:
            delta = jaw - prev
            # Old hard-switch code jumped ±0.85 in one frame. A plosive
            # release ("ma") legitimately opens ~0.4/frame at 30 fps —
            # and the sprite cross-fade smooths the shape on top.
            assert delta < 0.50, f"opening pop at t={t}ms ({delta:+.2f})"
            assert delta > -0.65, f"closing snap at t={t}ms ({delta:+.2f})"
        prev = jaw


def test_bilabial_dominance_reaches_closure_early():
    """Blending INTO a bilabial closure must run ahead of linear time:
    the lips should own >50% of the mix before the vowel event ends."""
    track = VisemeTrack.from_words([_Word("aam", 0, 400)])  # OPEN_A → BILABIAL
    vowel_end = next(e.end_ms for e in track.events
                     if e.viseme == V.OPEN_A)
    entered_blend = False
    early_closure = False
    for t in range(0, 400, 5):
        weights, _ = track.weights_at(float(t))
        w_bil = weights.get(V.BILABIAL, 0.0)
        if 0.05 < w_bil < 0.95 and weights.get(V.OPEN_A, 0.0) > 0.0:
            entered_blend = True
        if t < vowel_end and w_bil >= 0.5:
            early_closure = True
    assert entered_blend, "OPEN_A→BILABIAL boundary never entered its blend window"
    assert early_closure, "bilabial closure did not dominate ahead of linear time"


def test_rest_outside_track():
    track = VisemeTrack.from_words([_Word("hi", 1000, 1400)])
    weights, jaw = track.weights_at(300.0)
    assert weights.get(V.REST, 0.0) == pytest.approx(1.0)
    assert jaw == pytest.approx(0.0)
    weights, jaw = track.weights_at(5000.0)
    assert max(weights, key=weights.get) == V.REST
    assert jaw == pytest.approx(0.0)


def test_visemes_for_word_never_empty():
    for word in ("", "क्या", "9.8", "sin²θ", "velocity", "म्म्म"):
        assert visemes_for_word(word), word


# ─── Amplitude envelope ─────────────────────────────────────

def test_envelope_attack_faster_than_release():
    env = AmplitudeEnvelope(fps=30)
    for _ in range(3):
        env.step(-15.0)                    # loud
    peak = env.level
    assert peak > 0.5, "attack too slow"
    env.step(-80.0)                        # sudden silence
    assert env.level > peak * 0.5, "release too fast (jaw snapping shut)"
    for _ in range(30):
        env.step(-80.0)
    assert env.level < 0.05, "envelope never settles"


def test_envelope_bounded():
    env = AmplitudeEnvelope(fps=30)
    for db in (-80, 0, 40, -200, -15):
        assert 0.0 <= env.step(db) <= 1.0


# ─── Beat gestures ──────────────────────────────────────────

def _words(n, gap_ms=400, dur_ms=300):
    return [_Word(f"shabd{i}", i * gap_ms, i * gap_ms + dur_ms)
            for i in range(n)]


def test_beats_fill_keywordless_speech():
    track = GestureTrack()
    n = track.schedule_beats(_words(30), random.Random(7))
    assert n >= 3, "hands went dead on a 12-second keyword-less turn"


def test_beats_are_deterministic():
    a, b = GestureTrack(), GestureTrack()
    a.schedule_beats(_words(30), random.Random(7))
    b.schedule_beats(_words(30), random.Random(7))
    assert [(i.gesture.name, i.start_ms) for i in a._items] \
        == [(i.gesture.name, i.start_ms) for i in b._items]


def test_numbers_trigger_point():
    track = GestureTrack()
    track.schedule_beats([_Word("9.8", 0, 400)], random.Random(1))
    assert any(i.gesture.name == "point" for i in track._items)


# ─── Safe framing ───────────────────────────────────────────

def test_head_never_cut_off_at_top():
    f = _Frame()
    th = f._char_target_h(1.6)             # close_up preset
    tw = int(th * 0.6)
    ax, ay = f._safe_anchor(540, 1650, tw, th)
    top = ay - th
    assert top >= int(f.height * f.HEADROOM_FRAC), \
        f"head cropped: sprite top at {top}px"


def test_horizontal_clamp_keeps_sprite_on_screen():
    f = _Frame()
    th, tw = 1000, 600
    ax, _ = f._safe_anchor(30, 1800, tw, th)      # far off left
    slack = int(f.width * f.EDGE_SLACK_FRAC)
    assert ax - tw / 2 >= -slack
    ax, _ = f._safe_anchor(1070, 1800, tw, th)    # far off right
    assert ax + tw / 2 <= f.width + slack


def test_size_ceiling_holds():
    f = _Frame()
    assert f._char_target_h(99.0) <= int(f.height * f.CHAR_H_CEILING)


# ─── Camera push-in ─────────────────────────────────────────

def test_push_in_is_bounded_forever():
    cam = CameraDynamics(1080, 1920, seed=3, fps=30)
    zooms = [cam.frame_transform()["zoom"] for _ in range(30 * 120)]  # 2 min
    assert max(zooms) <= 1.0 + CameraDynamics.PUSH_IN_MAX + 0.02, \
        "push-in zoom escaped its cap — characters drift off frame"
