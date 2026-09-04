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


def test_horizontal_clamp_is_reach_aware_not_canvas_aware():
    """Sprite canvases carry wide transparent margins. The clamp must bound
    the VISIBLE reach envelope (REACH_FRAC · h), never the raw canvas width,
    or every flanking preset gets dragged toward center."""
    f = _Frame()
    th = 1000
    tw = int(th * 1.00)                            # girl's real canvas aspect
    slack = int(f.width * f.EDGE_SLACK_FRAC)
    reach = f.REACH_FRAC * th
    ax, _ = f._safe_anchor(30, 1800, tw, th)
    assert ax - reach >= -slack - 1e-6
    assert ax < tw / 2 - slack, "clamped on canvas width, not reach"
    ax, _ = f._safe_anchor(1070, 1800, tw, th)
    assert ax + reach <= f.width + slack + 1e-6


def test_flanking_presets_survive_the_clamp_at_rest():
    """Every committed two_shot/medium/closeup preset must sit inside the
    clamp with zero displacement — otherwise the compositor silently
    re-frames the shot the designer approved."""
    from config import brand
    f = _Frame()
    # Canvas aspects (measured from assets/characters/*/body.png):
    # gudiya 1254×1254 → 1.00 (opaque body only 0.485·h wide),
    # chintu  941×1672 → 0.56 (opaque body 0.295·h wide)
    aspect = {"girl": 1.00, "boy": 0.56}
    for shot in ("two_shot", "medium", "extreme_closeup",
                 "reaction_cut", "reveal", "fullscreen_explain"):
        for key, p in brand.SHOT_PRESETS[shot].items():
            who = key.split("_")[0]
            if who not in aspect or p["scale"] <= 0.01:
                continue
            th = f._char_target_h(p["scale"])
            tw = int(th * aspect[who])
            ax, _ = f._safe_anchor(p["x"], p["y"], tw, th)
            assert ax == pytest.approx(p["x"]), \
                f"{shot}.{key}: preset x={p['x']} displaced to {ax:.1f}"


def test_safe_anchor_preserves_subpixel_position():
    """The puppet path pastes on a continuous raster; integer truncation
    inside the clamp would reintroduce the motion judder it exists to
    remove."""
    f = _Frame()
    ax, ay = f._safe_anchor(540.37, 1650.61, 600, 900)
    assert ax == pytest.approx(540.37)
    assert ay == pytest.approx(1650.61)


def test_oversized_sprite_centres_instead_of_oscillating():
    f = _Frame()
    ax, _ = f._safe_anchor(0, 1800, 1400, 4000)    # both bounds cross
    assert ax == pytest.approx(f.width / 2)


# ── YouTube Shorts player UI contract ────────────────────────────────────
# Conservative end of the published 2025–26 safe-zone ranges (the overlay
# shifts by device/app version): top 250, rail 180 px wide from y 900,
# bottom 420. Must match tools/shorts_overlay.py ZONES. The player paints
# these over the video; faces and captions must avoid them or the viewer
# never sees what we rendered.
SHORTS_TOP_BAR_Y1 = 250
SHORTS_RAIL_X0, SHORTS_RAIL_Y0, SHORTS_RAIL_Y1 = 900, 900, 1560
SHORTS_META_Y0 = 1500
# Opaque body half-width as a fraction of sprite height (assets/…/body.png)
BODY_HALF = {"girl": 0.2425, "boy": 0.1475}
FACE_FRAC = 0.24    # head occupies the top ~24 % of the sprite


def test_shorts_zones_match_the_overlay_tool():
    """One set of numbers: QA overlay, tests and configs cannot drift."""
    from tools import shorts_overlay as so
    assert so.ZONES["TOP_BAR"][3] == SHORTS_TOP_BAR_Y1
    assert so.ZONES["ACTION_RAIL"][0] == SHORTS_RAIL_X0
    assert so.ZONES["ACTION_RAIL"][1] == SHORTS_RAIL_Y0
    assert so.ZONES["ACTION_RAIL"][3] == SHORTS_RAIL_Y1
    assert so.ZONES["METADATA"][1] == SHORTS_META_Y0


def test_caption_band_sits_in_the_shorts_safe_window():
    from config import settings
    W, H = 1080, 1920
    band_h = settings.CAPTION_FONT_SIZE * settings.CAPTION_LINE_HEIGHT \
        * settings.CAPTION_MAX_LINES
    top = settings.CAPTION_Y_POSITION * H - band_h / 2
    bot = settings.CAPTION_Y_POSITION * H + band_h / 2
    assert top > SHORTS_TOP_BAR_Y1, "caption under the Shorts header"
    assert bot < SHORTS_RAIL_Y0, "caption drops into the action-rail rows"
    # The band hugs the header so the full safe window below it goes to
    # the characters (no dead space above, feet above the title row).
    # Too high and the glyphs slide under the header; too low and the
    # heads it pushes down drag the feet into the metadata block.
    assert 0.17 <= settings.CAPTION_Y_POSITION <= 0.26
    # Mobile legibility: every 2025–26 guide lands on 60–75 px @ 1080 wide
    assert 60 <= settings.CAPTION_FONT_SIZE <= 75
    assert settings.CAPTION_MAX_LINES <= 2
    # Centred caption must stay inside the ~900 px centre column
    assert settings.CAPTION_MAX_WIDTH_FRAC * W <= 900


def test_v5_head_clear_line_matches_the_presets():
    """compositor_v5 pushes heads below caption_clear_y at run time; the
    presets must already satisfy it or every cut hops on frame 1."""
    from config import brand, settings
    from engine.captions import CaptionStyle
    # Same formula as CompositorV5.__init__ (caption_clear_y)
    st = CaptionStyle.for_frame(1080, 1920)
    pad = st.stroke_px + st.shadow_blur * 2 + 4
    band_h = st.font_px * st.line_height * st.max_lines + pad * 2
    clear_y = int(1920 * settings.CAPTION_Y_POSITION + band_h / 2
                  + settings.CAPTION_MIN_HEAD_CLEARANCE)
    f = _Frame()
    for shot in ("two_shot", "medium", "extreme_closeup", "reaction_cut"):
        for key, p in brand.SHOT_PRESETS[shot].items():
            if p["scale"] <= 0.01:
                continue
            th = f._char_target_h(p["scale"])
            assert p["y"] - th >= clear_y, \
                f"{shot}.{key}: head top {p['y'] - th} above clear line {clear_y}"


def test_no_face_or_body_under_the_shorts_ui():
    """Every visible preset: face clear of the metadata row, and the
    right-hand character's opaque body clear of the action rail."""
    from config import brand
    f = _Frame()
    for shot in ("two_shot", "medium", "extreme_closeup",
                 "reaction_cut", "reveal", "fullscreen_explain"):
        for key, p in brand.SHOT_PRESETS[shot].items():
            who = key.split("_")[0]
            if who not in BODY_HALF or p["scale"] <= 0.01:
                continue
            th = f._char_target_h(p["scale"])
            head_top = p["y"] - th
            face_bot = head_top + FACE_FRAC * th
            body_r = p["x"] + BODY_HALF[who] * th
            assert face_bot <= SHORTS_META_Y0, \
                f"{shot}.{key}: face (to y={face_bot:.0f}) under metadata row"
            if face_bot >= SHORTS_RAIL_Y0 and head_top <= SHORTS_RAIL_Y1:
                assert body_r <= SHORTS_RAIL_X0, \
                    f"{shot}.{key}: body edge x={body_r:.0f} under action rail"


def test_legs_never_hide_behind_the_title_block():
    """Every visible preset stands with its FEET above the metadata row:
    the title / @channel / Subscribe block must never cover the legs."""
    from config import brand
    for shot, presets in brand.SHOT_PRESETS.items():
        for key, p in presets.items():
            if p["scale"] <= 0.01:
                continue
            assert p["y"] <= SHORTS_META_Y0, \
                f"{shot}.{key}: feet at y={p['y']} under the title block"


def test_safe_window_is_used_top_to_bottom():
    """No dead band above the caption, no dead band under the heads: the
    caption starts within ~60 px of the header and the tallest speaker's
    head starts within ~40 px of the caption clear line, so the whole
    250→1500 window is picture."""
    from config import brand, settings
    from engine.captions import CaptionStyle
    st = CaptionStyle.for_frame(1080, 1920)
    pad = st.stroke_px + st.shadow_blur * 2 + 4
    band_h = st.font_px * st.line_height * st.max_lines + pad * 2
    band_top = 1920 * settings.CAPTION_Y_POSITION - band_h / 2
    glyph_top = band_top + pad
    assert SHORTS_TOP_BAR_Y1 < glyph_top <= SHORTS_TOP_BAR_Y1 + 60, \
        f"caption glyphs start at y={glyph_top:.0f}; dead space above"
    clear_y = int(1920 * settings.CAPTION_Y_POSITION + band_h / 2
                  + settings.CAPTION_MIN_HEAD_CLEARANCE)
    f = _Frame()
    p = brand.SHOT_PRESETS["two_shot"]["girl_active"]
    head_top = p["y"] - f._char_target_h(p["scale"])
    assert clear_y <= head_top <= clear_y + 40, \
        f"speaker head at y={head_top}; gap to caption {head_top - clear_y}px"


def test_two_shot_keeps_the_characters_apart():
    """Pulling Chintu off the rail must not make the pair collide."""
    from config import brand
    f = _Frame()
    ts = brand.SHOT_PRESETS["two_shot"]
    g, b = ts["girl_active"], ts["boy_inactive"]
    gh, bh = f._char_target_h(g["scale"]), f._char_target_h(b["scale"])
    gap = (b["x"] - BODY_HALF["boy"] * bh) - (g["x"] + BODY_HALF["girl"] * gh)
    assert gap >= 100, f"bodies only {gap:.0f}px apart"


def test_size_ceiling_holds():
    f = _Frame()
    assert f._char_target_h(99.0) <= int(f.height * f.CHAR_H_CEILING)


# ─── Camera push-in ─────────────────────────────────────────

def test_push_in_is_bounded_forever():
    cam = CameraDynamics(1080, 1920, seed=3, fps=30)
    zooms = [cam.frame_transform()["zoom"] for _ in range(30 * 120)]  # 2 min
    assert max(zooms) <= 1.0 + CameraDynamics.PUSH_IN_MAX + 0.02, \
        "push-in zoom escaped its cap — characters drift off frame"
