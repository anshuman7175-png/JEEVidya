"""
Adversarial temporal gauntlet tests (Terminal Plan, Part XXI).

Every artifact that survives per-frame delivery QC and still reads as
fake is temporal, so each detector is driven against a synthetic defect
whose magnitude is KNOWN — the only way to show a threshold measures
what it claims to measure rather than merely producing a number.

Decoding is deliberately out of scope here: `decode_burst` needs ffmpeg
and a real container, while the detectors are pure array maths. They are
tested on frames built in memory, and `run()` is tested for the failure
that must never raise (a missing deliverable).
"""
from __future__ import annotations

import numpy as np
import pytest

from tools import gauntlet as G

H, W = 192, 108          # a small 9:16, wide enough for sub-pixel phase fits


def _texture(seed: int = 0) -> np.ndarray:
    """A structured, non-repeating frame: phase correlation needs real
    content, and flat noise is not what a puppet on a background is."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:H, 0:W]
    base = (110 + 40 * np.sin(xs / 3.0) + 30 * np.cos(ys / 5.0)
            + rng.normal(0, 6, (H, W)))
    frame = np.stack([base, base * 0.96 + 6, base * 0.9 + 12], axis=-1)
    return np.clip(frame, 0, 255).astype(np.uint8)


def _shifted(frame: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return np.roll(np.roll(frame, dy, axis=0), dx, axis=1)


def _scaled(frame: np.ndarray, k: float) -> np.ndarray:
    """Uniform gain. Rounded, not truncated: a half-code quantisation
    bias would be a bigger signal than the 1% step under test."""
    return np.clip(np.rint(frame.astype(np.float32) * k),
                   0, 255).astype(np.uint8)


# ═══════════════════════════════════════════
# flicker
# ═══════════════════════════════════════════

def test_a_clean_pan_has_no_flicker():
    frames = [_shifted(_texture(), 0, i) for i in range(12)]
    assert float(G.luma_steps(frames).max()) < G.FLICKER_STEP_LIMIT


def test_one_percent_luma_step_is_caught():
    """Weber contrast: ~1% is the floor of what a phone screen shows, so
    it is exactly where the gate has to sit."""
    base = _texture()
    frames = [base, _scaled(base, 1.05), base]
    assert float(G.luma_steps(frames).max()) > G.FLICKER_STEP_LIMIT


def test_luma_steps_are_relative_not_absolute():
    """A dark frame and a bright frame stepping by the same PERCENTAGE
    must score the same — that is what makes the threshold portable."""
    dark, bright = _scaled(_texture(), 0.5), _texture()
    a = float(G.luma_steps([dark, _scaled(dark, 1.04)])[0])
    b = float(G.luma_steps([bright, _scaled(bright, 1.04)])[0])
    assert a == pytest.approx(0.04, rel=0.1)
    assert a == pytest.approx(b, rel=0.1)


# ═══════════════════════════════════════════
# freeze
# ═══════════════════════════════════════════

def test_moving_frames_never_read_as_frozen():
    frames = [_shifted(_texture(), 0, i) for i in range(10)]
    assert G.freeze_runs(frames) == 1


def test_a_stalled_compositor_is_caught():
    base = _texture()
    frames = [_shifted(base, 0, 1), base, base, base, base,
              _shifted(base, 0, 2)]
    assert G.freeze_runs(frames) == 4
    assert G.freeze_runs(frames) > G.FREEZE_MAX_RUN


def test_a_single_repeat_is_tolerated():
    """One duplicate is a legal 2-frame hold; three in a row is a stall."""
    base = _texture()
    assert G.freeze_runs([_shifted(base, 0, 1), base, base,
                          _shifted(base, 0, 2)]) <= G.FREEZE_MAX_RUN


# ═══════════════════════════════════════════
# motion: teleport + jitter
# ═══════════════════════════════════════════

def test_phase_correlation_recovers_a_known_shift():
    base = _texture()
    dx, dy = G.phase_shift(G._luma(base), G._luma(_shifted(base, 3, -5)))
    assert abs(abs(dx) - 5.0) < 1.0
    assert abs(abs(dy) - 3.0) < 1.0


def test_phase_correlation_ignores_a_grade_change():
    """Lighting must not register as motion: the magnitude is normalised
    away, which is what separates 'the subject moved' from 'the grade
    shifted'."""
    base = _texture()
    dx, dy = G.phase_shift(G._luma(base), G._luma(_scaled(base, 1.25)))
    assert np.hypot(dx, dy) < 1.0


def test_a_steady_pan_passes_teleport_and_jitter():
    frames = [_shifted(_texture(), 0, i) for i in range(16)]
    motion = G.motion_track(frames)
    assert float(motion.max()) <= G.TELEPORT_LIMIT_FRAC
    assert float(np.abs(np.diff(motion, n=2)).max()) <= G.JITTER_LIMIT_FRAC


def test_a_jump_cut_inside_a_shot_is_caught():
    base = _texture()
    frames = [base, _shifted(base, 0, 1), _shifted(base, 0, 14),
              _shifted(base, 0, 15)]
    assert float(G.motion_track(frames).max()) > G.TELEPORT_LIMIT_FRAC


def test_buzzing_motion_is_caught_even_when_small():
    """Every individual step is far below the teleport limit; the 3rd
    derivative is what the eye reads as buzzing."""
    base = _texture()
    # Positions 0,1,4,5,8,9,… → per-frame steps alternate 1 px, 3 px.
    xs = [(i // 2) * 4 + (i % 2) for i in range(16)]
    motion = G.motion_track([_shifted(base, 0, x) for x in xs])
    assert float(motion.max()) <= G.TELEPORT_LIMIT_FRAC
    assert float(np.abs(np.diff(motion, n=2)).max()) > G.JITTER_LIMIT_FRAC


# ═══════════════════════════════════════════
# letterbox
# ═══════════════════════════════════════════

def test_a_full_bleed_frame_has_no_border():
    assert G.uniform_border_frac(_texture()) <= G.LETTERBOX_MAX_FRAC


def test_wrong_scale_math_shows_up_as_a_border():
    frame = _texture()
    frame[:6, :, :] = 0
    frame[-6:, :, :] = 0
    assert G.uniform_border_frac(frame) == pytest.approx(12.0 / H)
    assert G.uniform_border_frac(frame) > G.LETTERBOX_MAX_FRAC


def test_pillarbox_is_caught_on_the_other_axis():
    frame = _texture()
    frame[:, :4, :] = 0
    assert G.uniform_border_frac(frame) > G.LETTERBOX_MAX_FRAC


# ═══════════════════════════════════════════
# chroma drift
# ═══════════════════════════════════════════

def test_a_stable_grade_does_not_drift():
    bursts = [[_texture(i) for _ in range(4)] for i in range(3)]
    assert G.chroma_drift(bursts) <= G.CHROMA_DRIFT_LIMIT


def test_slow_hue_migration_is_caught():
    """A grade that is not deterministic walks the opponent channels
    across the runtime — invisible frame to frame, obvious end to end."""
    bursts = []
    for b in range(4):
        burst = []
        for _ in range(4):
            f = _texture(0).astype(np.int16)
            f[..., 0] = np.clip(f[..., 0] + 5 * b, 0, 255)
            burst.append(f.astype(np.uint8))
        bursts.append(burst)
    assert G.chroma_drift(bursts) > G.CHROMA_DRIFT_LIMIT


def test_chroma_drift_needs_two_samples():
    assert G.chroma_drift([[_texture()]]) == 0.0


# ═══════════════════════════════════════════
# the report the ship DAG consumes
# ═══════════════════════════════════════════

def test_a_missing_deliverable_fails_instead_of_raising():
    """buildgraph merges these gates into the delivery manifest, so the
    gauntlet must always return a report — never explode the ship DAG."""
    report = G.run("/nonexistent/never/out.mp4")
    assert not report.passed
    assert [g.name for g in report.gates] == ["gauntlet_input"]
    assert report.to_dict()["passed"] is False


def test_an_undecodable_file_fails_instead_of_raising(tmp_path):
    path = tmp_path / "not-a-video.mp4"
    path.write_bytes(b"\x00" * 512)
    report = G.run(str(path))
    assert not report.passed
    assert report.gates and not report.gates[0].passed


def test_report_is_writable_json_for_the_publisher(tmp_path):
    out = tmp_path / "gauntlet.json"
    G.run("/nonexistent/never/out.mp4").save(str(out))
    import json
    data = json.loads(out.read_text())
    assert data["passed"] is False and data["gates"]


def test_cli_surface_reports_a_gate_failure_as_an_exit_code(tmp_path):
    """`jvmake gauntlet` and the ship DAG must agree: a failed gate is a
    non-zero exit, not a traceback."""
    import subprocess
    import sys
    proc = subprocess.run(
        [sys.executable, "jvmake.py", "gauntlet",
         str(tmp_path / "missing.mp4")],
        capture_output=True, text=True, timeout=180)
    assert proc.returncode == 1, proc.stderr
    assert "Traceback" not in proc.stderr
    assert "gauntlet_input" in proc.stdout


def test_thresholds_are_module_constants_not_call_site_literals():
    """The CLI and buildgraph both source their defaults from here, so a
    tuning change lands in exactly one place (the constants lint)."""
    assert G.BURSTS > 1 and G.BURST_FRAMES >= 4
    for name in ("FLICKER_STEP_LIMIT", "FLICKER_P99_LIMIT",
                 "TELEPORT_LIMIT_FRAC", "JITTER_LIMIT_FRAC",
                 "LETTERBOX_MAX_FRAC", "CHROMA_DRIFT_LIMIT"):
        assert getattr(G, name) > 0.0
    assert G.FLICKER_P99_LIMIT < G.FLICKER_STEP_LIMIT
