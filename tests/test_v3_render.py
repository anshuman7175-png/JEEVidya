"""
Rig v3 render-path tests (Terminal Plan, Part VI wiring).

The synthetic v3 rig — inpainted head plate, 478 canonical landmarks,
parametric feature geometry, registered poses with headless bakes — is
built by `tests/conftest.py` so every suite that needs a real Rig on
disk shares ONE definition of what a rig is. This file only drives
BoneEngine through the unified HeadAssembly path and asserts the
contracts the wiring promises:

  • a v3 rig routes render() through HeadAssembly; a v1 rig never does
  • art-fitted mouth targets override the anatomical defaults
  • the amplitude envelope GATES the jaw (min), it never forces it open
  • pose cross-fades and physics channels render without error
  • a rig that claims v3 with a missing plate fails LOUDLY at build time
"""
from __future__ import annotations

import os

import pytest

from engine.bone_engine import BoneEngine, PuppetPose
from engine.mouth_model import DEFAULT_TARGETS
from engine.rig import Rig
from engine.visemes import V


# ═══════════════════════════════════════════
# routing
# ═══════════════════════════════════════════

def test_v1_rig_never_builds_assembly(v1_rig):
    engine = BoneEngine(v1_rig)
    assert engine.assembly is None
    frame = engine.render(PuppetPose())
    assert frame.mode == "RGBA"
    assert frame.size == (engine.width, engine.height)


def test_v3_rig_builds_assembly(v3_rig):
    engine = BoneEngine(v3_rig)
    assert engine.assembly is not None
    # Art-fitted targets override the anatomical defaults …
    assert engine._mouth_targets[V.OPEN_A].jaw == pytest.approx(0.9)
    assert engine._mouth_targets[V.BILABIAL].press == pytest.approx(0.8)
    # … while untouched visemes keep the defaults, and unknown names
    # from an old bake are skipped, not fatal.
    assert engine._mouth_targets[V.DENTAL] == DEFAULT_TARGETS[V.DENTAL]


def test_missing_plate_fails_loudly(v3_rig, characters_dir, char_name):
    os.remove(os.path.join(characters_dir, char_name, "rig", "head_plate.png"))
    rig = Rig.load(char_name)
    with pytest.raises((FileNotFoundError, RuntimeError)):
        BoneEngine(rig)


# ═══════════════════════════════════════════
# rendering
# ═══════════════════════════════════════════

def test_v3_neutral_frame_renders(v3_rig):
    engine = BoneEngine(v3_rig)
    frame = engine.render(PuppetPose())
    assert frame.mode == "RGBA"
    assert frame.size == (engine.width, engine.height)
    assert frame.getchannel("A").getbbox() is not None, "frame is empty"


def test_v3_speech_frame_differs_from_neutral(v3_rig):
    engine = BoneEngine(v3_rig)
    neutral = engine.render(PuppetPose())
    speech = engine.render(
        PuppetPose(viseme="OPEN_A", viseme_to="MID_E", viseme_blend=0.3,
                   mouth_open=1.0, blink=0.4, brow=0.5,
                   head_tilt=6.0, head_yaw=0.3, head_nod=-0.2,
                   eye_dx=3.0, eye_dy=-2.0),
        physics=(2.0, 3.5))
    assert speech.size == neutral.size
    assert speech.tobytes() != neutral.tobytes()


def test_v3_pose_crossfade_renders(v3_rig):
    engine = BoneEngine(v3_rig)
    for t in (0.0, 0.5, 1.0):
        frame = engine.render(PuppetPose(body_pose="neutral",
                                         body_pose_to="lean",
                                         body_pose_blend=t))
        assert frame.size == (engine.width, engine.height)
        assert frame.getchannel("A").getbbox() is not None


def test_v3_squash_keeps_canvas_size(v3_rig):
    engine = BoneEngine(v3_rig)
    frame = engine.render(PuppetPose(squash=0.15))
    assert frame.size == (engine.width, engine.height)


# ═══════════════════════════════════════════
# channel mapping contracts
# ═══════════════════════════════════════════

def test_jaw_is_gated_by_amplitude_envelope(v3_rig):
    """rendered jaw = min(articulatory target, envelope): silence can
    close a mouth the aligner left open, it can never force it wider."""
    engine = BoneEngine(v3_rig)
    silent = engine._mouth_params(PuppetPose(viseme="OPEN_A", mouth_open=0.0))
    assert silent.jaw == pytest.approx(0.0)
    loud = engine._mouth_params(PuppetPose(viseme="OPEN_A", mouth_open=1.0))
    assert loud.jaw == pytest.approx(0.9)
    # A BILABIAL stays sealed no matter how loud the envelope is.
    sealed = engine._mouth_params(PuppetPose(viseme="BILABIAL", mouth_open=1.0))
    assert sealed.jaw == pytest.approx(0.0)


def test_unknown_viseme_falls_back_to_rest(v3_rig):
    engine = BoneEngine(v3_rig)
    p = engine._mouth_params(PuppetPose(viseme="GIBBERISH", mouth_open=1.0))
    rest = engine._mouth_targets[V.REST]
    assert p.jaw == pytest.approx(min(rest.jaw, 1.0))


def test_eye_state_normalizes_saccades_by_iris(v3_rig):
    """Saccades arrive in head-local px; EyeState speaks iris radii."""
    engine = BoneEngine(v3_rig)
    r = engine.assembly.eyes.left.geo.iris_r
    st = engine._eye_state(PuppetPose(eye_dx=r, eye_dy=-r / 2))
    assert st.eye_dx == pytest.approx(1.0)
    assert st.eye_dy == pytest.approx(-0.5)
    # Clamped to the unit ball even for a wild saccade.
    wild = engine._eye_state(PuppetPose(eye_dx=r * 10))
    assert wild.eye_dx == pytest.approx(1.0)


def test_v3_plate_cache_hits_on_repeat_pose(v3_rig):
    engine = BoneEngine(v3_rig)
    pose = PuppetPose(viseme="OPEN_A", mouth_open=0.7)
    engine.render(pose)
    misses = engine.assembly.plate_misses
    engine.render(pose)
    assert engine.assembly.plate_misses == misses
    assert engine.assembly.plate_hits >= 1
