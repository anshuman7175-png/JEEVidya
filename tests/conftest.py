"""
Shared synthetic-character fixtures.

Every test that needs a real Rig on disk builds it here instead of
reaching for the shipped assets: the render path is then exercised
identically on a laptop, in CI and on a machine that has never run
`jvmake rig`. Two variants are offered — a v1 sprite rig and a full v3
rig (inpainted head plate, 478 landmarks, parametric feature geometry,
registered poses with headless bakes).
"""
from __future__ import annotations

import math
import os

import pytest
from PIL import Image, ImageDraw

from config import settings
from engine.registration import SimilarityTransform
from engine.rig import HeadGeometry, Layer, PoseEntry, Rig

CHAR = "testchar_v3"
BODY_SIZE = (220, 440)
PLATE_SIZE = (100, 130)
PLATE_OFFSET = (60.0, 30.0)


# ═══════════════════════════════════════════
# synthetic art
# ═══════════════════════════════════════════

def _body_png() -> Image.Image:
    img = Image.new("RGBA", BODY_SIZE, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([70, 160, 150, 400], fill=(80, 60, 140, 255))    # torso
    d.ellipse([70, 40, 150, 160], fill=(230, 190, 160, 255))     # head
    return img


def _headless_png(tint: int = 0) -> Image.Image:
    """Body with the head cut out (the D2 contract)."""
    img = Image.new("RGBA", BODY_SIZE, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([70, 160, 150, 400], fill=(80 + tint, 60, 140, 255))
    return img


def _plate_png() -> Image.Image:
    """Inpainted head plate: clean skin, no painted mouth or eyes."""
    img = Image.new("RGBA", PLATE_SIZE, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([5, 5, PLATE_SIZE[0] - 5, PLATE_SIZE[1] - 5],
              fill=(230, 190, 160, 255))
    return img


def _ellipse_ring(cx: float, cy: float, rx: float, ry: float,
                  n: int = 12) -> list:
    return [(cx + rx * math.cos(2 * math.pi * i / n),
             cy + ry * math.sin(2 * math.pi * i / n)) for i in range(n)]


def _lid(cx: float, cy: float, half_w: float, bow: float) -> list:
    """5-point lid polyline, bowed up (bow < 0) or down (bow > 0)."""
    out = []
    for i in range(5):
        t = i / 4.0
        x = cx - half_w + 2 * half_w * t
        out.append((x, cy + bow * math.sin(math.pi * t)))
    return out


# ═══════════════════════════════════════════
# rig builders
# ═══════════════════════════════════════════

def _base_rig() -> Rig:
    """The shared v1 skeleton every variant builds on."""
    rig = Rig(character=CHAR, size=BODY_SIZE, generated_by="manual")
    rig.joints = {"hips": (110.0, 400.0), "neck": (110.0, 160.0),
                  "head_center": (110.0, 100.0)}
    rig.face = {"mouth": (90, 120, 130, 140),
                "eye_l": (85, 80, 105, 95), "eye_r": (115, 80, 135, 95),
                "brow_l": (85, 68, 105, 76), "brow_r": (115, 68, 135, 76),
                "skin": (230, 190, 160), "lip": (170, 80, 80)}
    rig.layers = {"torso": Layer("torso", "torso.png", (70.0, 160.0)),
                  "head": Layer("head", "head.png", (70.0, 40.0))}
    rig.visemes = {}
    rig.params = {}
    return rig


def _head_geometry() -> HeadGeometry:
    lm = [(float(8 + (i % 20) * 4.4), float(8 + (i // 20) * 4.8))
          for i in range(478)]
    return HeadGeometry(
        plate="head_plate.png",
        landmarks=lm,
        lip_outer=_ellipse_ring(50, 95, 16, 8),
        lip_inner=_ellipse_ring(50, 95, 9, 4),
        lid_upper_l=_lid(32, 55, 10, -4), lid_lower_l=_lid(32, 55, 10, 3),
        lid_upper_r=_lid(68, 55, 10, -4), lid_lower_r=_lid(68, 55, 10, 3),
        brow_l=_lid(32, 44, 11, -2), brow_r=_lid(68, 44, 11, -2),
        iris_l=(32.0, 55.0, 6.0), iris_r=(68.0, 55.0, 6.0),
        palette={"skin": (230, 190, 160), "lip": (170, 80, 80),
                 "sclera": (248, 246, 242), "iris": (92, 58, 38),
                 "pupil": (24, 18, 14), "lash": (46, 32, 26),
                 "lip_shadow": (120, 52, 52)},
        offset=PLATE_OFFSET,
        face_height=85.0,
    )


def _write_art(rig_d: str) -> None:
    _body_png().save(os.path.join(rig_d, "..", "body.png"))
    _body_png().crop((70, 160, 150, 400)).save(os.path.join(rig_d, "torso.png"))
    _body_png().crop((70, 40, 150, 160)).save(os.path.join(rig_d, "head.png"))


# ═══════════════════════════════════════════
# fixtures
# ═══════════════════════════════════════════

@pytest.fixture()
def char_name() -> str:
    return CHAR


@pytest.fixture()
def characters_dir(tmp_path, monkeypatch):
    root = tmp_path / "characters"
    root.mkdir()
    monkeypatch.setattr(settings, "CHARACTERS_DIR", str(root))
    return str(root)


@pytest.fixture()
def v1_rig(characters_dir) -> Rig:
    rig = _base_rig()
    rig_d = os.path.join(characters_dir, CHAR, "rig")
    os.makedirs(rig_d)
    _write_art(rig_d)
    rig.save()
    return Rig.load(CHAR)


@pytest.fixture()
def v3_rig(characters_dir) -> Rig:
    rig = _base_rig()
    rig_d = os.path.join(characters_dir, CHAR, "rig")
    os.makedirs(rig_d)
    _write_art(rig_d)
    _plate_png().save(os.path.join(rig_d, "head_plate.png"))
    _headless_png().save(os.path.join(rig_d, "headless_neutral.png"))
    _headless_png(tint=40).save(os.path.join(rig_d, "headless_lean.png"))

    rig.version = 3
    rig.canonical_pose = "neutral"
    rig.head = _head_geometry()
    rig.mouth_targets = {
        "OPEN_A": {"jaw": 0.9, "width": 0.62, "round": 0.05,
                   "press": 0.0, "pull": 0.1},
        "BILABIAL": {"jaw": 0.0, "width": 0.5, "round": 0.0,
                     "press": 0.8, "pull": 0.0},
        "NOT_A_VISEME": {"jaw": 1.0},   # unknown name in an old bake: skipped
    }
    rig.poses = {
        "neutral": PoseEntry(name="neutral",
                             xform=SimilarityTransform.identity().to_dict(),
                             headless="headless_neutral.png"),
        "lean": PoseEntry(name="lean",
                          xform=SimilarityTransform(
                              1.02, 0.05, 4.0, 2.0, 0.4).to_dict(),
                          headless="headless_lean.png"),
    }
    rig.save()
    loaded = Rig.load(CHAR)
    assert loaded.is_v3(), "fixture must satisfy the v3 self-consistency gate"
    return loaded
