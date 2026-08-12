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

import ast
import importlib.util
import math
import os

import pytest

# PIL / engine.rig are imported lazily inside the builders below. A
# module-level import here would make the whole tests/ directory
# uncollectable without the render stack installed, which would take the
# pure-logic suite (phonology, timing, cache keys — the tests that are
# supposed to run in milliseconds anywhere) down with it.

CHAR = "testchar_v3"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIRST_PARTY = frozenset({"engine", "pipeline", "config", "factory",
                          "agents", "tools", "tests"})
_STDLIB_OK = frozenset({"__future__"})
_spec_cache: dict = {}
_heavy_cache: dict = {}


def _installed(root: str) -> bool:
    if root not in _spec_cache:
        try:
            _spec_cache[root] = importlib.util.find_spec(root) is not None
        except (ImportError, ValueError):
            _spec_cache[root] = False
    return _spec_cache[root]


def _imported_roots(path: str):
    """Third-party roots and first-party modules a file imports."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
    except (OSError, SyntaxError):
        return set(), set()
    third, first = set(), set()
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        for name in names:
            root = name.split(".")[0]
            if root in _FIRST_PARTY:
                first.add(name)
            elif root not in _STDLIB_OK:
                third.add(root)
    return third, first


def _missing_deps(module: str, seen=None) -> set:
    """Uninstalled third-party packages a module needs, transitively
    through first-party imports.

    Derived from the source itself rather than a hand-maintained list, so
    a test that grows a numpy dependency is classified correctly without
    anyone remembering to update this file.
    """
    if module in _heavy_cache:
        return _heavy_cache[module]
    _heavy_cache[module] = set()          # cycle guard
    seen = seen if seen is not None else set()
    if module in seen:
        return set()
    seen.add(module)

    path = os.path.join(_REPO_ROOT, module.replace(".", os.sep) + ".py")
    if not os.path.exists(path):
        pkg_init = os.path.join(_REPO_ROOT, module.replace(".", os.sep),
                                "__init__.py")
        if not os.path.exists(pkg_init):
            return set()
        path = pkg_init

    third, first = _imported_roots(path)
    missing = {r for r in third if not _installed(r)}
    for dep in first:
        missing |= _missing_deps(dep, seen)
    _heavy_cache[module] = missing
    return missing


def pytest_ignore_collect(collection_path, config):
    """Skip test modules whose render-stack dependencies are absent.

    Without this the pure-logic gates cannot run at all on a machine that
    lacks numpy/PIL/ffmpeg — pytest fails at *collection*, before a
    single assertion executes, so a green-or-red signal is unavailable
    exactly when it is cheapest to get.
    """
    path = str(collection_path)
    if not (os.path.basename(path).startswith("test_") and path.endswith(".py")):
        return None
    third, first = _imported_roots(path)
    missing = {r for r in third if r != "pytest" and not _installed(r)}
    for dep in first:
        missing |= _missing_deps(dep)
    if missing:
        _SKIPPED_MODULES[os.path.basename(path)] = sorted(missing)
        return True
    return None


_SKIPPED_MODULES: dict = {}


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Never let a dependency-skipped module masquerade as a pass."""
    if not _SKIPPED_MODULES:
        return
    terminalreporter.write_sep(
        "=", "not collected — missing optional render stack")
    for name, deps in sorted(_SKIPPED_MODULES.items()):
        terminalreporter.write_line(f"  {name}: needs {', '.join(deps)}")
BODY_SIZE = (220, 440)
PLATE_SIZE = (100, 130)
PLATE_OFFSET = (60.0, 30.0)


# ═══════════════════════════════════════════
# synthetic art
# ═══════════════════════════════════════════

def _body_png():
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", BODY_SIZE, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([70, 160, 150, 400], fill=(80, 60, 140, 255))    # torso
    d.ellipse([70, 40, 150, 160], fill=(230, 190, 160, 255))     # head
    return img


def _headless_png(tint: int = 0):
    """Body with the head cut out (the D2 contract)."""
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", BODY_SIZE, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([70, 160, 150, 400], fill=(80 + tint, 60, 140, 255))
    return img


def _plate_png():
    """Inpainted head plate: clean skin, no painted mouth or eyes."""
    from PIL import Image, ImageDraw
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

def _base_rig():
    """The shared v1 skeleton every variant builds on."""
    from engine.rig import Layer, Rig
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


def _head_geometry():
    from engine.rig import HeadGeometry
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
    from config import settings
    root = tmp_path / "characters"
    root.mkdir()
    monkeypatch.setattr(settings, "CHARACTERS_DIR", str(root))
    return str(root)


@pytest.fixture()
def v1_rig(characters_dir):
    from engine.rig import Rig
    rig = _base_rig()
    rig_d = os.path.join(characters_dir, CHAR, "rig")
    os.makedirs(rig_d)
    _write_art(rig_d)
    rig.save()
    return Rig.load(CHAR)


@pytest.fixture()
def v3_rig(characters_dir):
    from engine.registration import SimilarityTransform
    from engine.rig import PoseEntry, Rig
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
