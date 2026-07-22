"""
JEEVidya V5 — Puppet Rig Data Model (Tier 1)
════════════════════════════════════════════
The contract between the Rig Builder (tools/rig_builder.py), the Bone
Engine (engine/bone_engine.py) and the /studio editor.

A Rig describes, in **puppet space** (the pixel space of body.png):
  • joints    — hips, neck, head_center, head_top (the 2-bone skeleton)
  • face      — mouth / eye / brow boxes + sampled skin & lip colors
  • layers    — sliced PNGs (head, torso) with paste offsets
  • visemes   — baked mouth-shape sprite files
  • params    — feathering, hair line, rig version

Everything is stored in assets/characters/<name>/rig/rig.json so a
project re-renders bit-identically months later (Tier 5 .jvproj).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import settings

RIG_VERSION = 1
VISEME_NAMES = ["REST", "MBP", "E", "AI", "O", "FV"]

Box = Tuple[int, int, int, int]          # x0, y0, x1, y1
Point = Tuple[float, float]


def rig_dir(character: str) -> str:
    return os.path.join(settings.CHARACTERS_DIR, character, "rig")


def rig_path(character: str) -> str:
    return os.path.join(rig_dir(character), "rig.json")


def has_rig(character: str) -> bool:
    """True when a complete, loadable rig exists for the character."""
    try:
        rig = Rig.load(character)
    except Exception:
        return False
    return rig.is_complete()


@dataclass
class Layer:
    """A sliced puppet layer: PNG file + paste offset in puppet space."""
    name: str
    file: str                 # relative to rig dir
    offset: Point = (0.0, 0.0)

    def to_dict(self) -> dict:
        return {"name": self.name, "file": self.file, "offset": list(self.offset)}

    @staticmethod
    def from_dict(d: dict) -> "Layer":
        return Layer(name=d["name"], file=d["file"],
                     offset=tuple(d.get("offset", (0, 0))))


@dataclass
class Rig:
    character: str
    size: Tuple[int, int] = (0, 0)                     # body.png (w, h)
    generated_by: str = "unknown"                      # mediapipe | heuristic | manual
    version: int = RIG_VERSION

    # Skeleton (puppet-space px). 2 bones: spine (hips→neck), neck (neck→head_center)
    joints: Dict[str, Point] = field(default_factory=dict)

    # Face geometry + colors
    face: Dict[str, object] = field(default_factory=dict)
    # keys: mouth, eye_l, eye_r, brow_l, brow_r (Box), skin, lip (RGB)

    layers: Dict[str, Layer] = field(default_factory=dict)   # head, torso
    visemes: Dict[str, str] = field(default_factory=dict)    # name → file
    params: Dict[str, float] = field(default_factory=dict)   # feather_px, hair_line_y

    # ─── Convenience accessors ────────────────────────────

    def joint(self, name: str) -> Point:
        return tuple(self.joints[name])

    def box(self, name: str) -> Box:
        return tuple(int(v) for v in self.face[name])

    def color(self, name: str) -> Tuple[int, int, int]:
        return tuple(int(v) for v in self.face[name])

    def is_complete(self) -> bool:
        need_joints = {"hips", "neck", "head_center"}
        need_face = {"mouth", "eye_l", "eye_r", "brow_l", "brow_r", "skin", "lip"}
        if not need_joints <= set(self.joints):
            return False
        if not need_face <= set(self.face):
            return False
        if not {"head", "torso"} <= set(self.layers):
            return False
        d = rig_dir(self.character)
        for layer in self.layers.values():
            if not os.path.exists(os.path.join(d, layer.file)):
                return False
        for f in self.visemes.values():
            if not os.path.exists(os.path.join(d, f)):
                return False
        return True

    # ─── Persistence ──────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "character": self.character,
            "size": list(self.size),
            "generated_by": self.generated_by,
            "joints": {k: list(v) for k, v in self.joints.items()},
            "face": {k: (list(v) if isinstance(v, (list, tuple)) else v)
                     for k, v in self.face.items()},
            "layers": {k: v.to_dict() for k, v in self.layers.items()},
            "visemes": dict(self.visemes),
            "params": dict(self.params),
        }

    def save(self) -> str:
        d = rig_dir(self.character)
        os.makedirs(d, exist_ok=True)
        path = rig_path(self.character)
        tmp = path + ".part"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        os.replace(tmp, path)
        return path

    @staticmethod
    def load(character: str) -> "Rig":
        with open(rig_path(character), "r", encoding="utf-8") as f:
            d = json.load(f)
        rig = Rig(character=d["character"],
                  size=tuple(d.get("size", (0, 0))),
                  generated_by=d.get("generated_by", "unknown"),
                  version=d.get("version", 1))
        rig.joints = {k: tuple(v) for k, v in d.get("joints", {}).items()}
        rig.face = {k: (tuple(v) if isinstance(v, list) else v)
                    for k, v in d.get("face", {}).items()}
        rig.layers = {k: Layer.from_dict(v) for k, v in d.get("layers", {}).items()}
        rig.visemes = dict(d.get("visemes", {}))
        rig.params = dict(d.get("params", {}))
        return rig

    def layer_path(self, name: str) -> str:
        return os.path.join(rig_dir(self.character), self.layers[name].file)

    def viseme_path(self, name: str) -> Optional[str]:
        f = self.visemes.get(name)
        return os.path.join(rig_dir(self.character), f) if f else None
