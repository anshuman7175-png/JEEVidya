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

Rig v3 (Terminal Plan, Part III) adds the geometry that makes the face
defect classes D1–D3 UNREPRESENTABLE rather than merely fixed:

  • head          — the canonical head plate (mouth + eyes inpainted out)
                    plus the 478 canonical landmarks and the parametric
                    feature geometry (lip rings, lid polylines, irises,
                    brows), a sampled palette and a mouth shading map.
  • mouth_targets — the 5-D mouth parameters least-squares fitted FROM
                    the character's own viseme art (never hand-guessed).
  • poses         — EVERY pose landmarked independently and registered
                    to canonical with its own similarity transform, plus
                    headless-body / head-mask / occluder bakes.

D1 died because a pose no longer inherits `body.png`'s face boxes: it
carries its own `xform`. D3 died because the plate has no painted mouth
left to hide. Both are schema properties now, not code conventions —
`require_v3()` refuses to render anything older.

Everything is stored in assets/characters/<name>/rig/rig.json so a
project re-renders bit-identically months later (Tier 5 .jvproj).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import settings

RIG_VERSION = 3

# Read support for older rigs is retained so /studio and archived
# .jvproj bundles still load, but `render()` requires v3 — a v2 rig
# raises instead of silently misplacing the face (Part III).
MIN_RENDERABLE_VERSION = 3


def _fit_lids_to_aperture(upper: List[Tuple[float, float]],
                          lower: List[Tuple[float, float]],
                          aperture: List[Tuple[float, float]]
                          ) -> Tuple[List[Tuple[float, float]],
                                     List[Tuple[float, float]]]:
    """Rescale MediaPipe's lid margins onto the MEASURED eye.

    MediaPipe gives lid polylines whose SHAPE is right (a lid is a
    shallow arc, higher at the centre) but whose SCALE is human — on
    chibi art they span ~30px of a ~68px drawn eye. A lid sweep driven
    by them can only ever cross the middle of the eye, which is why
    blink=1 left the iris visible.

    So map the lid pair affinely from its own bounding box onto the
    aperture's: shape preserved, extent corrected. The upper margin is
    placed on the aperture's top edge and the lower on its bottom, so a
    full sweep traverses the entire drawn opening and closure is
    geometric rather than tuned.
    """
    if not upper or not lower or not aperture:
        return upper, lower
    ap = [(float(x), float(y)) for x, y in aperture]
    axs = [p[0] for p in ap]
    ays = [p[1] for p in ap]
    ax0, ax1, ay0, ay1 = min(axs), max(axs), min(ays), max(ays)

    lid = [(float(x), float(y)) for x, y in list(upper) + list(lower)]
    lxs = [p[0] for p in lid]
    lys = [p[1] for p in lid]
    lx0, lx1, ly0, ly1 = min(lxs), max(lxs), min(lys), max(lys)
    if lx1 - lx0 < 1e-6 or ly1 - ly0 < 1e-6:
        return upper, lower

    sx = (ax1 - ax0) / (lx1 - lx0)
    sy = (ay1 - ay0) / (ly1 - ly0)

    def warp(pts):
        return [(ax0 + (x - lx0) * sx, ay0 + (y - ly0) * sy)
                for x, y in pts]

    return warp(upper), warp(lower)

# Canonical MediaPipe FaceLandmarker landmark count (468 mesh + 10 iris).
N_LANDMARKS = 478
# The 10-class viseme set — MUST match engine/visemes.py V enum values,
# because BoneEngine looks sprites up by pose.viseme (e.g. "OPEN_A").
VISEME_NAMES = ["REST", "BILABIAL", "LABIODENTAL", "DENTAL", "RETROFLEX",
                "OPEN_A", "MID_E", "CLOSED_I", "ROUNDED_TENSE", "ROUNDED_LAX"]

# Legacy 5-class sprite names (rig v1) → nearest 10-class name.
LEGACY_VISEME_ALIAS = {
    "MBP": "BILABIAL", "FV": "LABIODENTAL", "E": "MID_E",
    "AI": "OPEN_A", "O": "ROUNDED_LAX",
}

# When a sprite is missing for a 10-class viseme, fall back to the
# nearest available shape (ordered by articulatory similarity).
VISEME_FALLBACK = {
    "DENTAL": ["MID_E", "LABIODENTAL", "OPEN_A"],
    "RETROFLEX": ["MID_E", "OPEN_A"],
    "CLOSED_I": ["MID_E", "DENTAL"],
    "ROUNDED_TENSE": ["ROUNDED_LAX", "OPEN_A"],
    "ROUNDED_LAX": ["ROUNDED_TENSE", "OPEN_A"],
    "LABIODENTAL": ["DENTAL", "MID_E"],
    "BILABIAL": ["REST"],
    "MID_E": ["OPEN_A", "DENTAL"],
    "OPEN_A": ["MID_E"],
}

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
class HeadGeometry:
    """Rig v3 §3.5 — everything about the canonical head, in HEAD-PLATE
    space (pixels relative to the plate's own top-left corner).

    `plate` is the inpainted face: the painted mouth and both eyes have
    been removed with `cv2.inpaint(INPAINT_NS)` seeded from the dilated
    lip/lid contours, so the parametric mouth (Part IV) and eyes (Part V)
    draw onto clean skin carrying the artwork's own shading. There is no
    feathered backing, no tone match, no ring clutter — D3 cannot recur
    because there is nothing left to hide.
    """
    plate: str = "head_plate.png"           # relative to rig dir
    landmarks: List[Point] = field(default_factory=list)   # 478, plate space
    lip_outer: List[Point] = field(default_factory=list)
    lip_inner: List[Point] = field(default_factory=list)
    lid_upper_l: List[Point] = field(default_factory=list)
    lid_lower_l: List[Point] = field(default_factory=list)
    lid_upper_r: List[Point] = field(default_factory=list)
    lid_lower_r: List[Point] = field(default_factory=list)
    brow_l: List[Point] = field(default_factory=list)
    brow_r: List[Point] = field(default_factory=list)
    iris_l: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # cx, cy, r
    iris_r: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # §3.5b — the eye the ARTIST drew, measured from pixels by
    # tools/art_eyes.py. MediaPipe is trained on photographs, so on
    # stylised art its lid ring is 2.2–2.9× too small and sits on the
    # eyeball's upper third; believing it painted a small procedural eye
    # inside the big drawn one and let a blink's skin fill escape as a
    # hard rectangle. These entries are the measured truth and take
    # precedence; the MediaPipe lids above are kept only as the seed
    # that located the eye and for brow/lid tracking.
    art_eye_l: Dict[str, object] = field(default_factory=dict)
    art_eye_r: Dict[str, object] = field(default_factory=dict)
    palette: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)
    shading: Optional[str] = None           # mouth-region shading map PNG
    offset: Point = (0.0, 0.0)              # plate origin in puppet space
    face_height: float = 0.0                # chin→brow, for scale-free gates

    def to_dict(self) -> dict:
        return {
            "plate": self.plate,
            "landmarks": [list(p) for p in self.landmarks],
            "lip_outer": [list(p) for p in self.lip_outer],
            "lip_inner": [list(p) for p in self.lip_inner],
            "lid_upper_l": [list(p) for p in self.lid_upper_l],
            "lid_lower_l": [list(p) for p in self.lid_lower_l],
            "lid_upper_r": [list(p) for p in self.lid_upper_r],
            "lid_lower_r": [list(p) for p in self.lid_lower_r],
            "brow_l": [list(p) for p in self.brow_l],
            "brow_r": [list(p) for p in self.brow_r],
            "iris_l": list(self.iris_l),
            "iris_r": list(self.iris_r),
            "art_eye_l": dict(self.art_eye_l),
            "art_eye_r": dict(self.art_eye_r),
            "palette": {k: list(v) for k, v in self.palette.items()},
            "shading": self.shading,
            "offset": list(self.offset),
            "face_height": self.face_height,
        }

    @staticmethod
    def from_dict(d: dict) -> "HeadGeometry":
        def poly(key: str) -> List[Point]:
            return [tuple(p) for p in d.get(key, [])]
        return HeadGeometry(
            plate=d.get("plate", "head_plate.png"),
            landmarks=poly("landmarks"),
            lip_outer=poly("lip_outer"), lip_inner=poly("lip_inner"),
            lid_upper_l=poly("lid_upper_l"), lid_lower_l=poly("lid_lower_l"),
            lid_upper_r=poly("lid_upper_r"), lid_lower_r=poly("lid_lower_r"),
            brow_l=poly("brow_l"), brow_r=poly("brow_r"),
            iris_l=tuple(d.get("iris_l", (0.0, 0.0, 0.0))),
            iris_r=tuple(d.get("iris_r", (0.0, 0.0, 0.0))),
            art_eye_l=dict(d.get("art_eye_l") or {}),
            art_eye_r=dict(d.get("art_eye_r") or {}),
            palette={k: tuple(v) for k, v in d.get("palette", {}).items()},
            shading=d.get("shading"),
            offset=tuple(d.get("offset", (0.0, 0.0))),
            face_height=float(d.get("face_height", 0.0)),
        )

    def eye_dict(self, left: bool) -> dict:
        """The EyeGeometry payload for one side (engine/eye_model.py).

        The APERTURE is the measured outline of the eye the artist drew
        (§3.5b). It is the authoritative bound: every pixel the eye
        renderer paints is clipped to it, so a blink's skin fill cannot
        escape onto the brow and a saccade cannot slide the iris onto the
        cheek. Where the measurement exists it also supplies the iris,
        because MediaPipe's ring is ~2.5× too small on this art.

        The lid margins stay MediaPipe-derived: they are polylines used
        to shape the lid sweep and track the brow, and their SHAPE is
        right even where their scale is not. They are rescaled to span
        the measured aperture so the sweep crosses the whole drawn eye.

        Without a measurement (legacy v1/v2 rigs) the socket is DERIVED
        from the two lid margins rather than stored: a stored socket
        could disagree with the lids it is supposed to bound, and a
        sclera that leaks past a lid is the class of bug Law 1 forbids
        representing. Derivation makes the two consistent by construction.
        """
        def _pt(v) -> list:
            """A 2-vector from a possibly-absent rig field."""
            try:
                return [float(v[0]), float(v[1])]
            except Exception:
                return [0.0, 0.0]

        upper = [tuple(p) for p in (self.lid_upper_l if left else self.lid_upper_r)]
        lower = [tuple(p) for p in (self.lid_lower_l if left else self.lid_lower_r)]
        art = self.art_eye_l if left else self.art_eye_r
        aperture = list(art.get("aperture") or ())

        if aperture:
            ic = list(art.get("iris_c") or (0.0, 0.0))
            axes = list(art.get("iris_axes") or (0.0, 0.0))
            r = float(art.get("iris_r") or 0.0)
            upper, lower = _fit_lids_to_aperture(upper, lower, aperture)
            org = _pt(art.get("eyeball_origin"))
            return {
                "socket": [list(p) for p in aperture],
                "aperture": [list(p) for p in aperture],
                "lid_upper": [list(p) for p in upper],
                "lid_lower": [list(p) for p in lower],
                "iris": [ic[0], ic[1], r],
                "iris_axes": axes,
                "iris_angle": float(art.get("iris_angle") or 0.0),
                "colors": {k: list(v) for k, v in
                           (art.get("colors") or {}).items()},
                # The artist's own eyeball, cut out at bake time. The
                # renderer translates THESE pixels for gaze instead of
                # synthesizing an eye, so a resting frame is the artwork.
                "eyeball": str(art.get("eyeball") or ""),
                "eyeball_origin": [float(org[0]), float(org[1])],
                # The socket the eyeball uncovers when gaze moves, and the
                # artist's own eyelid skin that slides down to blink. Both
                # replace flat palette fills with real painted pixels.
                "socket_img": str(art.get("socket_img") or ""),
                "socket_origin": _pt(art.get("socket_origin")),
                "lid_img": str(art.get("lid_img") or ""),
                "lid_origin": _pt(art.get("lid_origin")),
                # How far the eyeball may travel before its rim reaches the
                # drawn opening, measured from the artwork's own sclera
                # margin, as (left, right, up, down). Dropping this here is
                # what made the renderer fall back to a generic 0.55·iris_r
                # excursion and slide the iris onto the lash on art whose
                # iris nearly fills the eye. Four values because the eyeball
                # is drawn against one corner of the opening on this art, so
                # a symmetric budget measures ~0 (see art_eyes
                # GAZE_H_SPAN_MIN).
                "gaze_box": [float(v) for v in
                             (art.get("gaze_box") or (0.0, 0.0, 0.0, 0.0))],
            }

        socket = upper + list(reversed(lower))
        return {
            "socket": [list(p) for p in socket],
            "lid_upper": [list(p) for p in upper],
            "lid_lower": [list(p) for p in lower],
            "iris": list(self.iris_l if left else self.iris_r),
        }


@dataclass
class PoseEntry:
    """Rig v3 §3.1/3.3/3.4 — one registered pose.

    `xform` is THE fix for D1: this pose's own canonical→pose similarity
    transform (scale, roll θ, translation), solved by Umeyama+IRLS on the
    rigid skull subset. `headless` is the body with the head cut out, so
    a cross-fade between two poses can never composite two faces (D2) —
    there is only ever one head plate, drawn once, on top.
    """
    name: str
    landmarks: List[Point] = field(default_factory=list)
    xform: Dict[str, float] = field(default_factory=dict)
    headless: str = ""
    headmask: str = ""
    occluder: Optional[str] = None
    seam_y: float = 0.0
    occluded: bool = False

    def to_dict(self) -> dict:
        return {"landmarks": [list(p) for p in self.landmarks],
                "xform": dict(self.xform),
                "headless": self.headless, "headmask": self.headmask,
                "occluder": self.occluder, "seam_y": self.seam_y,
                "occluded": self.occluded}

    @staticmethod
    def from_dict(name: str, d: dict) -> "PoseEntry":
        return PoseEntry(
            name=name,
            landmarks=[tuple(p) for p in d.get("landmarks", [])],
            xform=dict(d.get("xform", {})),
            headless=d.get("headless", ""),
            headmask=d.get("headmask", ""),
            occluder=d.get("occluder"),
            seam_y=float(d.get("seam_y", 0.0)),
            occluded=bool(d.get("occluded", False)),
        )


class RigVersionError(RuntimeError):
    """Raised when a pre-v3 rig reaches the render path. Never downgrade
    to a heuristic — a wrong face is worse than a loud failure."""


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

    # ���── Rig v3 (Part III) ────────────────────────────────
    canonical_pose: str = "neutral"
    head: Optional[HeadGeometry] = None
    # viseme name → {"jaw","width","round","press","pull"}, fitted from art
    mouth_targets: Dict[str, Dict[str, float]] = field(default_factory=dict)
    poses: Dict[str, PoseEntry] = field(default_factory=dict)

    # ─── Convenience accessors ────────────────────────────

    def joint(self, name: str) -> Point:
        return tuple(self.joints[name])

    def box(self, name: str) -> Box:
        return tuple(int(v) for v in self.face[name])

    def color(self, name: str) -> Tuple[int, int, int]:
        return tuple(int(v) for v in self.face[name])

    # ─── v3 accessors + the render-path gate ──────────────

    def is_v3(self) -> bool:
        """True when the v3 geometry is present AND self-consistent.
        Presence of the version integer alone is not trusted — a rig is
        v3 only if it actually carries a head plate and registered poses.
        """
        if self.version < 3 or self.head is None:
            return False
        if len(self.head.landmarks) != N_LANDMARKS:
            return False
        if not self.poses:
            return False
        return all(p.xform and p.headless for p in self.poses.values())

    def require_v3(self) -> None:
        """Hard gate for the render path (Part III). Loud, actionable."""
        if self.is_v3():
            return
        why = "missing head plate" if self.head is None else (
            f"version {self.version}" if self.version < 3 else
            "incomplete pose registration")
        raise RigVersionError(
            f"character '{self.character}': rig is not v3 ({why}). "
            f"Run `jvmake rig --force` to rebuild with multi-pose "
            f"registration. Rendering a pre-v3 rig is refused because it "
            f"would place the face using body.png's boxes on every pose "
            f"(defect D1).")

    def pose_xform(self, name: str):
        """This pose's canonical→pose SimilarityTransform (identity for
        the canonical pose itself, so callers never special-case it)."""
        from engine.registration import SimilarityTransform
        entry = self.poses.get(name)
        if entry is None or not entry.xform:
            return SimilarityTransform.identity()
        return SimilarityTransform.from_dict(entry.xform)

    def pose_file(self, name: str, kind: str) -> Optional[str]:
        """Absolute path of a pose bake ('headless' | 'headmask' |
        'occluder'), or None when that bake does not exist."""
        entry = self.poses.get(name)
        if entry is None:
            return None
        rel = getattr(entry, kind, None)
        if not rel:
            return None
        path = os.path.join(rig_dir(self.character), rel)
        return path if os.path.exists(path) else None

    def head_plate_path(self) -> Optional[str]:
        if self.head is None:
            return None
        path = os.path.join(rig_dir(self.character), self.head.plate)
        return path if os.path.exists(path) else None

    def worst_pose_rms(self) -> float:
        """Largest post-fit registration RMS across poses — the number
        the rig-sanity QC gate asserts against its budget."""
        return max((float(p.xform.get("rms", 0.0))
                    for p in self.poses.values()), default=0.0)

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
        d = {
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
        # v3 blocks are omitted entirely when absent so a v2 rig
        # round-trips unchanged instead of gaining empty stubs that
        # `is_v3()` would then have to disambiguate.
        if self.head is not None:
            d["canonical_pose"] = self.canonical_pose
            d["head"] = self.head.to_dict()
        if self.mouth_targets:
            d["mouth_targets"] = {k: dict(v)
                                  for k, v in self.mouth_targets.items()}
        if self.poses:
            d["poses"] = {k: v.to_dict() for k, v in self.poses.items()}
        return d

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
        # v3 geometry — absent on v1/v2 rigs, which stay readable (for
        # /studio and archived bundles) but not renderable.
        rig.canonical_pose = d.get("canonical_pose", "neutral")
        if "head" in d:
            rig.head = HeadGeometry.from_dict(d["head"])
        rig.mouth_targets = {k: dict(v) for k, v
                             in d.get("mouth_targets", {}).items()}
        rig.poses = {k: PoseEntry.from_dict(k, v)
                     for k, v in d.get("poses", {}).items()}
        return rig

    def layer_path(self, name: str) -> str:
        return os.path.join(rig_dir(self.character), self.layers[name].file)

    def viseme_path(self, name: str) -> Optional[str]:
        f = self.visemes.get(name)
        return os.path.join(rig_dir(self.character), f) if f else None
