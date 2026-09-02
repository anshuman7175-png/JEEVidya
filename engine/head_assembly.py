"""
JEEVidya — Unified Head Assembly (Terminal Plan, Part VI)
════════════════════════════════════════════════════════
THE one head-compose path. Defect D4 existed because `_compose_head`
and `_staged_head` coexisted with a flat `render()` that ignored both;
two implementations of one idea always diverge, and the divergence put a
mouth on a pair of eyes. Law 1 says make that unrepresentable: this
module is the only place a head is built, and `engine/bone_engine.py`
calls it. There is nothing left to diverge from.

Per frame, exactly:

  1. body   = crossfade(headless[from], headless[to], eased_t)   # D2
  2. plate  = head_plate ⊕ brows ⊕ eyes/lids ⊕ parametric mouth  # D3/D5/D9
  3. M      = compose(interp(xform[from], xform[to], eased_t), …) # D1/D4
             head = plate.transform(ONE affine, BICUBIC)          # one resample
  4. body  ← alpha_composite(head, sub-pixel dest)
  5. body  ← alpha_composite(occluder[from→to])                   # hands in front

Both caches the plan requires live here: level 1 (face-channel key →
composed plate) and level 2 (plate id + quantized affine → transformed
head, in `engine/head_transform.TransformCache`).

Every registration prediction QC needs is exposed by `predict()`, which
runs the SAME affine the renderer used — so a passing gate proves the
renderer, not a parallel model of it.
"""
from __future__ import annotations

import math
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from engine import head_transform as ht
from engine.eye_model import EyeGeometry, EyePair, EyeState
from engine.mouth_model import MouthParams
from engine.mouth_raster import MouthRasterizer
from engine.registration import SimilarityTransform
from engine.rig import Rig, rig_dir

# Brow polyline warp gain: fraction of the brow's own span the raise
# travels at |brow| = 1. Conservative — brows that fly are a cartoon tell.
BROW_RAISE_GAIN = 0.30

# Face-channel cache quantization. Fine enough to be invisible, coarse
# enough that steady-state speech is a cache hit.
_Q_BROW = 1 / 32
_Q_EYE = 1 / 32


@dataclass(frozen=True)
class FaceChannels:
    """Everything that changes INSIDE the head plate for one frame."""
    mouth: MouthParams = field(default_factory=MouthParams)
    viseme_class: str = "REST"
    eyes: EyeState = field(default_factory=EyeState)
    brow: float = 0.0

    def key(self) -> Tuple:
        return (self.mouth.quantized_key(), self.viseme_class,
                self.eyes.quantized_key(), round(self.brow / _Q_BROW) * _Q_BROW)


class HeadAssembly:
    """Owns the head plate, the eye pair, the mouth rasterizer and both
    caches for ONE character. v3-only by contract: a pre-v3 rig raises
    instead of silently placing the face with body.png's boxes (D1)."""

    def __init__(self, rig: Rig, scale: float = 1.0, fps: int = 60,
                 seed: Optional[str] = None, plate_cache: int = 64,
                 depth: bool = False):
        rig.require_v3()
        self.rig = rig
        self.scale = float(scale)
        geo = rig.head
        d = rig_dir(rig.character)

        plate_path = rig.head_plate_path()
        if plate_path is None:
            raise FileNotFoundError(
                f"character '{rig.character}': rig claims v3 but "
                f"{geo.plate} is missing. Run `jvmake rig --force`.")
        self.plate = self._load(plate_path)

        # Plate origin in puppet space, and plate size in work space.
        self.plate_offset = (geo.offset[0] * self.scale,
                             geo.offset[1] * self.scale)
        self.plate_size = self.plate.size
        self.face_height = geo.face_height * self.scale

        shading = None
        if geo.shading:
            sp = os.path.join(d, geo.shading)
            if os.path.exists(sp):
                shading = self._load(sp).convert("L")

        pal = geo.palette
        self.mouth = MouthRasterizer(self._pts(geo.lip_outer), pal, shading)
        geo_l = EyeGeometry.from_rig_dict(self._scaled_eye(geo.eye_dict(True)))
        geo_r = EyeGeometry.from_rig_dict(self._scaled_eye(geo.eye_dict(False)))
        # The eye's THREE art layers are loaded HERE, through the same
        # `_load` that scales the plate, so their pixels and the eye
        # geometry are always in one space (`_scaled_eye` scaled the
        # matching origins). Together they remove every flat palette fill
        # from inside the eye: the eyeball is the artist's drawn iris, the
        # socket is what gaze uncovers behind it, and the lid is the
        # artist's own eyelid skin that slides down to blink.
        self.eyes = EyePair(
            geo_l, geo_r, palette=pal, seed=seed or rig.character, fps=fps,
            sprite_l=self._eye_img(d, geo_l.eyeball),
            sprite_r=self._eye_img(d, geo_r.eyeball),
            socket_l=self._eye_img(d, geo_l.socket_img),
            socket_r=self._eye_img(d, geo_r.socket_img),
            lid_l=self._eye_img(d, geo_l.lid_img),
            lid_r=self._eye_img(d, geo_r.lid_img))

        # Brow polylines in plate space (warped, never patch-pasted: a
        # feathered ellipse patch was how brow ghosting reached the eyes)
        self.brow_l = self._pts(geo.brow_l)
        self.brow_r = self._pts(geo.brow_r)

        # Art viseme sprites, when the character has them: geometry from
        # the model, pixels from the artwork (Part IV §4.3).
        self.art: Dict[str, Image.Image] = {}
        for name, fname in (rig.visemes or {}).items():
            if name.startswith("LID_"):
                continue
            p = os.path.join(d, fname)
            if os.path.exists(p):
                try:
                    self.art[name] = self._load(p)
                except Exception:
                    pass

        # Level-1 cache: face channels → composed plate
        self._plates: "OrderedDict[Tuple, Image.Image]" = OrderedDict()
        self._plate_cap = int(plate_cache)
        self.plate_hits = 0
        self.plate_misses = 0
        # Level-2 cache: (plate key, quantized affine) → transformed head
        self._xcache = ht.TransformCache()

        # Headless bodies + occluders, lazily loaded per pose
        self._headless: Dict[str, Image.Image] = {}
        self._occluders: Dict[str, Optional[Image.Image]] = {}

        # 2.5D depth head (Part XV) — behind a flag; when off the affine
        # path renders identically to the Terminal core.
        self.depth = None
        if depth:
            try:
                from engine.depth_head import DepthHead
                self.depth = DepthHead(
                    np.asarray(self._pts(geo.landmarks), dtype=np.float64),
                    np.asarray(self.plate.getchannel("A")))
            except Exception as e:      # never let the ceiling break the floor
                print(f"  [HeadAssembly] depth head disabled: {e}")
                self.depth = None

    # ─── loading helpers ──────────────────────────────────

    def _load(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGBA")
        if self.scale != 1.0:
            img = img.resize((max(1, int(img.width * self.scale)),
                              max(1, int(img.height * self.scale))),
                             Image.LANCZOS)
        return img

    def _pts(self, pts) -> list:
        return [(p[0] * self.scale, p[1] * self.scale) for p in pts]

    def _eye_img(self, d: str, fname: str) -> Optional[Image.Image]:
        """One baked eye asset (eyeball / socket backdrop / lid strip).

        All three go through `_load`, so their pixels land in the same
        space as the eye geometry whose origins `_scaled_eye` scaled.

        Missing pixels are not fatal: the rasterizer falls back to the
        synthetic eye, which is worse-looking but correct. A hard failure
        here would take down a render for a cosmetic asset.
        """
        if not fname:
            return None
        p = os.path.join(d, fname)
        if not os.path.exists(p):
            return None
        try:
            return self._load(p)
        except Exception as e:
            print(f"  [HeadAssembly] eye asset {fname} unusable: {e}")
            return None

    def _scaled_eye(self, d: dict) -> dict:
        """Scale one eye payload into work space.

        Keys are scaled by KIND, not by position: `colors` is RGB and must
        pass through untouched, `iris_angle` is degrees (scale-invariant),
        `iris_axes` is a length pair, `eyeball` is a FILENAME, and
        everything else is a point list. Scaling blindly here would
        multiply a colour channel by the render scale and tint the eye —
        and would iterate a filename character by character.
        """
        s = self.scale
        out: dict = {}
        for k, v in d.items():
            if k == "iris":
                cx, cy, r = v
                out[k] = [cx * s, cy * s, r * s]
            elif k == "iris_axes":
                # A LENGTH pair (semi-axes in px), not a point, but it scales
                # identically with the render.
                out[k] = [float(v[0]) * s, float(v[1]) * s]
            elif k == "gaze_box":
                # Four LENGTHS — (left, right, up, down) travel budget in px.
                # It must be matched by name: the fallback branch below would
                # try to unpack each float as an (x, y) point and raise.
                out[k] = [float(t) * s for t in v]
            elif k in ("iris_angle",):
                out[k] = float(v)
            elif k == "colors":
                out[k] = {ck: list(cv) for ck, cv in (v or {}).items()}
            elif k in ("eyeball", "socket_img", "lid_img"):
                out[k] = str(v or "")
            elif k in ("eyeball_origin", "socket_origin", "lid_origin"):
                # A single point, not a list of them. It must scale with
                # the sprite `_load` resizes, or the eyeball pastes off
                # the eye at any render scale but 1.0.
                out[k] = [float(v[0]) * s, float(v[1]) * s]
            else:
                out[k] = [[x * s, y * s] for x, y in v]
        return out

    def headless(self, pose: str) -> Optional[Image.Image]:
        if pose not in self._headless:
            path = self.rig.pose_file(pose, "headless")
            self._headless[pose] = self._load(path) if path else None
        return self._headless[pose]

    def occluder(self, pose: str) -> Optional[Image.Image]:
        if pose not in self._occluders:
            path = self.rig.pose_file(pose, "occluder")
            self._occluders[pose] = self._load(path) if path else None
        return self._occluders[pose]

    # ─── step 1: the headless body cross-fade (D2) ────────

    def body(self, from_pose: str, to_pose: str, blend_t: float
             ) -> Optional[Image.Image]:
        """Cross-fade two HEADLESS bodies. Both layers fade symmetrically
        in lockstep with the head affine transform."""
        a = self.headless(from_pose) or self.headless(self.rig.canonical_pose)
        b = self.headless(to_pose) or a
        if a is None:
            return None
        t = max(0.0, min(1.0, blend_t))
        if b is None or b is a or t <= 0.005:
            return a.copy()
        if t >= 0.995:
            return b.copy()

        # True symmetric alpha cross-fade
        a_arr = np.array(a, dtype=np.float32)
        b_arr = np.array(b, dtype=np.float32)
        blended = a_arr * (1.0 - t) + b_arr * t
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    # ─── step 2: the composed plate (D3/D5/D9/D10) ────────

    def compose_plate(self, ch: FaceChannels) -> Image.Image:
        key = ch.key()
        hit = self._plates.get(key)
        if hit is not None:
            self._plates.move_to_end(key)
            self.plate_hits += 1
            return hit
        self.plate_misses += 1
        plate = self._build_plate(ch)
        self._plates[key] = plate
        if len(self._plates) > self._plate_cap:
            self._plates.popitem(last=False)
        return plate

    def _build_plate(self, ch: FaceChannels) -> Image.Image:
        plate = self.plate.copy()
        self._draw_brows(plate, ch.brow)
        self.eyes.composite(plate, ch.eyes)
        art = self.art.get(ch.viseme_class)
        if art is not None:
            mcx, mcy = self.mouth.center
            px = int(round(mcx - art.width / 2.0))
            py = int(round(mcy - art.height / 2.0))
            plate.alpha_composite(art, (px, py))
        else:
            self.mouth.composite(plate, ch.mouth, ch.viseme_class)
        return plate

    def _draw_brows(self, plate: Image.Image, brow: float) -> None:
        """Warp the baked brow polylines instead of sliding a patch.

        The patch is masked to the BROW'S OWN STROKE — a thick polyline
        along the baked brow, feathered in every direction — not to its
        bounding box. Feathering only the top and bottom of a rectangle
        (what this did before) leaves two hard vertical edges, which is
        exactly the pale rectangle that showed over gudiya's brow.

        The measured eye apertures are then subtracted from that mask, so
        a brow raise CANNOT paint inside the drawn eye however large the
        gain grows. Smearing skin across the eyes stops being a bug to
        avoid and becomes a state the code cannot represent.
        """
        if abs(brow) < 0.02:
            return
        for poly in (self.brow_l, self.brow_r):
            if len(poly) < 2:
                continue
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            x0, x1 = int(math.floor(min(xs))), int(math.ceil(max(xs)))
            y0, y1 = int(math.floor(min(ys))), int(math.ceil(max(ys)))
            span = max(2.0, (y1 - y0))
            dy = -brow * BROW_RAISE_GAIN * span
            # Only as much room as the stroke plus its travel and feather.
            pad = int(math.ceil(span * 1.1 + abs(dy)))
            box = (max(0, x0 - pad), max(0, y0 - pad),
                   min(plate.width, x1 + pad), min(plate.height, y1 + pad))
            if box[2] - box[0] < 3 or box[3] - box[1] < 3:
                continue
            patch = plate.crop(box)
            shifted = patch.transform(patch.size, Image.AFFINE,
                                      (1, 0, 0, 0, 1, -dy),
                                      resample=Image.BICUBIC)

            stroke = Image.new("L", patch.size, 0)
            sd = ImageDraw.Draw(stroke)
            sd.line([(p[0] - box[0], p[1] - box[1] + dy) for p in poly],
                    fill=255, width=max(3, int(span * 1.25)), joint="curve")
            # Round the ends so the stroke has no square corner to show.
            r = max(2, int(span * 0.62))
            for p in (poly[0], poly[-1]):
                cx, cy = p[0] - box[0], p[1] - box[1] + dy
                sd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
            # The eye is off-limits, at bake-measured accuracy.
            for ras in (self.eyes.left, self.eyes.right):
                clip = ras.geo.clip
                if len(clip) >= 3:
                    sd.polygon([(p[0] - box[0], p[1] - box[1]) for p in clip],
                               fill=0)
            stroke = stroke.filter(ImageFilter.GaussianBlur(
                max(1.0, span * 0.30)))

            a = (np.asarray(shifted.getchannel("A"), dtype=np.float32)
                 * (np.asarray(stroke, dtype=np.float32) / 255.0))
            shifted.putalpha(Image.fromarray(
                np.clip(a, 0, 255).astype(np.uint8)))
            plate.alpha_composite(shifted, (box[0], box[1]))

    # ─── step 3: THE affine ───────────────────────────────

    def affine(self, from_pose: str, to_pose: str, blend_t: float,
               head: ht.HeadPose) -> ht.ComposedAffine:
        """The single composed head→body affine for this frame.

        `M_pose` is interpolated with the SAME eased blend_t the body
        cross-fade uses, so the head travels in lockstep through every
        transition — that lockstep IS the D1 fix.
        """
        xa = self.rig.pose_xform(from_pose)
        xb = self.rig.pose_xform(to_pose)
        xf = self._rescale(xa.lerp(xb, max(0.0, min(1.0, blend_t))))
        # Fold the plate origin INTO the pose similarity. Plate space +
        # offset IS canonical body space, and P(x + o) = sR·x + (sR��o + t)
        # is still a similarity — so the composed affine maps plate
        # pixels straight to canvas pixels with no second translate to
        # get wrong. Adding the offset AFTER the rotation (the obvious
        # mistake) drifts the head as soon as any pose has roll.
        ox, oy = self.plate_offset
        tx, ty = xf.apply_point(ox, oy)
        xf = SimilarityTransform(xf.s, xf.theta, tx, ty, xf.rms)
        return ht.compose(xf, head, self.plate_size, self._pivot())

    def _rescale(self, xf: SimilarityTransform) -> SimilarityTransform:
        """Pose transforms were fitted in ORIGINAL art pixels; the work
        canvas may be downscaled. Translation scales, rotation and scale
        are invariant — getting this wrong is a silent sub-pixel drift,
        so it is done once, here, and nowhere else."""
        if self.scale == 1.0:
            return xf
        return SimilarityTransform(s=xf.s, theta=xf.theta,
                                   tx=xf.tx * self.scale,
                                   ty=xf.ty * self.scale,
                                   rms=getattr(xf, "rms", 0.0))

    def _pivot(self) -> Tuple[float, float]:
        """Neck joint in plate space: the pin a cut-out puppet turns on."""
        neck = self.rig.joints.get("neck")
        if neck is None:
            return (self.plate_size[0] / 2.0, float(self.plate_size[1]))
        return (neck[0] * self.scale - self.plate_offset[0],
                neck[1] * self.scale - self.plate_offset[1])

    # ─── step 4/5: the frame ──────────────────────────────

    def render(self, ch: FaceChannels, head: ht.HeadPose,
               from_pose: str, to_pose: str, blend_t: float,
               canvas_size: Tuple[int, int]) -> Image.Image:
        """Assemble one full-body frame. Exactly one head resample."""
        body = self.body(from_pose, to_pose, blend_t)
        if body is None:
            body = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        elif body.size != canvas_size:
            frame = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
            frame.alpha_composite(body, (0, 0))
            body = frame

        plate = self.compose_plate(ch)
        aff = self.affine(from_pose, to_pose, blend_t, head)

        # Feature parallax lives INSIDE head space, before the affine.
        if abs(aff.face_dx) > 1e-4 or abs(aff.face_dy) > 1e-4:
            plate = ht.shift_features_subpixel(plate, aff.face_dx, aff.face_dy)
        if self.depth is not None and (abs(head.yaw) > 1e-3
                                       or abs(head.nod) > 1e-3):
            plate = self.depth.warp(plate, head.yaw * ht.YAW_PARALLAX_GAIN * 15.0,
                                    head.nod * 3.0)

        # `aff` already maps plate pixels → canvas pixels (the plate
        # origin is folded into the pose similarity in `affine()`).
        head_img = self._xcache.transform(plate, ch.key(), aff, canvas_size)
        body.alpha_composite(head_img, (0, 0))

        # Step 5: occluder cross-fade — hands/props that must render IN
        # FRONT of the head are composited here, blended with the same
        # eased blend_t the body and head use. Without this, the occluder
        # snaps from one pose's hands to the other's during transitions,
        # creating a flash/discontinuity.
        t = max(0.0, min(1.0, blend_t))
        occ_a = self.occluder(from_pose)
        occ_b = self.occluder(to_pose) if to_pose != from_pose else occ_a
        if occ_a is not None or occ_b is not None:
            if occ_a is None:
                occ_a = occ_b
            if occ_b is None:
                occ_b = occ_a
            if occ_a is occ_b or t <= 0.005:
                body.alpha_composite(occ_a, (0, 0))
            elif t >= 0.995:
                body.alpha_composite(occ_b, (0, 0))
            else:
                # Linear blend in float space, same as body()
                oa = np.array(occ_a, dtype=np.float32)
                ob = np.array(occ_b, dtype=np.float32)
                if oa.shape == ob.shape:
                    occ_blend = np.clip(oa * (1.0 - t) + ob * t,
                                        0, 255).astype(np.uint8)
                    body.alpha_composite(
                        Image.fromarray(occ_blend), (0, 0))
                else:
                    # Different sizes: just composite the dominant one
                    body.alpha_composite(
                        occ_b if t > 0.5 else occ_a, (0, 0))
        return body

    # ─── QC surface ───────────────────────���───────────────

    def predict(self, ch: FaceChannels, head: ht.HeadPose,
                from_pose: str, to_pose: str, blend_t: float
                ) -> Dict[str, Tuple[float, float]]:
        """Where the mouth centroid and both iris centres MUST land, in
        canvas coordinates, for exactly these channels. QC compares
        re-detected pixels against this; the renderer and the prediction
        share `affine()`, so the gate cannot pass a lie."""
        aff = self.affine(from_pose, to_pose, blend_t, head)

        def feat(x: float, y: float) -> Tuple[float, float]:
            return aff.apply_feature_point(x, y)

        mcx, mcy = self.mouth.predicted_centroid(ch.mouth)
        il = self.eyes.left.geo
        ir = self.eyes.right.geo
        # The excursion MUST come from the same `gaze_offset` the renderer
        # uses. Re-deriving it here as `iris_r * 0.55` duplicated the legacy
        # guess in the gate itself, so the prediction and the pixels could
        # only ever agree while both were wrong, and the gate would have
        # certified a measured renderer as broken.
        dxl, dyl = il.gaze_offset(ch.eyes.eye_dx, ch.eyes.eye_dy)
        dxr, dyr = ir.gaze_offset(ch.eyes.eye_dx, ch.eyes.eye_dy)
        gaze_l = (il.iris_c[0] + dxl, il.iris_c[1] + dyl)
        gaze_r = (ir.iris_c[0] + dxr, ir.iris_c[1] + dyr)
        return {"mouth": feat(mcx, mcy),
                "iris_l": feat(*gaze_l),
                "iris_r": feat(*gaze_r)}

    def cache_report(self) -> Dict[str, float]:
        n = self.plate_hits + self.plate_misses
        return {"plate_hit_rate": self.plate_hits / n if n else 0.0,
                "mouth_hit_rate": self.mouth.hit_rate,
                "affine_hit_rate": (
                    self._xcache.hits /
                    max(1, self._xcache.hits + self._xcache.misses))}


__all__ = ["HeadAssembly", "FaceChannels", "BROW_RAISE_GAIN"]
