"""
JEEVidya V5 — Bone Engine (Tier 1)
══════════════════════════════════
True 2.5D skeletal puppet rendering from a single character image.

  Skeleton   2 bones (spine: hips→neck, neck: neck→head) solved with
             forward kinematics every frame.
  Torso      bent through a 12-strip PIL quad-MESH transform — the spine
             curves (quadratic bend profile), it doesn't hinge.
  Head       composed head-local (yaw squash + parallax face shift,
             viseme mouth, eyelid blink, brow raise) then rotated about
             the NECK pivot — exactly like a cut-out puppet on a pin.
  Physics    a verlet spring chain hangs from the head bone; its
             deflection adds rotational follow-through + a hair-lag
             shear. Motion overshoots and settles like mass, not math.

Hot paths are content-addressed LRU caches keyed by quantized pose
channels, so steady-state rendering is paste-only.
"""
from __future__ import annotations

import math
import os
import numpy as np
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter

from engine.rig import (Rig, rig_dir, LEGACY_VISEME_ALIAS, VISEME_FALLBACK,
                        VISEME_NAMES)

# Working resolution cap: puppet layers are downscaled so the canvas is at
# most this tall. Keeps the pose caches small enough to live in RAM.
MAX_WORK_HEIGHT = 1600

# Cache sizes (entries)
_TORSO_LRU = 20
_HEAD_LRU = 48
_ROT_LRU = 28

# Quantization steps for cache keys
_Q_LEAN = 1.0        # degrees
_Q_OPEN = 1 / 6      # mouth openness levels
_Q_BLINK = 1 / 3
_Q_BROW = 1 / 4
_Q_YAW = 1 / 5
_Q_ANGLE = 0.6       # final head rotation, degrees
_Q_LAG = 1.5         # hair-lag shear, px
_Q_VBLEND = 0.2      # coarticulation cross-fade levels


# ═══════════════════════════════════════════
# POSE
# ═══════════════════════════════════════════

@dataclass
class PuppetPose:
    """Every animatable channel of the puppet for one frame."""
    lean: float = 0.0        # spine bend, degrees (+ = screen right)
    head_tilt: float = 0.0   # head roll, degrees
    head_yaw: float = 0.0    # −1..1 fake 3D turn
    head_nod: float = 0.0    # −1 (up) .. 1 (down)
    bounce: float = 0.0      # whole-body y offset px (− = up)
    sway: float = 0.0        # whole-body x offset px
    squash: float = 0.0      # scale-y delta (−0.1 = squashed, +0.1 = stretched)
    viseme: str = "REST"
    viseme_to: str = "REST"     # incoming shape during coarticulation
    viseme_blend: float = 0.0   # 0 = fully `viseme`, 1 = fully `viseme_to`
    mouth_open: float = 0.0  # 0..1
    blink: float = 0.0       # 0..1
    brow: float = 0.0        # −1..1
    energy: float = 1.0
    eye_dx: float = 0.0      # micro-saccade x offset (px, head-local)
    eye_dy: float = 0.0      # micro-saccade y offset (px, head-local)
    body_pose: str = ""      # active pose name (for alt torso selection)
    body_pose_to: str = ""   # target pose name during cross-fade
    body_pose_blend: float = 0.0 # 0..1 blend progress between pose and pose_to

    def clamped(self) -> "PuppetPose":
        self.lean = max(-14.0, min(14.0, self.lean))
        self.head_tilt = max(-18.0, min(18.0, self.head_tilt))
        self.head_yaw = max(-1.0, min(1.0, self.head_yaw))
        self.head_nod = max(-1.0, min(1.0, self.head_nod))
        self.squash = max(-0.18, min(0.18, self.squash))
        self.mouth_open = max(0.0, min(1.0, self.mouth_open))
        self.viseme_blend = max(0.0, min(1.0, self.viseme_blend))
        self.blink = max(0.0, min(1.0, self.blink))
        self.brow = max(-1.0, min(1.0, self.brow))
        return self


# ═══════════════════════════════════════════
# VERLET SPRING CHAIN (secondary motion)
# ═══════════════════════════════════════════

class VerletChain:
    """
    A hanging chain of point masses pinned to a bone. Verlet integration
    + distance constraints → hair/dupatta follow-through and overshoot,
    unconditionally stable at our time step.
    """

    def __init__(self, anchor: Tuple[float, float], links: int = 4,
                 seg_len: float = 14.0, damping: float = 0.88,
                 gravity: float = 0.9, iterations: int = 3):
        self.seg_len = seg_len
        self.damping = damping
        self.gravity = gravity
        self.iterations = iterations
        ax, ay = anchor
        # Rest state: hanging straight down
        self.px = [ax] * (links + 1)
        self.py = [ay + i * seg_len for i in range(links + 1)]
        self.ox = list(self.px)     # previous positions
        self.oy = list(self.py)

    def update(self, anchor_x: float, anchor_y: float) -> None:
        n = len(self.px)
        # Integrate free points
        for i in range(1, n):
            vx = (self.px[i] - self.ox[i]) * self.damping
            vy = (self.py[i] - self.oy[i]) * self.damping
            self.ox[i], self.oy[i] = self.px[i], self.py[i]
            self.px[i] += vx
            self.py[i] += vy + self.gravity
        # Pin the anchor
        self.ox[0], self.oy[0] = self.px[0], self.py[0]
        self.px[0], self.py[0] = anchor_x, anchor_y
        # Satisfy distance constraints
        for _ in range(self.iterations):
            for i in range(1, n):
                dx = self.px[i] - self.px[i - 1]
                dy = self.py[i] - self.py[i - 1]
                dist = math.hypot(dx, dy) or 1e-6
                diff = (dist - self.seg_len) / dist
                if i == 1:      # parent is pinned — child absorbs all error
                    self.px[i] -= dx * diff
                    self.py[i] -= dy * diff
                else:
                    self.px[i] -= dx * diff * 0.5
                    self.py[i] -= dy * diff * 0.5
                    self.px[i - 1] += dx * diff * 0.5
                    self.py[i - 1] += dy * diff * 0.5

    def deflection(self) -> float:
        """Horizontal tail deflection normalized to −1..1 (0 = at rest)."""
        total = self.seg_len * (len(self.px) - 1)
        return max(-1.0, min(1.0, (self.px[-1] - self.px[0]) / max(1e-6, total)))


# ═══════════════════════════════════════════
# SKELETON (forward kinematics)
# ═══════════════════════════════════════════

class Skeleton:
    """The inferred 2-bone skeleton. FK maps pose angles → joint world
    positions, including how the neck travels when the spine bends."""

    def __init__(self, hips: Tuple[float, float], neck: Tuple[float, float],
                 head_center: Tuple[float, float]):
        self.hips = hips
        self.neck = neck
        self.head_center = head_center
        self.spine_len = max(1.0, hips[1] - neck[1])

    def bend_offset(self, y: float, lean_deg: float) -> float:
        """Quadratic spine-bend x-offset at height y (0 at hips)."""
        t = (self.hips[1] - y) / self.spine_len
        t = max(0.0, min(1.25, t))
        return math.sin(math.radians(lean_deg)) * self.spine_len * t * t * 0.5

    def solve(self, lean_deg: float) -> Dict[str, Tuple[float, float]]:
        """FK: world positions of every joint for a given spine bend."""
        nx = self.neck[0] + self.bend_offset(self.neck[1], lean_deg)
        hx = self.head_center[0] + self.bend_offset(self.neck[1], lean_deg)
        return {"hips": self.hips,
                "neck": (nx, self.neck[1]),
                "head_center": (hx, self.head_center[1])}


# ═══════════════════════════════════════════
# LRU helper
# ═══════════════════════════════════════════

class _LRU(OrderedDict):
    def __init__(self, cap: int):
        super().__init__()
        self.cap = cap

    def get_or(self, key, factory):
        if key in self:
            self.move_to_end(key)
            return self[key]
        val = factory()
        self[key] = val
        if len(self) > self.cap:
            self.popitem(last=False)
        return val


def _q(v: float, step: float) -> float:
    return round(v / step) * step


# ═══════════════════════════════════════════
# BONE ENGINE
# ═══════════════════════════════════════════

class BoneEngine:
    """Renders a rigged character in any PuppetPose to an RGBA canvas the
    same aspect as body.png (so all existing camera framing still holds)."""

    def __init__(self, rig: Rig):
        self.rig = rig
        w, h = rig.size
        self.scale = min(1.0, MAX_WORK_HEIGHT / max(1, h))
        self.width = int(w * self.scale)
        self.height = int(h * self.scale)

        d = rig_dir(rig.character)

        def load(path: str) -> Image.Image:
            img = Image.open(path).convert("RGBA")
            if self.scale != 1.0:
                img = img.resize((max(1, int(img.width * self.scale)),
                                  max(1, int(img.height * self.scale))),
                                 Image.Resampling.LANCZOS)
            return img

        def pt(p) -> Tuple[float, float]:
            return (p[0] * self.scale, p[1] * self.scale)

        def bx(b) -> Tuple[int, int, int, int]:
            return tuple(int(v * self.scale) for v in b)

        # Layers
        self.body_img = load(os.path.join(d, "..", "body.png"))
        self.torso = load(rig.layer_path("torso"))
        self.torso_off = pt(rig.layers["torso"].offset)
        head_img = load(rig.layer_path("head"))
        self.head_off = pt(rig.layers["head"].offset)

        # Sprites
        self.viseme_sprites: Dict[str, Image.Image] = {}
        self.lid_sprites: Dict[str, Image.Image] = {}
        for name, fname in rig.visemes.items():
            img = load(os.path.join(d, fname))
            if name.startswith("LID_"):
                self.lid_sprites[name[4:].lower()] = img
            else:
                self.viseme_sprites[name] = img

        # Resolve legacy sprite keys (rig v1 procedural bakes use
        # MBP/E/AI/O/FV) onto their 10-class names — the lip-sync
        # pipeline emits ONLY 10-class names (engine/visemes.py V enum),
        # so without this remap a procedural rig renders no mouth at
        # all during speech.
        for legacy, modern in LEGACY_VISEME_ALIAS.items():
            sp = self.viseme_sprites.get(legacy)
            if sp is not None and modern not in self.viseme_sprites:
                self.viseme_sprites[modern] = sp

        # Fill still-missing 10-class shapes from the nearest available
        # articulatory neighbour (engine/rig.py VISEME_FALLBACK).
        # Iterated to a fixpoint because chains reference each other
        # (e.g. CLOSED_I → MID_E → OPEN_A): each pass fills at least
        # one entry or stops, so it terminates in ≤ len(VISEME_NAMES)
        # passes. Shapes with no reachable fallback stay absent, which
        # the render path treats as "keep the base resting mouth".
        filled = True
        while filled:
            filled = False
            for name in VISEME_NAMES:
                if name in self.viseme_sprites:
                    continue
                for alt in VISEME_FALLBACK.get(name, ()):
                    sp = self.viseme_sprites.get(alt)
                    if sp is not None:
                        self.viseme_sprites[name] = sp
                        filled = True
                        break

        # Skeleton + geometry (work space)
        self.skel = Skeleton(pt(rig.joint("hips")), pt(rig.joint("neck")),
                             pt(rig.joint("head_center")))
        self.mouth_box = bx(rig.box("mouth"))
        self.eye_boxes = {"eye_l": bx(rig.box("eye_l")),
                          "eye_r": bx(rig.box("eye_r"))}
        self.brow_boxes = {"brow_l": bx(rig.box("brow_l")),
                           "brow_r": bx(rig.box("brow_r"))}
        self.skin = rig.color("skin")
        self.hair_line = rig.params.get("hair_line_y", 0) * self.scale

        # Brow patches cropped from head_img with feathered alpha mask (no rectangular edges)
        self._brow_patches: Dict[str, Image.Image] = {}
        # World-space paste position of each patch at brow=0. Must be the
        # exact crop origin mapped back to canvas coords — pasting at the
        # raw box top-left ignores the pad_y crop extension and shifts the
        # feathered skin patch down onto the eyes (nose-band ghosting).
        self._brow_patch_pos: Dict[str, Tuple[int, int]] = {}
        for name, box in list(self.brow_boxes.items()):
            local = self._to_head_local(box)
            pad_y = max(2, (local[3] - local[1]) // 2)
            lb = (max(0, local[0]), max(0, local[1] - pad_y),
                  min(head_img.width, local[2]),
                  min(head_img.height, local[3] + pad_y))
            self._brow_patch_pos[name] = (int(lb[0] + self.head_off[0]),
                                          int(lb[1] + self.head_off[1]))
            raw_patch = head_img.crop(lb)

            # Feather outer 25% to eliminate rectangular box edges
            w_p, h_p = raw_patch.size
            mask_p = Image.new("L", (w_p, h_p), 0)
            draw_p = ImageDraw.Draw(mask_p)
            fx_p, fy_p = max(2, int(w_p * 0.25)), max(2, int(h_p * 0.25))
            draw_p.ellipse([fx_p, fy_p, w_p - fx_p - 1, h_p - fy_p - 1], fill=255)
            mask_p = mask_p.filter(ImageFilter.GaussianBlur(radius=max(2, min(fx_p, fy_p) * 0.8)))
            r_p, g_p, b_p, a_p = raw_patch.split()
            blended_a = (np.asarray(a_p, dtype=np.float32) * np.asarray(mask_p, dtype=np.float32) / 255.0).astype(np.uint8)
            feathered_patch = raw_patch.copy()
            feathered_patch.putalpha(Image.fromarray(blended_a))
            self._brow_patches[name] = feathered_patch

        # Padded head base
        self.head_pad = int(max(head_img.width, head_img.height) * 0.18)
        p = self.head_pad
        self.head_base = Image.new(
            "RGBA", (head_img.width + 2 * p, head_img.height + 2 * p),
            (0, 0, 0, 0))
        self.head_base.alpha_composite(head_img, dest=(p, p))
        # Neck pivot inside the padded head sprite
        self.pivot = (self.skel.neck[0] - self.head_off[0] + p,
                      self.skel.neck[1] - self.head_off[1] + p)

        # Physics: spring chain hanging from the head bone
        seg = max(8.0, self.height * 0.012)
        self.chain = VerletChain(self.skel.head_center, links=4, seg_len=seg)

        # Caches
        self._torso_cache = _LRU(_TORSO_LRU)
        self._head_cache = _LRU(_HEAD_LRU)
        self._rot_cache = _LRU(_ROT_LRU)

        # Alt torsos & foreground hand overlays for pose library
        self._alt_torsos: Dict[str, Image.Image] = {}
        self._hand_overlays: Dict[str, Image.Image] = {}
        self._active_torso_name: str = ""

    def set_alt_torsos(self, torsos: Dict[str, Image.Image]) -> None:
        """Register alternative torso images and pre-compute hand overlays."""
        self._alt_torsos = torsos
        self._hand_overlays = {}
        try:
            import numpy as np
            # Base body crop matching get_torso() region
            crop_y0 = int(self.torso_off[1])
            crop_y1 = min(self.body_img.height if hasattr(self, 'body_img') else self.torso.height, crop_y0 + self.torso.height)
            if hasattr(self, 'body_img'):
                base_crop = self.body_img.crop((0, crop_y0, self.body_img.width, crop_y1))
            else:
                base_crop = self.torso
            base_arr = np.asarray(base_crop.convert("RGBA"), dtype=np.float32)

            # Head rect in local crop coordinates
            hx0, hy0 = int(self.head_off[0]), int(self.head_off[1] - self.torso_off[1])
            hx1, hy1 = int(self.head_off[0] + self.head_base.width), int(self.head_off[1] + self.head_base.height - self.torso_off[1])

            for name, t_img in torsos.items():
                if t_img is self.torso:
                    continue
                arr_pose = np.asarray(t_img.convert("RGBA"), dtype=np.float32)
                if arr_pose.shape != base_arr.shape:
                    continue
                diff = np.abs(arr_pose[..., :3] - base_arr[..., :3]).mean(axis=-1)
                hand_mask = (diff > 25) & (arr_pose[..., 3] > 50)

                fg_mask = np.zeros_like(hand_mask, dtype=bool)
                y0_cl, y1_cl = max(0, hy0), min(arr_pose.shape[0], hy1)
                x0_cl, x1_cl = max(0, hx0), min(arr_pose.shape[1], hx1)
                fg_mask[y0_cl:y1_cl, x0_cl:x1_cl] = hand_mask[y0_cl:y1_cl, x0_cl:x1_cl]

                if fg_mask.any():
                    out_arr = np.zeros_like(arr_pose, dtype=np.uint8)
                    out_arr[fg_mask] = arr_pose[fg_mask].astype(np.uint8)
                    self._hand_overlays[name] = Image.fromarray(out_arr, "RGBA")
        except Exception:
            pass

    def _get_torso(self, pose_name: str) -> Image.Image:
        """Get the torso for the given pose, falling back to default."""
        if pose_name and pose_name in self._alt_torsos:
            return self._alt_torsos[pose_name]
        return self.torso

    # ─── coordinate helpers ───────────────────────────────

    def _to_head_local(self, box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        ox, oy = self.head_off
        return (int(box[0] - ox), int(box[1] - oy),
                int(box[2] - ox), int(box[3] - oy))

    def _make_backing(self, size: Tuple[int, int], crop: Image.Image) -> Image.Image:
        from PIL import ImageDraw, ImageFilter
        import numpy as np
        w, h = max(4, size[0]), max(4, size[1])
        arr = np.asarray(crop)
        if arr.shape[-1] == 4:
            mask = arr[..., 3] > 50
            if mask.any():
                avg_color = tuple(arr[mask, :3].mean(axis=0).astype(int))
            else:
                avg_color = self.skin
        else:
            avg_color = self.skin
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(img).ellipse((0, 0, w - 1, h - 1), fill=avg_color + (255,))
        img.putalpha(img.split()[3].filter(
            ImageFilter.GaussianBlur(max(2, w // 10))))
        return img

    # ─── torso (quad-mesh spine bend) ─────────────────────

    def _bent_torso(self, lean: float) -> Image.Image:
        key = _q(lean, _Q_LEAN)
        return self._torso_cache.get_or(key, lambda: self._bend_torso(key))

    def _bend_torso(self, lean: float) -> Image.Image:
        if abs(lean) < 0.25:
            return self.torso
        w, h = self.torso.size
        oy = self.torso_off[1]
        strips = 12
        mesh = []
        for i in range(strips):
            y0 = int(h * i / strips)
            y1 = int(h * (i + 1) / strips) if i < strips - 1 else h
            # Source is shifted OPPOSITE the destination offset
            off0 = self.skel.bend_offset(y0 + oy, lean)
            off1 = self.skel.bend_offset(y1 + oy, lean)
            mesh.append(((0, y0, w, y1),
                         (-off0, y0, -off1, y1, w - off1, y1, w - off0, y0)))
        return self.torso.transform((w, h), Image.Transform.MESH, mesh,
                                    resample=Image.Resampling.BICUBIC)

    # ─── head (face composite + hair shear + rotation) ────

    def _compose_key(self, pose: PuppetPose) -> tuple:
        return (pose.viseme, _q(pose.mouth_open, _Q_OPEN),
                _q(pose.blink, _Q_BLINK), _q(pose.brow, _Q_BROW),
                _q(pose.head_yaw, _Q_YAW),
                _q(pose.eye_dx, 0.5), _q(pose.eye_dy, 0.5),
                pose.viseme_to, _q(pose.viseme_blend, _Q_VBLEND))

    def _composed_head(self, pose: PuppetPose) -> Image.Image:
        key = self._compose_key(pose)
        return self._head_cache.get_or(key, lambda: self._compose_head(*key))

    def _staged_head(self, pose: PuppetPose, angle_q: float,
                     lag_q: float) -> Tuple[Image.Image, Tuple[float, float]]:
        """Fully-staged head (compose → hair shear → rotate) with the
        paste pivot. All three stages cached by quantized channels."""
        key = self._compose_key(pose) + (angle_q, lag_q)

        def build():
            head = self._composed_head(pose)
            head = self._hair_shear(head, lag_q)
            squish = head.width / self.head_base.width
            pivot = (self.pivot[0] * squish, self.pivot[1])
            if abs(angle_q) > 0.2:
                head = head.rotate(angle_q,
                                   resample=Image.Resampling.BILINEAR,
                                   center=pivot, expand=False)
            return head, pivot

        return self._rot_cache.get_or(key, build)

    def _compose_head(self, viseme: str, mouth_open: float, blink: float,
                      brow: float, yaw: float,
                      saccade_dx: float = 0.0,
                      saccade_dy: float = 0.0,
                      viseme_to: str = "REST",
                      viseme_blend: float = 0.0) -> Image.Image:
        head = self.head_base.copy()
        p = self.head_pad
        ox, oy = self.head_off
        # 2.5D parallax: face features slide across the head with yaw
        face_dx = yaw * 0.05 * (head.width - 2 * p)

        def dest(box: Tuple[int, int, int, int], sprite: Image.Image,
                 extra_dy: float = 0.0) -> Tuple[int, int]:
            cx = (box[0] + box[2]) / 2 - ox + p + face_dx
            cy = (box[1] + box[3]) / 2 - oy + p + extra_dy
            return (int(cx - sprite.width / 2), int(cy - sprite.height / 2))

        # Brows (raise/furrow): composite eyebrow patches onto clean head_base
        for name, box in self.brow_boxes.items():
            patch = self._brow_patches[name]
            raise_px = -brow * 0.35 * (box[3] - box[1])
            head.alpha_composite(patch, dest=dest(box, patch, raise_px))

        # Eyes: eyelids glide smoothly down over the eye box during blinks
        if blink > 0.05:
            for name, box in self.eye_boxes.items():
                lid = self.lid_sprites.get(name)
                if lid is None:
                    continue
                x = int((box[0] + box[2]) / 2 - ox + p + face_dx + saccade_dx - lid.width / 2)
                y_closed = box[1] - oy + p - lid.height * 0.15 + saccade_dy
                y_open = y_closed - lid.height * 0.85
                y = int(y_open + (y_closed - y_open) * min(1.0, blink))
                # Clip the descending lid to the eye region (same fix
                # as the flat render() path): a mid-blink lid must not
                # hover over the brow as a translucent disc.
                clip_top = int(box[1] - oy + p - (box[3] - box[1]) * 0.35)
                if y < clip_top:
                    cut = clip_top - y
                    if cut >= lid.height:
                        continue
                    lid = lid.crop((0, cut, lid.width, lid.height))
                    y = clip_top
                head.alpha_composite(lid, dest=(x, y))

        # Mouth: coarticulated cross-fade between two viseme sprites.
        # The outgoing shape is drawn opaque; the incoming shape fades
        # in over it — the lips GLIDE between phonemes instead of
        # popping, exactly like a real mouth mid-word.
        def mouth_sprite(name: str) -> Optional[Image.Image]:
            sp = self.viseme_sprites.get(name)
            if sp is None:
                return None
            if mouth_open <= 0.03 and name not in ("REST", "BILABIAL"):
                return None
            return sp

        base_mouth = mouth_sprite(viseme)
        if base_mouth is not None:
            head.alpha_composite(base_mouth, dest=dest(self.mouth_box,
                                                       base_mouth))
        if viseme_to != viseme and viseme_blend > 0.05:
            incoming = mouth_sprite(viseme_to)
            if incoming is not None:
                faded = incoming.copy()
                faded.putalpha(faded.split()[3].point(
                    lambda p, b=viseme_blend: int(p * b)))
                head.alpha_composite(faded, dest=dest(self.mouth_box, faded))

        # Yaw squash: the head narrows as it turns
        if abs(yaw) > 0.05:
            new_w = int(head.width * (1.0 - abs(yaw) * 0.07))
            head = head.resize((new_w, head.height), Image.Resampling.BILINEAR)
        return head

    def _hair_shear(self, head: Image.Image, lag_px: float) -> Image.Image:
        """Shear everything above the hair line by the chain lag — hair
        trails behind fast head motion (follow-through)."""
        if abs(lag_px) < 0.75:
            return head
        w, h = head.size
        split = int(self.hair_line - self.head_off[1] + self.head_pad)
        split = max(4, min(h - 4, split))
        half = split // 2
        mesh = [
            ((0, 0, w, half),
             (-lag_px, 0, -lag_px * 0.5, half,
              w - lag_px * 0.5, half, w - lag_px, 0)),
            ((0, half, w, split),
             (-lag_px * 0.5, half, 0, split, w, split,
              w - lag_px * 0.5, half)),
            ((0, split, w, h), (0, split, 0, h, w, h, w, split)),
        ]
        return head.transform((w, h), Image.Transform.MESH, mesh,
                              resample=Image.Resampling.BILINEAR)

    # ─── physics step ─────────────────────────────────────

    def step_physics(self, pose: PuppetPose) -> Tuple[float, float]:
        """Advance the spring chain one frame; returns
        (overshoot_deg, hair_lag_px) driven by the chain deflection."""
        joints = self.skel.solve(pose.lean)
        hx, hy = joints["head_center"]
        self.chain.update(hx + pose.sway, hy + pose.bounce)
        defl = self.chain.deflection()
        return (-defl * 5.0, defl * min(14.0, self.height * 0.008))

    # ─── main render ──────────────────────────────────────

    def render(self, pose: PuppetPose,
               physics: Optional[Tuple[float, float]] = None) -> Image.Image:
        """Render the puppet in `pose` to an RGBA canvas."""
        pose = pose.clamped()

        # 1 · Base Body Canvas (Pristine 3D Pose Artwork with smooth S-curve cross-fade)
        if hasattr(self, "pose_lib") and self.pose_lib and self.pose_lib.has_poses:
            body_from = pose.body_pose or "neutral"
            body_to = pose.body_pose_to or body_from
            canvas = self.pose_lib.blended_body(body_from, body_to, pose.body_pose_blend)
            if canvas is not None:
                canvas = canvas.copy()
            else:
                canvas = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
                canvas.alpha_composite(self.body_img, dest=(0, 0))
        else:
            canvas = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
            canvas.alpha_composite(self.body_img, dest=(0, 0))

        # 2 · Facial Animation Pass: Eyelids, Brows, and 10-Class Viseme Mouth Overlay
        # face_dx = 0.0 (facial features locked 100% to character landmark coordinates)
        face_dx = 0.0

        # 2a. Eyelid Blinks (smooth descending vertical glide).
        # y_closed anchors to the eye-box TOP (matching _compose_head) so a
        # full blink covers the eye instead of landing on the cheek.
        if pose.blink > 0.05:
            for name, box in self.eye_boxes.items():
                lid = self.lid_sprites.get(name)
                if lid is None:
                    continue
                lx = int((box[0] + box[2]) / 2 + pose.eye_dx - lid.width / 2)
                y_closed = box[1] + pose.eye_dy - lid.height * 0.15
                y_open = y_closed - lid.height * 0.85
                ly = int(y_open + (y_closed - y_open) * min(1.0, pose.blink))
                # Clip the descending lid to the eye region: without
                # this, a mid-blink lid hangs ABOVE the eye over the
                # brow/forehead as a translucent skin disc ("goggles"
                # artifact). Only the portion that has actually slid
                # into the eye area is drawn.
                clip_top = int(box[1] - (box[3] - box[1]) * 0.35)
                if ly < clip_top:
                    cut = clip_top - ly
                    if cut >= lid.height:
                        continue
                    lid = _crop_lid_feathered(lid, cut, box[3] - box[1])
                    ly = clip_top
                canvas.alpha_composite(lid, dest=(lx, ly))

        # 2b. Brows: paste each patch at its exact crop-back position so it
        # realigns pixel-perfectly with the artwork underneath at brow=0
        # and only the raise offset moves it.
        if abs(pose.brow) > 0.08:
            for name, box in self.brow_boxes.items():
                patch = self._brow_patches.get(name)
                if patch is None:
                    continue
                px0, py0 = self._brow_patch_pos[name]
                raise_px = int(-pose.brow * 0.35 * (box[3] - box[1]))
                canvas.alpha_composite(patch, dest=(px0, py0 + raise_px))

        # 2c. Mouth Viseme Overlay (10-class Hindi viseme blend).
        # NO separate erase pass: every baked sprite (art bake AND
        # procedural bake) already carries its own feathered backing
        # that hides the base pose's resting mouth by construction.
        # A render-time flat skin ellipse on top of that is (a)
        # redundant, (b) erases the face's painted 3D shading into a
        # pale panel, and (c) — because the ellipse touched its canvas
        # bounds and PIL's GaussianBlur edge-extends at borders — left
        # hard straight alpha edges (the top/left "band" artifact).
        if pose.mouth_open > 0.03 or pose.viseme not in ("REST", "BILABIAL"):
            sp = self.viseme_sprites.get(pose.viseme)
            if sp is not None:
                mx0, my0, mx1, my1 = self.mouth_box
                mcx = (mx0 + mx1) // 2
                mcy = (my0 + my1) // 2

                # Composite animated mouth sprite (self-backed)
                mw, mh = sp.size
                canvas.alpha_composite(sp, dest=(int(mcx - mw / 2), int(mcy - mh * 0.42)))

                if pose.viseme_to != pose.viseme and pose.viseme_blend > 0.05:
                    incoming = self.viseme_sprites.get(pose.viseme_to)
                    if incoming is not None:
                        faded = incoming.copy()
                        faded.putalpha(faded.split()[3].point(
                            lambda p, b=pose.viseme_blend: int(p * b)))
                        mw_inc, mh_inc = faded.size
                        canvas.alpha_composite(faded, dest=(int(mcx - mw_inc / 2), int(mcy - mh_inc * 0.42)))

        # 3 · Whole-body squash & stretch (volume-preserving)
        if abs(pose.squash) > 0.01:
            sy = 1.0 + pose.squash
            sx = 1.0 - pose.squash * 0.6
            new_w = max(2, int(self.width * sx))
            new_h = max(2, int(self.height * sy))
            squashed = canvas.resize((new_w, new_h), Image.Resampling.BILINEAR)
            canvas = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
            canvas.alpha_composite(
                squashed.crop((0, max(0, new_h - self.height), new_w, new_h)),
                dest=(max(0, (self.width - new_w) // 2), max(0, self.height - new_h)))

        return canvas
