"""
JEEVidya V5 — Rig Builder (Tier 1)
══════════════════════════════════
One-time, per-character puppet construction from a single body.png:

  1. MediaPipe FaceLandmarker (Tasks API) finds 478 face landmarks —
     verified to work on the show's cartoon faces. If detection fails,
     a silhouette heuristic takes over (fix-up later in /studio).
  2. The alpha silhouette locates the NECK (narrowest row below the chin)
     and the HIPS (bottom centroid) → the inferred 2-bone skeleton.
  3. body.png is soft-sliced into HEAD and TORSO layers with a feathered
     overlap band, so the seam stays invisible when the head rotates.
  4. Mouth/eye/brow boxes + skin & lip colors are extracted, and a full
     set of VISEME mouth sprites (MBP/E/AI/O/FV) plus eyelid sprites are
     procedurally baked in the character's own colors.
     Zero new art assets — ever.

Run:  python3 jvmake.py rig            (all characters)
      python3 jvmake.py rig gudiya --force
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from config import settings
from engine.rig import Rig, Layer, VISEME_NAMES, rig_dir, rig_path

FACE_MODEL = os.path.join(settings.ASSETS_DIR, "models", "face_landmarker.task")
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                  "face_landmarker/face_landmarker/float16/1/face_landmarker.task")

# MediaPipe FaceMesh landmark indices
_LM = {
    "chin": 152, "forehead": 10,
    "cheek_r": 50, "cheek_l": 280,
    "mouth": [61, 291, 0, 17, 13, 14, 78, 308, 87, 317],
    "eye_r": [33, 133, 159, 145, 160, 144, 158, 153],       # subject's right
    "eye_l": [362, 263, 386, 374, 385, 380, 387, 373],      # subject's left
    "brow_r": [70, 63, 105, 66, 107, 46, 53, 52, 65, 55],
    "brow_l": [336, 296, 334, 293, 300, 285, 295, 282, 283, 276],
}


# ═══════════════════════════════════════════
# DETECTION
# ═══════════════════════════════════════════

def _ensure_face_model() -> Optional[str]:
    if os.path.exists(FACE_MODEL):
        return FACE_MODEL
    try:
        import urllib.request
        os.makedirs(os.path.dirname(FACE_MODEL), exist_ok=True)
        print(f"  [Rig] Downloading face landmark model → {FACE_MODEL}")
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL)
        return FACE_MODEL
    except Exception as e:
        print(f"  [Rig] Model download failed ({e}) — heuristic mode")
        return None


def _detect_landmarks(img: Image.Image) -> Optional[List[Tuple[float, float]]]:
    """Run MediaPipe FaceLandmarker; return px-space landmarks or None."""
    model = _ensure_face_model()
    if not model:
        return None
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as tp
        from mediapipe.tasks.python import vision
    except ImportError:
        print("  [Rig] mediapipe not installed — heuristic mode "
              "(pip install mediapipe)")
        return None

    rgb = Image.new("RGB", img.size, (255, 255, 255))
    rgb.paste(img, mask=img.split()[3])
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(rgb))

    opts = vision.FaceLandmarkerOptions(
        base_options=tp.BaseOptions(model_asset_path=model),
        running_mode=vision.RunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=0.2, min_face_presence_confidence=0.2)
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        res = lm.detect(mp_img)
    if not res.face_landmarks:
        return None
    w, h = img.size
    return [(p.x * w, p.y * h) for p in res.face_landmarks[0]]


def _silhouette(alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-row silhouette width and centroid-x from the alpha channel."""
    mask = alpha > 40
    widths = mask.sum(axis=1).astype(np.float32)
    xs = np.arange(mask.shape[1], dtype=np.float32)
    with np.errstate(invalid="ignore"):
        cx = (mask * xs).sum(axis=1) / np.maximum(widths, 1)
    return widths, cx


def _find_neck(widths: np.ndarray, cx: np.ndarray, chin_y: int, h: int,
               face_h: Optional[float] = None) -> Tuple[float, float]:
    """
    Neck = narrowest smoothed silhouette row just below the chin.
    When the face height is known (landmarks), the search is bounded to
    0.35×face_h below the chin — long hair or a chibi waistline further
    down must never win. If the minimum lands on the search boundary
    (profile still shrinking → not a true neck), fall back to the
    anthropometric default of chin + 0.12×face_h.
    """
    reach = int(0.35 * face_h) if face_h else int(0.20 * h)
    default_y = chin_y + (int(0.12 * face_h) if face_h else int(0.04 * h))
    lo = min(h - 2, chin_y + max(2, int(0.02 * (face_h or h * 0.2))))
    hi = min(h - 1, chin_y + reach)
    if hi <= lo:
        return float(cx[min(h - 1, default_y)]), float(default_y)
    seg = widths[lo:hi].astype(np.float32).copy()
    # Smooth (moving average) so hair wisps don't create false minima
    k = max(3, int(0.01 * h)) | 1
    kernel = np.ones(k, dtype=np.float32) / k
    seg = np.convolve(seg, kernel, mode="same")
    idx = int(np.argmin(seg))
    if idx >= len(seg) - k:      # boundary hit → still narrowing, not a neck
        neck_y = min(h - 1, default_y)
    else:
        neck_y = lo + idx
    return float(cx[neck_y]), float(neck_y)


def _box_of(points: List[Tuple[float, float]], pad: float,
            size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    w, h = size
    px = (max(xs) - min(xs)) * pad
    py = (max(ys) - min(ys)) * pad
    return (int(max(0, min(xs) - px)), int(max(0, min(ys) - py)),
            int(min(w - 1, max(xs) + px)), int(min(h - 1, max(ys) + py)))


def _sample_color(arr: np.ndarray, x: float, y: float,
                  r: int = 4) -> Tuple[int, int, int]:
    h, w = arr.shape[:2]
    x0, x1 = max(0, int(x) - r), min(w, int(x) + r + 1)
    y0, y1 = max(0, int(y) - r), min(h, int(y) + r + 1)
    patch = arr[y0:y1, x0:x1]
    a = patch[..., 3:4].astype(np.float32) / 255.0
    if a.sum() < 1:
        return (224, 172, 138)
    rgb = (patch[..., :3].astype(np.float32) * a).sum(axis=(0, 1)) / a.sum()
    return tuple(int(v) for v in rgb)


def _lip_color(arr: np.ndarray, box: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
    """Mean of the darkest third of the mouth box → the lip line color."""
    x0, y0, x1, y1 = box
    patch = arr[y0:y1, x0:x1]
    if patch.size == 0:
        return (150, 60, 60)
    rgb = patch[..., :3].astype(np.float32)
    lum = rgb.mean(axis=2)
    thresh = np.percentile(lum, 33)
    dark = rgb[lum <= thresh]
    if len(dark) == 0:
        return (150, 60, 60)
    c = dark.mean(axis=0)
    return tuple(int(v) for v in c)


# ═══════════════════════════════════════════
# GEOMETRY (mediapipe or heuristic)
# ═══════════════════════════════════════════

def _geometry_from_landmarks(img: Image.Image, arr: np.ndarray,
                             lms: List[Tuple[float, float]]) -> dict:
    w, h = img.size
    widths, cx = _silhouette(arr[..., 3])
    chin = lms[_LM["chin"]]
    forehead = lms[_LM["forehead"]]
    face_h = max(8.0, chin[1] - forehead[1])
    neck_x, neck_y = _find_neck(widths, cx, int(chin[1]), h, face_h=face_h)

    rows = np.nonzero(widths > 0)[0]
    bottom = int(rows[-1]) if len(rows) else h - 1
    hips = (float(cx[max(0, bottom - int(0.02 * h)):bottom + 1].mean()),
            float(bottom))

    mouth = _box_of([lms[i] for i in _LM["mouth"]], 0.45, (w, h))
    eye_l = _box_of([lms[i] for i in _LM["eye_l"]], 0.55, (w, h))
    eye_r = _box_of([lms[i] for i in _LM["eye_r"]], 0.55, (w, h))
    brow_l = _box_of([lms[i] for i in _LM["brow_l"]], 0.35, (w, h))
    brow_r = _box_of([lms[i] for i in _LM["brow_r"]], 0.35, (w, h))

    skin = _sample_color(arr, *lms[_LM["cheek_r"]])
    top_rows = np.nonzero(widths > 0)[0]
    head_top_y = float(top_rows[0]) if len(top_rows) else 0.0

    return {
        "generated_by": "mediapipe",
        "joints": {
            "hips": hips,
            "neck": (neck_x, neck_y),
            "head_center": (float((eye_l[0] + eye_r[2]) / 2),
                            float((eye_l[1] + mouth[1]) / 2)),
            "head_top": (forehead[0], head_top_y),
        },
        "face": {
            "mouth": mouth, "eye_l": eye_l, "eye_r": eye_r,
            "brow_l": brow_l, "brow_r": brow_r,
            "skin": skin, "lip": _lip_color(arr, mouth),
        },
        "hair_line_y": float(min(brow_l[1], brow_r[1]) - 0.04 * h),
    }


def _geometry_heuristic(img: Image.Image, arr: np.ndarray) -> dict:
    """No face found: proportion-based guess. Nudge joints in /studio."""
    w, h = img.size
    widths, cx = _silhouette(arr[..., 3])
    rows = np.nonzero(widths > 0)[0]
    top = int(rows[0]) if len(rows) else 0
    bottom = int(rows[-1]) if len(rows) else h - 1
    span = bottom - top

    chin_y = top + int(span * 0.28)
    neck_x, neck_y = _find_neck(widths, cx, chin_y, h)
    head_h = neck_y - top
    face_cx = float(cx[top + int(head_h * 0.55)])

    def box(cx_, cy_, bw, bh):
        return (int(cx_ - bw / 2), int(cy_ - bh / 2),
                int(cx_ + bw / 2), int(cy_ + bh / 2))

    eye_y = top + head_h * 0.52
    mouth_y = top + head_h * 0.80
    eye_dx = head_h * 0.18
    skin = _sample_color(arr, face_cx, eye_y + head_h * 0.12)
    mouth = box(face_cx, mouth_y, head_h * 0.30, head_h * 0.14)

    return {
        "generated_by": "heuristic",
        "joints": {
            "hips": (float(cx[bottom - 2]), float(bottom)),
            "neck": (neck_x, neck_y),
            "head_center": (face_cx, float(top + head_h * 0.55)),
            "head_top": (face_cx, float(top)),
        },
        "face": {
            "mouth": mouth,
            "eye_l": box(face_cx + eye_dx, eye_y, head_h * 0.16, head_h * 0.10),
            "eye_r": box(face_cx - eye_dx, eye_y, head_h * 0.16, head_h * 0.10),
            "brow_l": box(face_cx + eye_dx, eye_y - head_h * 0.12,
                          head_h * 0.18, head_h * 0.05),
            "brow_r": box(face_cx - eye_dx, eye_y - head_h * 0.12,
                          head_h * 0.18, head_h * 0.05),
            "skin": skin, "lip": _lip_color(arr, mouth),
        },
        "hair_line_y": float(top + head_h * 0.35),
    }


# ═══════════════════════════════════════════
# LAYER SLICING (feathered neck seam)
# ═══════════════════════════════════════════

def _slice_layers(rig: Rig, img: Image.Image) -> None:
    """
    Soft-split body.png at the neck:
      head  — full alpha above the neck, fading to 0 by neck + 2f
      torso — full alpha below the neck, fading in from neck − 2f
    The 4f overlap band (head drawn on top) keeps the joint seamless
    under rotation.
    """
    w, h = img.size
    arr = np.asarray(img).copy()
    alpha = arr[..., 3].astype(np.float32)
    neck_y = rig.joint("neck")[1]
    f = max(4.0, rig.params.get("feather_px", 0.02 * h))
    ys = np.arange(h, dtype=np.float32)[:, None]

    head_ramp = np.clip(((neck_y + 2 * f) - ys) / (2 * f), 0, 1)
    torso_ramp = np.clip((ys - (neck_y - 2 * f)) / (2 * f), 0, 1)

    d = rig_dir(rig.character)
    os.makedirs(d, exist_ok=True)

    for name, ramp in (("head", head_ramp), ("torso", torso_ramp)):
        layer_arr = arr.copy()
        layer_arr[..., 3] = (alpha * ramp).astype(np.uint8)
        mask = layer_arr[..., 3] > 0
        if not mask.any():
            continue
        ys_nz, xs_nz = np.nonzero(mask)
        x0, x1 = int(xs_nz.min()), int(xs_nz.max()) + 1
        y0, y1 = int(ys_nz.min()), int(ys_nz.max()) + 1
        crop = Image.fromarray(layer_arr[y0:y1, x0:x1], "RGBA")
        fname = f"{name}.png"
        crop.save(os.path.join(d, fname))
        rig.layers[name] = Layer(name=name, file=fname, offset=(x0, y0))


# ═══════════════════════════════════════════
# VISEME + EYELID SPRITE BAKING
# ═══════════════════════════════════════════

def _feathered_backing(size: Tuple[int, int], color: Tuple[int, int, int],
                       feather: int) -> Image.Image:
    """Skin-colored ellipse with a soft alpha edge — hides the source mouth."""
    w, h = size
    backing = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(backing)
    draw.ellipse((feather // 2, feather // 2, w - feather // 2, h - feather // 2),
                 fill=color + (255,))
    a = backing.split()[3].filter(ImageFilter.GaussianBlur(feather / 2))
    backing.putalpha(a)
    return backing


def _art_viseme_dir(character: str) -> str:
    return os.path.join(settings.CHARACTERS_DIR, character, "visemes_src")


def _bake_visemes_from_art(rig: Rig, img: Image.Image) -> bool:
    """
    PERFECT lip sync path: bake viseme sprites from REAL hand-made art.

    Looks for assets/characters/<name>/visemes_src/<VISEME>.png — one
    full-body render per mouth shape (identical framing, only the mouth
    differs). For each one we locate the mouth (MediaPipe per image when
    available, else the rig's own mouth box scaled to the source frame),
    crop a padded patch around it, feather the edges with an elliptical
    alpha falloff, and save it as the viseme sprite.

    Returns True when at least OPEN_A + one closed shape were baked from
    art (the minimum for convincing speech); False → procedural fallback.
    """
    src_dir = _art_viseme_dir(rig.character)
    if not os.path.isdir(src_dir):
        return False

    x0, y0, x1, y1 = rig.box("mouth")
    mw, mh = max(8, x1 - x0), max(6, y1 - y0)
    # Sprite canvas: same conventions as the procedural path so the
    # BoneEngine composite (centered-x, 42% above mouth center) is exact.
    cw = int(mw * 1.7)
    ch = int(max(mh * 2.2, mw * 1.1))
    feather = max(4, cw // 8)
    rig_w, rig_h = rig.size

    def _mouth_geo_of(src: Image.Image,
                      lms: Optional[List[Tuple[float, float]]]
                      ) -> Tuple[float, float, float, float]:
        """Per-image mouth center + extent (w, h): landmarks if possible,
        else the rig's mouth box mapped through the frame-size ratio."""
        if lms:
            pts = [lms[i] for i in _LM["mouth"]]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            return (sum(xs) / len(xs), sum(ys) / len(ys),
                    max(xs) - min(xs), max(ys) - min(ys))
        sx = src.width / max(1, rig_w)
        sy = src.height / max(1, rig_h)
        return ((x0 + x1) / 2 * sx, (y0 + y1) / 2 * sy,
                (x1 - x0) * sx, (y1 - y0) * sy)

    # Outer lip contour ring (MediaPipe FaceMesh) — traces the actual
    # lip boundary so the occluded-bake mask hugs the real mouth.
    OUTER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                  291, 409, 270, 269, 267, 0, 37, 39, 40, 185]

    def _ellipse_mask(w: int, h: int, rx: float, ry: float,
                      blur: float) -> Image.Image:
        """Feathered ellipse centered on the mouth anchor (w/2, 0.42h)
        per the BoneEngine contract."""
        cx, cy = w / 2, h * 0.42
        m = Image.new("L", (w, h), 0)
        ImageDraw.Draw(m).ellipse((cx - rx, cy - ry, cx + rx, cy + ry),
                                  fill=255)
        return m.filter(ImageFilter.GaussianBlur(blur))

    # The exact region of the BASE body that the sprite is composited
    # onto — the reference for "what belongs around this mouth".
    base_left = int((x0 + x1) / 2 - cw / 2)
    base_top = int((y0 + y1) / 2 - ch * 0.42)
    base_patch = img.crop((base_left, base_top,
                           base_left + cw, base_top + ch))

    def _ring_clutter(patch: Image.Image,
                      cover_rx: float, cover_ry: float,
                      art_rx: float, art_ry: float) -> float:
        """Fraction of pixels in the ring between the tight mouth
        ellipse and the cover ellipse that differ STRONGLY from the
        base body at the same spot. Aligned facial features (glasses,
        chin shadow, nose) match the base and score ~0; a hand or a
        shifted collar that isn't there in the neutral pose scores
        high — meaning the wide window would flash it during speech."""
        w, h = patch.size
        cx, cy = w / 2, h * 0.42
        cover = Image.new("L", (w, h), 0)
        ImageDraw.Draw(cover).ellipse(
            (cx - cover_rx, cy - cover_ry, cx + cover_rx, cy + cover_ry),
            fill=255)
        art = Image.new("L", (w, h), 0)
        ImageDraw.Draw(art).ellipse(
            (cx - art_rx, cy - art_ry, cx + art_rx, cy + art_ry), fill=255)
        ring = (np.asarray(cover) > 128) & (np.asarray(art) <= 128)
        if ring.sum() < 16:
            return 0.0
        soft = ImageFilter.GaussianBlur(4)
        pa = np.asarray(patch.filter(soft), dtype=np.float32)
        ba = np.asarray(base_patch.filter(soft), dtype=np.float32)
        both = ring & (pa[..., 3] > 128) & (ba[..., 3] > 128)
        # Silhouette mismatch (sprite opaque where base is transparent,
        # or vice versa) is always a visible flash.
        alpha_flash = ring & (np.abs(pa[..., 3] - ba[..., 3]) > 128)
        diff = np.abs(pa[..., :3] - ba[..., :3]).sum(axis=2)
        bad = int(alpha_flash.sum()) + int((diff[both] > 140).sum())
        return bad / float(ring.sum())

    d = rig_dir(rig.character)
    os.makedirs(d, exist_ok=True)
    baked: Dict[str, str] = {}

    for name in VISEME_NAMES:
        path = os.path.join(src_dir, f"{name}.png")
        if not os.path.exists(path):
            continue
        try:
            src = Image.open(path).convert("RGBA")
        except Exception as e:
            print(f"  [Rig] {rig.character}: bad viseme art {path}: {e}")
            continue

        src_lms = _detect_landmarks(src)
        mcx, mcy, mvw, mvh = _mouth_geo_of(src, src_lms)
        # Source-space crop size (rescaled if frames differ)
        s = src.width / max(1, rig_w)
        scw, sch = max(8, int(cw * s)), max(8, int(ch * s))
        # Mouth center sits at 42% of sprite height (BoneEngine contract)
        left = int(mcx - scw / 2)
        top = int(mcy - sch * 0.42)
        patch = src.crop((left, top, left + scw, top + sch))
        if patch.size != (cw, ch):
            patch = patch.resize((cw, ch), Image.Resampling.LANCZOS)

        # ── Two ellipses, two jobs ──────────────────────────────────
        # COVER — big enough to fully hide the base pose's mouth that
        #         sits underneath the sprite at composite time.
        # ART   — hugs THIS frame's actual detected mouth, so nearby
        #         hands, hoodie collars and hair can never flash inside
        #         the sprite (mouth extent scaled into sprite space).
        s_mvw, s_mvh = mvw / max(s, 1e-6), mvh / max(s, 1e-6)
        cover_rx = min(cw / 2 - 1, max(cw * 0.30, s_mvw * 0.85))
        cover_ry = min(ch * 0.42 - 1, ch * (1 - 0.42) - 1,
                       max(ch * 0.24, s_mvh * 1.05))
        art_rx = min(cw / 2 - 1, max(s_mvw * 0.65, cw * 0.12))
        art_ry = min(ch * 0.42 - 1, ch * (1 - 0.42) - 1,
                     max(s_mvh * 0.65, ch * 0.10))

        # Facial tone near the mouth of THIS frame (just above the lips),
        # used both for clutter scoring and — if needed — the backing.
        tone = _sample_color(np.asarray(src), mcx,
                             max(0.0, mcy - mvh * 1.1), r=6)
        clutter = _ring_clutter(patch, cover_rx, cover_ry, art_rx, art_ry)
        occluded = (clutter > 0.10
                    and (art_rx < cover_rx - 2 or art_ry < cover_ry - 2))
        blur = max(2, feather // 2)

        if not occluded:
            # Clean surroundings: keep the full patch (real cheeks and
            # chin shading) inside one wide feathered window, exactly
            # as large as needed to cover the base mouth.
            mask = _ellipse_mask(cw, ch, max(cover_rx, art_rx),
                                 max(cover_ry, art_ry), blur)
            a = patch.split()[3]
            blended = (np.asarray(a, dtype=np.float32)
                       * np.asarray(mask, dtype=np.float32) / 255.0
                       ).astype(np.uint8)
            patch.putalpha(Image.fromarray(blended))
        else:
            # Occluders (hand/collar/hair) in the ring around the mouth:
            # keep only a tight ellipse of real mouth art, and build the
            # backing from the BASE body's own pixels — pixel-identical
            # to what sits underneath at composite time, so the seam is
            # invisible by construction. Only the base's own mouth line
            # gets a small feathered skin dab (tone sampled from the
            # base frame just above its mouth) so it can't peek around
            # the new tight mouth art.
            print(f"  [Rig] {rig.character}: {name} occluder detected "
                  f"(clutter {clutter:.2f}) — tight-mouth hybrid bake")
            base_tone = _sample_color(np.asarray(img), (x0 + x1) / 2,
                                      max(0.0, y0 - mh * 0.6), r=6)
            # Tone-match the source frame's skin to the base frame's, so
            # the kept mouth art can't read as a lighter/darker panel.
            # `tone` = skin just above the source mouth; per-channel gain
            # toward `base_tone`, gently clamped.
            pa = np.asarray(patch, dtype=np.float32)
            gain = np.array([np.clip(bt / max(t, 1.0), 0.72, 1.38)
                             for bt, t in zip(base_tone, tone)],
                            dtype=np.float32)
            pa[..., :3] = np.clip(pa[..., :3] * gain, 0, 255)
            patch = Image.fromarray(pa.astype(np.uint8), "RGBA")

            # Art window: the actual OUTER-LIP polygon from this frame's
            # landmarks (grown slightly, small feather) — keeps ONLY the
            # lips and mouth interior, so surrounding source skin,
            # chin shadows, collars, and hands can never ghost over the
            # base face. Ellipse fallback when landmarks are missing.
            if src_lms:
                pts = [src_lms[i] for i in OUTER_LIPS]
                pcx = sum(p[0] for p in pts) / len(pts)
                pcy = sum(p[1] for p in pts) / len(pts)
                grow = 1.22
                poly = [(((pcx + (px - pcx) * grow) - left) * cw / scw,
                         ((pcy + (py - pcy) * grow) - top) * ch / sch)
                        for px, py in pts]
                m = Image.new("L", (cw, ch), 0)
                ImageDraw.Draw(m).polygon(poly, fill=255)
                art_mask = m.filter(
                    ImageFilter.GaussianBlur(max(2.0, blur * 0.75)))
            else:
                art_mask = _ellipse_mask(cw, ch, art_rx * 0.92,
                                         art_ry * 0.88, blur * 2.0)
            a = patch.split()[3]
            blended = (np.asarray(a, dtype=np.float32)
                       * np.asarray(art_mask, dtype=np.float32) / 255.0
                       ).astype(np.uint8)
            patch.putalpha(Image.fromarray(blended))

            backing = base_patch.copy()
            bcx, bcy = cw / 2, ch * 0.42
            dab = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
            drx, dry = mw * 0.62, mh * 0.80
            ImageDraw.Draw(dab).ellipse(
                (bcx - drx, bcy - dry, bcx + drx, bcy + dry),
                fill=base_tone + (255,))
            dab.putalpha(dab.split()[3].filter(
                ImageFilter.GaussianBlur(max(2, blur))))
            backing.alpha_composite(dab)
            # Feather the backing window edge (matches the base below,
            # but guards against sub-pixel drift under head rotation).
            cover_mask = _ellipse_mask(cw, ch, cover_rx, cover_ry, blur)
            back_a = (np.asarray(backing.split()[3], dtype=np.float32)
                      * np.asarray(cover_mask, dtype=np.float32) / 255.0
                      ).astype(np.uint8)
            backing.putalpha(Image.fromarray(back_a))
            backing.alpha_composite(patch)
            patch = backing

        fname = f"mouth_{name.lower()}.png"
        patch.save(os.path.join(d, fname))
        baked[name] = fname

    closed = {"REST", "BILABIAL"} & set(baked)
    if "OPEN_A" in baked and closed:
        rig.visemes = {**{k: v for k, v in rig.visemes.items()
                          if k.startswith("LID_")}, **baked}
        print(f"  [Rig] {rig.character}: {len(baked)} visemes baked "
              f"from REAL ART ({', '.join(sorted(baked))})")
        return True
    return False


def _bake_visemes(rig: Rig, img: Image.Image) -> None:
    """
    Procedurally draw the mouth shapes in the character's own lip/skin
    colors, sized to the detected mouth box. `REST` is the untouched
    original art (no overlay). Used only when no visemes_src/ art exists.
    """
    x0, y0, x1, y1 = rig.box("mouth")
    mw, mh = max(8, x1 - x0), max(6, y1 - y0)
    # Sprite canvas ~ 1.7x the mouth box so open shapes have room
    cw, ch = int(mw * 1.7), int(max(mh * 2.2, mw * 1.1))
    skin = rig.color("skin")
    lip = rig.color("lip")
    inner = tuple(max(0, int(c * 0.35)) for c in lip)
    teeth = (245, 240, 235)
    tongue = tuple(min(255, int(c * 1.35) + 25) for c in lip)
    feather = max(4, cw // 8)
    lw = max(3, cw // 16)                      # lip line width
    cx_, cy_ = cw / 2, ch / 2

    def canvas():
        return _feathered_backing((cw, ch), skin, feather), None

    def ellipse(draw, rx, ry, fill, outline=None, width=0, dy=0.0):
        draw.ellipse((cx_ - rx, cy_ - ry + dy, cx_ + rx, cy_ + ry + dy),
                     fill=fill, outline=outline, width=width)

    d = rig_dir(rig.character)
    sprites: Dict[str, Optional[Image.Image]] = {"REST": None}

    # MBP — pressed-closed lips: a slightly curved lip line
    im, _ = canvas()
    draw = ImageDraw.Draw(im)
    draw.line((cx_ - mw * 0.42, cy_, cx_ + mw * 0.42, cy_), fill=lip + (255,),
              width=lw, joint="curve")
    sprites["MBP"] = im

    # E — mid horizontal spread, teeth hint
    im, _ = canvas()
    draw = ImageDraw.Draw(im)
    ellipse(draw, mw * 0.46, mh * 0.42, inner + (255,), lip + (255,), lw)
    draw.rectangle((cx_ - mw * 0.32, cy_ - mh * 0.34,
                    cx_ + mw * 0.32, cy_ - mh * 0.08), fill=teeth + (255,))
    sprites["E"] = im

    # AI — wide open: dark cavity, teeth top, tongue bottom
    im, _ = canvas()
    draw = ImageDraw.Draw(im)
    ellipse(draw, mw * 0.50, mh * 0.95, inner + (255,), lip + (255,), lw)
    draw.rectangle((cx_ - mw * 0.34, cy_ - mh * 0.85,
                    cx_ + mw * 0.34, cy_ - mh * 0.45), fill=teeth + (255,))
    draw.ellipse((cx_ - mw * 0.30, cy_ + mh * 0.15,
                  cx_ + mw * 0.30, cy_ + mh * 0.95), fill=tongue + (255,))
    sprites["AI"] = im

    # O — round
    im, _ = canvas()
    draw = ImageDraw.Draw(im)
    r = mw * 0.30
    ellipse(draw, r, r * 1.15, inner + (255,), lip + (255,), lw)
    sprites["O"] = im

    # FV — lower lip tucked: thin opening + bright lower-lip band
    im, _ = canvas()
    draw = ImageDraw.Draw(im)
    ellipse(draw, mw * 0.40, mh * 0.26, inner + (255,), lip + (255,), lw)
    draw.rectangle((cx_ - mw * 0.30, cy_ - mh * 0.18,
                    cx_ + mw * 0.30, cy_ - mh * 0.02), fill=teeth + (255,))
    highlight = tuple(min(255, c + 40) for c in lip)
    draw.line((cx_ - mw * 0.36, cy_ + mh * 0.24, cx_ + mw * 0.36, cy_ + mh * 0.24),
              fill=highlight + (255,), width=lw)
    sprites["FV"] = im

    rig.visemes = {}
    for name, sprite in sprites.items():
        if sprite is None:
            continue
        fname = f"mouth_{name.lower()}.png"
        sprite.save(os.path.join(d, fname))
        rig.visemes[name] = fname


def _bake_eyelids(rig: Rig, img: Image.Image) -> None:
    """Eyelid sprites (blink): skin ellipse + lash line, one per eye.
    Baked separately so BOTH the art and procedural viseme paths get
    blinks."""
    skin = rig.color("skin")
    d = rig_dir(rig.character)
    os.makedirs(d, exist_ok=True)
    for eye in ("eye_l", "eye_r"):
        ex0, ey0, ex1, ey1 = rig.box(eye)
        ew, eh = max(6, ex1 - ex0), max(4, ey1 - ey0)
        lid = Image.new("RGBA", (int(ew * 1.4), int(eh * 1.6)), (0, 0, 0, 0))
        ld = ImageDraw.Draw(lid)
        ld.ellipse((0, 0, lid.width - 1, lid.height - 1), fill=skin + (255,))
        lash = tuple(max(0, int(c * 0.45)) for c in skin)
        ld.arc((2, lid.height * 0.15, lid.width - 3, lid.height * 1.1),
               start=15, end=165, fill=lash + (255,), width=max(2, eh // 5))
        a = lid.split()[3].filter(ImageFilter.GaussianBlur(max(2, ew // 12)))
        lid.putalpha(a)
        fname = f"lid_{eye}.png"
        lid.save(os.path.join(d, fname))
        rig.visemes[f"LID_{eye.upper()}"] = fname


# ═══════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════

def _load_body(character: str) -> Image.Image:
    char_dir = os.path.join(settings.CHARACTERS_DIR, character)
    for base in ("body", "neutral", "original"):
        for ext in (".png", ".jpg", ".jpeg"):
            p = os.path.join(char_dir, base + ext)
            if os.path.exists(p):
                return Image.open(p).convert("RGBA")
    raise FileNotFoundError(f"No source image for character '{character}'")


def rebake(rig: Rig) -> Rig:
    """Re-slice layers + re-bake sprites from the CURRENT rig geometry.
    Used by /studio after joints or face boxes are nudged."""
    img = _load_body(rig.character)
    rig.size = img.size
    _slice_layers(rig, img)
    if not _bake_visemes_from_art(rig, img):
        _bake_visemes(rig, img)
    _bake_eyelids(rig, img)
    rig.save()
    return rig


def build_rig(character: str, force: bool = False) -> Rig:
    """Full auto-rig: detect → skeleton → slice → bake → save."""
    if not force and os.path.exists(rig_path(character)):
        rig = Rig.load(character)
        if rig.is_complete():
            print(f"  [Rig] {character}: rig exists (use --force to rebuild)")
            return rig

    img = _load_body(character)
    arr = np.asarray(img)
    print(f"  [Rig] {character}: analyzing {img.size[0]}x{img.size[1]} source...")

    lms = _detect_landmarks(img)
    geo = (_geometry_from_landmarks(img, arr, lms) if lms
           else _geometry_heuristic(img, arr))

    rig = Rig(character=character, size=img.size,
              generated_by=geo["generated_by"])
    rig.joints = geo["joints"]
    rig.face = geo["face"]
    rig.params = {"feather_px": 0.02 * img.size[1],
                  "hair_line_y": geo["hair_line_y"]}

    _slice_layers(rig, img)
    if not _bake_visemes_from_art(rig, img):
        _bake_visemes(rig, img)
    _bake_eyelids(rig, img)
    rig.save()

    mode = "face landmarks" if geo["generated_by"] == "mediapipe" else \
           "HEURISTIC (nudge joints in /studio!)"
    print(f"  [Rig] {character}: ✓ rig built via {mode}")
    print(f"        neck=({rig.joint('neck')[0]:.0f},{rig.joint('neck')[1]:.0f}) "
          f"hips=({rig.joint('hips')[0]:.0f},{rig.joint('hips')[1]:.0f}) "
          f"visemes={len([v for v in rig.visemes if not v.startswith('LID')])}")
    return rig


def build_all(force: bool = False) -> List[Rig]:
    rigs = []
    for name in sorted(os.listdir(settings.CHARACTERS_DIR)):
        char_dir = os.path.join(settings.CHARACTERS_DIR, name)
        if not os.path.isdir(char_dir):
            continue
        try:
            rigs.append(build_rig(name, force=force))
        except FileNotFoundError as e:
            print(f"  [Rig] {name}: skipped ({e})")
    return rigs


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="JEEVidya puppet rig builder")
    ap.add_argument("character", nargs="?", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.character:
        build_rig(args.character, force=args.force)
    else:
        build_all(force=args.force)
