"""Diagnostic: what is inside the gate's "mouth" mask on a FULL FRAME?

`dev_diag_mouth3` proved the renderer innocent: in PATCH space the painted
lip body sits within ~1px of the analytic ring centroid on both
characters, every viseme. The 5–13px downward error only appears once the
same mouth is measured on the full frame, so the extra pixels come from
the frame — i.e. the detector's mask is picking up artwork.

This dumps the evidence: the mask's area/bbox/centroid against the
analytic contour's own bbox pushed through the same affine, plus PNG
overlays (mask in green, analytic outer contour in cyan, mask centroid
red, predicted ring centroid blue) so the contamination can be SEEN.

Usage:  python3 -m tools.dev_diag_mouth4 [chintu gudiya]
Writes: output/diag_mouth4/<character>_<viseme>.png
"""
from __future__ import annotations

import math
import os
import sys
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from config import settings
from engine.bone_engine import BoneEngine, PuppetPose
from engine.mouth_model import lip_contour
from engine.rig import Rig
from tools.dev_diag_mouth2 import ring_centroid
from tools.face_qc import mask_centroid
from tools.verify_face import (_baked_viseme_classes, _lip_mask, _mask_bbox,
                               _mouth_blob, _roi_from_variation)

OUT = os.path.join(settings.OUTPUT_DIR, "diag_mouth4")


def diag(character: str, visemes: Sequence[str] = ()) -> None:
    os.makedirs(OUT, exist_ok=True)
    rig = Rig.load(character)
    eng = BoneEngine(rig)
    asm = eng.assembly
    mr = asm.mouth
    face_h = max(1.0, rig.head.face_height * eng.scale)
    pal = rig.head.palette or {}
    lip = tuple(pal.get("lip", (170, 80, 80)))
    shadow = tuple(pal["lip_shadow"]) if "lip_shadow" in pal else None
    names = list(visemes) or _baked_viseme_classes(rig)

    frames = {v: eng.render(PuppetPose(viseme=v, viseme_to=v, mouth_open=1.0))
              for v in _baked_viseme_classes(rig)}
    rest = eng.render(PuppetPose(mouth_open=0.0))
    roi = _roi_from_variation(list(frames.values()) + [rest], face_h)
    rb = _mask_bbox(roi) if roi is not None else None
    print(f"\n═══ {character}  face_h={face_h:.1f}  lip={lip} shadow={shadow}")
    print(f"    ROI bbox={rb}  (dilation = {0.05 * face_h:.1f}px)")

    for v in names:
        pose = PuppetPose(viseme=v, viseme_to=v, mouth_open=1.0).clamped()
        ch, head, fp, tp, t = eng._channels(pose, None)
        aff = asm.affine(fp, tp, t, head)
        outer_n, inner_n = lip_contour(ch.mouth)
        cx, cy = mr.center
        outer = [(cx + x * mr.scale, cy + y * mr.scale) for x, y in outer_n]
        inner = [(cx + x * mr.scale, cy + y * mr.scale) for x, y in inner_n]
        canvas_outer = [aff.apply_feature_point(x, y) for x, y in outer]
        pred = aff.apply_feature_point(*ring_centroid(outer, inner))

        frame = frames[v]
        raw = _lip_mask(frame, lip, shadow, roi)
        blob = _mouth_blob(frame, lip, shadow, roi)
        det = mask_centroid(blob)
        bb = _mask_bbox(blob)
        xs = [p[0] for p in canvas_outer]
        ys = [p[1] for p in canvas_outer]
        abox = (min(xs), min(ys), max(xs), max(ys))
        print(f"    {v:14s} blob={int(blob.sum()):6d}px raw={int(raw.sum()):6d}px "
              f"bbox={bb} analytic_bbox=("
              f"{abox[0]:.0f},{abox[1]:.0f},{abox[2]:.0f},{abox[3]:.0f}) "
              f"err={math.dist(pred, det) if det else float('nan'):.2f}")

        # ── visual evidence ──
        pad = int(0.10 * face_h)
        x0 = int(min(abox[0], bb[0] if bb else abox[0])) - pad
        y0 = int(min(abox[1], bb[1] if bb else abox[1])) - pad
        x1 = int(max(abox[2], bb[2] if bb else abox[2])) + pad
        y1 = int(max(abox[3], bb[3] if bb else abox[3])) + pad
        crop = frame.convert("RGBA").crop((x0, y0, x1, y1))
        over = np.asarray(crop, dtype=np.int16).copy()
        sub = blob[y0:y1, x0:x1]
        if sub.shape == over.shape[:2]:
            over[..., 1] = np.where(sub, 255, over[..., 1])
        img = Image.fromarray(np.clip(over, 0, 255).astype(np.uint8), "RGBA")
        img = img.resize((img.width * 3, img.height * 3), Image.NEAREST)
        d = ImageDraw.Draw(img)
        d.polygon([((x - x0) * 3, (y - y0) * 3) for x, y in canvas_outer],
                  outline=(0, 220, 255, 255))
        if det:
            d.line([((det[0] - x0) * 3 - 9, (det[1] - y0) * 3),
                    ((det[0] - x0) * 3 + 9, (det[1] - y0) * 3)],
                   fill=(255, 0, 0, 255), width=2)
            d.line([((det[0] - x0) * 3, (det[1] - y0) * 3 - 9),
                    ((det[0] - x0) * 3, (det[1] - y0) * 3 + 9)],
                   fill=(255, 0, 0, 255), width=2)
        d.line([((pred[0] - x0) * 3 - 9, (pred[1] - y0) * 3),
                ((pred[0] - x0) * 3 + 9, (pred[1] - y0) * 3)],
               fill=(0, 80, 255, 255), width=2)
        d.line([((pred[0] - x0) * 3, (pred[1] - y0) * 3 - 9),
                ((pred[0] - x0) * 3, (pred[1] - y0) * 3 + 9)],
               fill=(0, 80, 255, 255), width=2)
        img.save(os.path.join(OUT, f"{character}_{v}.png"))
    print(f"    → {OUT}")


def main(argv: Sequence[str]) -> int:
    chars = [a for a in argv] or ["chintu", "gudiya"]
    for c in chars:
        diag(c, ["REST", "OPEN_A"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
