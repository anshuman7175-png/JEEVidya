"""Dev probe: what is the iris registration gate actually locking onto?

The gate reports ~150 px error on a render whose irises are visually in
the right place, so either the prediction or the detection is wrong. This
enumerates EVERY iris-coloured component the gate can see, with its area
and its distance from the prediction, so the answer is measured instead
of argued.

    python -m tools.dev_iris_probe [--out DIR]
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from engine.bone_engine import BoneEngine, PuppetPose
from engine.rig import Rig
from tools.face_qc import (color_mask, label_components, mask_centroid,
                           largest_component)
from tools.verify_face import _roi_from_variation, IRIS_TOL


def probe(character: str, out_dir: str) -> None:
    rig = Rig.load(character)
    engine = BoneEngine(rig)
    face_h = max(1.0, rig.head.face_height * engine.scale)
    palette = rig.head.palette or {}
    iris_rgb = tuple(palette.get("iris", (92, 58, 38)))

    neutral = PuppetPose()
    open_frame = engine.render(neutral)
    pred = engine.predict(neutral)
    closed = engine.render(PuppetPose(blink=1.0))
    eye_roi = _roi_from_variation([open_frame, closed], face_h)

    print(f"═══ {character}  face_h={face_h:.1f}px  iris_rgb={iris_rgb}  "
          f"tol={IRIS_TOL}  roi={0 if eye_roi is None else int(eye_roi.sum())}px")
    print(f"    predicted iris_l={pred['iris_l']}  iris_r={pred['iris_r']}")

    raw = color_mask(open_frame, iris_rgb, tol=IRIS_TOL)
    m = raw & eye_roi if eye_roi is not None else raw
    print(f"    iris-colour px: whole frame {int(raw.sum())}, "
          f"inside ROI {int(m.sum())}")

    mid_x = int((pred["iris_l"][0] + pred["iris_r"][0]) / 2.0)
    print(f"    midline x={mid_x}")

    ov = open_frame.convert("RGBA").copy()
    d = ImageDraw.Draw(ov)

    for eye in ("iris_l", "iris_r"):
        side = np.zeros_like(m)
        if eye == "iris_l":
            side[:, :mid_x] = m[:, :mid_x]
        else:
            side[:, mid_x:] = m[:, mid_x:]
        lab, n = label_components(side)
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        order = np.argsort(counts)[::-1]
        px, py = pred[eye]
        print(f"  ── {eye}: {n} component(s); predicted ({px:.0f},{py:.0f})")
        for rank, li in enumerate(order[:6]):
            if counts[li] == 0:
                break
            ys, xs = np.nonzero(lab == li)
            cx, cy = float(xs.mean()), float(ys.mean())
            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            tag = "  <-- gate picks this (largest)" if rank == 0 else ""
            print(f"      #{rank} area={counts[li]:6d}px  "
                  f"centroid=({cx:7.1f},{cy:7.1f})  "
                  f"bbox=({xs.min()},{ys.min()})-({xs.max()},{ys.max()})  "
                  f"dist={dist:7.1f}px{tag}")
            d.rectangle([int(xs.min()), int(ys.min()),
                         int(xs.max()), int(ys.max())],
                        outline=(0, 255, 0, 255) if rank == 0
                        else (255, 200, 0, 255))
        det = mask_centroid(largest_component(side))
        if det:
            d.line([det[0] - 9, det[1], det[0] + 9, det[1]],
                   fill=(0, 255, 0, 255), width=2)
            d.line([det[0], det[1] - 9, det[0], det[1] + 9],
                   fill=(0, 255, 0, 255), width=2)
        d.line([px - 9, py, px + 9, py], fill=(255, 0, 255, 255), width=2)
        d.line([px, py - 9, px, py + 9], fill=(255, 0, 255, 255), width=2)

    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{character}_iris_probe.png")
    ov.save(p)
    print(f"    → {p}  (magenta = predicted, green = gate detection)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("characters", nargs="*",
                    default=["chintu", "gudiya"])
    ap.add_argument("--out", default="/tmp/irisprobe")
    a = ap.parse_args()
    for c in a.characters:
        probe(c, a.out)


if __name__ == "__main__":
    main()
