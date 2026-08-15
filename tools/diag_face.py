"""Ad-hoc diagnostic: render frames, overlay PREDICTED vs DETECTED
feature positions, and dump the raw color masks the QC gates use.

Answers one question: is the RENDER wrong, or is the DETECTOR wrong?

    python tools/diag_face.py chintu gudiya
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

from engine.bone_engine import BoneEngine, PuppetPose
from engine.rig import Rig
from tools.face_qc import color_mask, connected_components, mask_centroid

OUT = "/tmp/diag"
TOL = 48


def cross(d: ImageDraw.ImageDraw, p, color, r=14, w=3, label=""):
    x, y = p
    d.line([(x - r, y), (x + r, y)], fill=color, width=w)
    d.line([(x, y - r), (x, y + r)], fill=color, width=w)
    if label:
        d.text((x + r + 3, y - 8), label, fill=color)


def tint(base: Image.Image, mask: np.ndarray, rgb) -> Image.Image:
    a = np.asarray(base.convert("RGB")).copy()
    a[mask] = rgb
    return Image.fromarray(a)


def run(name: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    rig = Rig.load(name)
    eng = BoneEngine(rig)
    pal = rig.head.palette or {}
    lip = tuple(pal.get("lip", (170, 80, 80)))
    iris = tuple(pal.get("iris", (92, 58, 38)))
    face_h = max(1.0, rig.head.face_height * eng.scale)
    tol_px = 0.006 * face_h
    print(f"\n===== {name} =====")
    print(f"  canvas={eng.width}x{eng.height} scale={eng.scale:.4f} "
          f"face_h={face_h:.1f} reg_tol={tol_px:.2f}px")
    print(f"  palette lip={lip} iris={iris}")

    for tag, pose in (("neutral", PuppetPose()),
                      ("openA", PuppetPose(viseme="OPEN_A",
                                           viseme_to="OPEN_A",
                                           mouth_open=1.0)),
                      ("blink1", PuppetPose(blink=1.0))):
        frame = eng.render(pose)
        pred = eng.predict(pose)
        lm = color_mask(frame, lip, tol=TOL)
        im = color_mask(frame, iris, tol=TOL)
        ldet = mask_centroid(lm)
        idet = mask_centroid(im)
        print(f"\n  [{tag}] frame={frame.size}")
        print(f"    lip mask px={int(lm.sum())} comps={connected_components(lm)}")
        print(f"    iris mask px={int(im.sum())}")
        print(f"    mouth  pred={tuple(round(v,1) for v in pred['mouth'])} "
              f"det={ldet and tuple(round(v,1) for v in ldet)} "
              f"err={ldet and round(math.dist(pred['mouth'], ldet),1)}")
        for e in ("iris_l", "iris_r"):
            print(f"    {e:6} pred={tuple(round(v,1) for v in pred[e])}")
        if idet:
            print(f"    iris   det(all)={tuple(round(v,1) for v in idet)}")

        vis = tint(tint(frame, lm, (0, 255, 0)), im, (0, 128, 255))
        d = ImageDraw.Draw(vis)
        cross(d, pred["mouth"], (255, 0, 0), label="pred mouth")
        if ldet:
            cross(d, ldet, (255, 255, 0), label="det mouth")
        for e in ("iris_l", "iris_r"):
            cross(d, pred[e], (255, 0, 255), r=9, label=e)
        vis.save(f"{OUT}/{name}_{tag}_mask.png")
        frame.convert("RGB").save(f"{OUT}/{name}_{tag}_raw.png")

    # where do lip-colored pixels actually live?
    frame = eng.render(PuppetPose(viseme="OPEN_A", viseme_to="OPEN_A",
                                 mouth_open=1.0))
    lm = color_mask(frame, lip, tol=TOL)
    ys, xs = np.nonzero(lm)
    if len(xs):
        print(f"\n  lip-mask bbox x[{xs.min()}..{xs.max()}] "
              f"y[{ys.min()}..{ys.max()}]  (canvas {frame.size})")
    im = color_mask(frame, iris, tol=TOL)
    ys, xs = np.nonzero(im)
    if len(xs):
        print(f"  iris-mask bbox x[{xs.min()}..{xs.max()}] "
              f"y[{ys.min()}..{ys.max()}]")


if __name__ == "__main__":
    names = sys.argv[1:] or ["chintu", "gudiya"]
    for n in names:
        run(n)
    print(f"\nartifacts → {OUT}")
