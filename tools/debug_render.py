"""One-off diagnostic: render key frames per character with the analytic
predictions and QC color-mask detections overlaid, so the true source of
registration error (renderer vs detector) is visible.

Usage: .venv/bin/python -m tools.debug_render
Writes: /tmp/agent-browser/debug_<char>_<tag>.png
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw

from engine.bone_engine import BoneEngine, PuppetPose
from engine.rig import Rig
from tools.face_qc import color_mask, mask_centroid
from tools.verify_face import _lip_mask

OUT = "/tmp/agent-browser"
os.makedirs(OUT, exist_ok=True)


def cross(draw: ImageDraw.ImageDraw, xy, color, r=14, w=4):
    x, y = xy
    draw.line([(x - r, y), (x + r, y)], fill=color, width=w)
    draw.line([(x, y - r), (x, y + r)], fill=color, width=w)


def main() -> None:
    for char in ("chintu", "gudiya"):
        rig = Rig.load(char)
        eng = BoneEngine(rig)
        palette = rig.head.palette or {}
        lip_rgb = tuple(palette.get("lip", (170, 80, 80)))
        shadow_rgb = tuple(palette["lip_shadow"]) if "lip_shadow" in palette else None
        iris_rgb = tuple(palette.get("iris", (92, 58, 38)))
        print(f"[v0] {char}: lip={lip_rgb} shadow={shadow_rgb} iris={iris_rgb}")

        cases = [
            ("neutral", PuppetPose()),
            ("open_a", PuppetPose(viseme="OPEN_A", viseme_to="OPEN_A",
                                  mouth_open=1.0)),
            ("blink1", PuppetPose(blink=1.0)),
        ]
        for tag, pose in cases:
            frame = eng.render(pose)
            pred = eng.predict(pose)
            lip_m = _lip_mask(frame, lip_rgb, shadow_rgb)
            iris_m = color_mask(frame, iris_rgb, tol=48)
            det_mouth = mask_centroid(lip_m)
            print(f"[v0] {char}/{tag}: pred_mouth={pred['mouth']} "
                  f"det_mouth={det_mouth} lip_px={int(lip_m.sum())} "
                  f"iris_px={int(iris_m.sum())}")

            vis = frame.convert("RGB")
            arr = np.asarray(vis).copy()
            arr[lip_m] = (0, 255, 0)       # lip-mask pixels → green
            arr[iris_m] = (0, 128, 255)    # iris-mask pixels → blue
            vis = Image.fromarray(arr)
            d = ImageDraw.Draw(vis)
            cross(d, pred["mouth"], (255, 0, 0))          # predicted mouth
            cross(d, pred["iris_l"], (255, 0, 255))
            cross(d, pred["iris_r"], (255, 0, 255))
            if det_mouth:
                cross(d, det_mouth, (255, 255, 0))        # detected mouth
            vis.save(os.path.join(OUT, f"debug_{char}_{tag}.png"))
    print("[v0] wrote overlays to", OUT)


if __name__ == "__main__":
    main()
