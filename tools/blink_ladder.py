"""Render a blink ladder on the real head plate and crop to the eyes.

Numbers prove coverage; only pixels prove it LOOKS like a blink. This
composites the actual plate through HeadAssembly at a series of closures
and lays the eye region out as one strip per character, at 3x, so the
lid's leading edge, the crease and the artwork's own lash are all
inspectable.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.eye_model import EyeState                       # noqa: E402
from engine.head_assembly import FaceChannels, HeadAssembly  # noqa: E402
from engine.rig import Rig                                   # noqa: E402

STEPS = (0.0, 0.2, 0.4, 0.6, 0.75, 0.9, 1.0)
ZOOM = 3
OUT = "/tmp/agent-browser"


def ladder(char: str, gaze=(0.0, 0.0)) -> str:
    rig = Rig.load(char)
    asm = HeadAssembly(rig, scale=1.0, fps=60, seed=char)

    # Crop box: both apertures plus a brow/cheek margin, so a stray fill
    # outside the eye would be visible rather than cropped away.
    xs, ys = [], []
    for rast in (asm.eyes.left, asm.eyes.right):
        xs += [rast._x0, rast._x1]
        ys += [rast._y0, rast._y1]
    pad = 26
    box = (max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad * 2),
           int(max(xs)) + pad, int(max(ys)) + pad)
    cw, chh = box[2] - box[0], box[3] - box[1]

    strip = Image.new("RGB", (cw * len(STEPS) * ZOOM // 1, chh * ZOOM),
                      (24, 24, 28))
    for i, c in enumerate(STEPS):
        ch = FaceChannels(eyes=EyeState(blink_l=c, blink_r=c,
                                        eye_dx=gaze[0], eye_dy=gaze[1]))
        plate = asm.compose_plate(ch)
        crop = plate.convert("RGB").crop(box).resize(
            (cw * ZOOM, chh * ZOOM), Image.NEAREST)
        strip.paste(crop, (i * cw * ZOOM, 0))
        d = ImageDraw.Draw(strip)
        d.text((i * cw * ZOOM + 6, 6), f"{c:.2f}", fill=(255, 240, 120))

    os.makedirs(OUT, exist_ok=True)
    tag = "" if gaze == (0.0, 0.0) else f"_gaze{gaze[0]:+.1f}{gaze[1]:+.1f}"
    path = f"{OUT}/blink_{char}{tag}.png"
    strip.save(path)
    print(f"{char}{tag}: {path}  ({strip.width}x{strip.height})")
    return path


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["chintu", "gudiya"]):
        ladder(name)
        ladder(name, gaze=(0.8, 0.0))
        ladder(name, gaze=(0.0, 0.8))
