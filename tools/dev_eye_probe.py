"""Dev-only visual probe: renders blink/gaze/viseme frames and writes
zoomed face crops so eye and mouth defects are visible to the eye.

Not part of the pipeline; run as:
    python -m tools.dev_eye_probe [character ...]
"""
from __future__ import annotations

import sys
from typing import List, Tuple

from PIL import Image

from engine.bone_engine import BoneEngine, PuppetPose
from engine.rig import Rig

OUT = "/tmp/probe"
ZOOM = 3


def _face_box(rig: Rig, frame: Image.Image) -> Tuple[int, int, int, int]:
    """Head box in FRAME space, padded, from the rig's head rect."""
    boxes = []
    for key in ("eye_l", "eye_r", "mouth", "brow_l", "brow_r"):
        try:
            boxes.append(rig.box(key))
        except Exception:
            pass
    if boxes:
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes)
        y1 = max(b[3] for b in boxes)
    else:
        x0, y0, x1, y1 = 0, 0, frame.width, frame.height // 2
    w, h = x1 - x0, y1 - y0
    px, py = int(w * 0.30), int(h * 0.45)
    return (max(0, x0 - px), max(0, y0 - py),
            min(frame.width, x1 + px), min(frame.height, y1 + py))


def _strip(tiles: List[Tuple[str, Image.Image]]) -> Image.Image:
    if not tiles:
        return Image.new("RGB", (8, 8), (0, 0, 0))
    tw = max(t.width for _, t in tiles)
    th = max(t.height for _, t in tiles)
    out = Image.new("RGB", (tw * len(tiles), th), (24, 24, 28))
    for i, (_, t) in enumerate(tiles):
        out.paste(t.convert("RGB"), (i * tw, 0))
    return out


def probe(name: str) -> None:
    import os
    os.makedirs(OUT, exist_ok=True)
    rig = Rig.load(name)
    eng = BoneEngine(rig)

    base = eng.render(PuppetPose().clamped())
    box = _face_box(rig, base)

    def crop(pose: PuppetPose) -> Image.Image:
        f = eng.render(pose.clamped()).convert("RGB").crop(box)
        return f.resize((f.width * ZOOM, f.height * ZOOM), Image.NEAREST)

    # Tight, heavily magnified crop of the eye pair only — a lid defect is
    # a few pixels on a 1000px plate and is invisible at face zoom.
    ebs = [rig.box("eye_l"), rig.box("eye_r")]
    ex0 = min(b[0] for b in ebs)
    ey0 = min(b[1] for b in ebs)
    ex1 = max(b[2] for b in ebs)
    ey1 = max(b[3] for b in ebs)
    ew, eh = ex1 - ex0, ey1 - ey0
    ebox = (max(0, ex0 - int(ew * 0.15)), max(0, ey0 - int(eh * 0.85)),
            min(base.width, ex1 + int(ew * 0.15)),
            min(base.height, ey1 + int(eh * 0.55)))

    def eyecrop(pose: PuppetPose, zoom: int = 6) -> Image.Image:
        f = eng.render(pose.clamped()).convert("RGB").crop(ebox)
        return f.resize((f.width * zoom, f.height * zoom), Image.NEAREST)

    _strip([(f"b{b}", eyecrop(PuppetPose(blink=b)))
            for b in (0.0, 0.35, 0.7, 1.0)]).save(f"{OUT}/{name}_eyeblink.png")
    _strip([(f"g{i}", eyecrop(PuppetPose(eye_dx=dx, eye_dy=dy)))
            for i, (dx, dy) in enumerate(
                [(-1.0, 0.0), (0.0, -1.0), (0.0, 1.0), (1.0, 0.0)])]) \
        .save(f"{OUT}/{name}_eyegaze.png")

    blinks = [0.0, 0.25, 0.5, 0.75, 1.0]
    _strip([(f"b{b}", crop(PuppetPose(blink=b))) for b in blinks]) \
        .save(f"{OUT}/{name}_blink.png")

    gaze = [(-1.0, 0.0), (0.0, -1.0), (0.0, 0.0), (0.0, 1.0), (1.0, 0.0)]
    _strip([(f"g{i}", crop(PuppetPose(eye_dx=dx, eye_dy=dy)))
            for i, (dx, dy) in enumerate(gaze)]).save(f"{OUT}/{name}_gaze.png")

    vis = sorted(rig.mouth_targets.keys()) or sorted(eng.viseme_sprites.keys())
    vis = list(vis)[:10]
    if vis:
        _strip([(v, crop(PuppetPose(viseme=v, mouth_open=1.0)))
                for v in vis]).save(f"{OUT}/{name}_visemes.png")

    crop(PuppetPose()).save(f"{OUT}/{name}_rest.png")
    print(f"[probe] {name}: box={box} visemes={vis}")


if __name__ == "__main__":
    names = sys.argv[1:] or ["chintu", "gudiya"]
    for n in names:
        probe(n)
