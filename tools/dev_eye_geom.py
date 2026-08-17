"""Draw the BAKED eye geometry on the head plate, magnified.

Every other probe shows rendered frames, which conflates two questions:
"is the measurement right?" and "does the renderer use it right?". This
answers only the first — it draws, over the artwork itself:

    green   the measured aperture (the clip every eye pixel obeys)
    cyan    the fitted iris ellipse
    yellow  the lid strip's source rectangle (what a blink slides down)

and beside the face it lays out the three baked sprites — eyeball, socket
backdrop, lid strip — at the same magnification, since a defect in one of
them is invisible at plate scale.

    python -m tools.dev_eye_geom [character ...]

Developer tool: read-only, ships nothing.
"""
from __future__ import annotations

import math
import os
import sys
from typing import List, Tuple

from PIL import Image, ImageDraw

from engine.rig import Rig, rig_dir

OUT = "/tmp/agent-browser"
ZOOM = 5
CHARS = ("chintu", "gudiya")


def _sprite(character: str, rel: str) -> Image.Image | None:
    if not rel:
        return None
    p = rel if os.path.isabs(rel) else os.path.join(rig_dir(character), rel)
    return Image.open(p).convert("RGBA") if os.path.exists(p) else None


def _tile(img: Image.Image, label: str, zoom: int = ZOOM) -> Image.Image:
    big = img.convert("RGB").resize(
        (max(1, img.width * zoom), max(1, img.height * zoom)), Image.NEAREST)
    out = Image.new("RGB", (big.width, big.height + 18), (22, 22, 26))
    out.paste(big, (0, 0))
    ImageDraw.Draw(out).text((4, big.height + 4), label, fill=(240, 240, 245))
    return out


def _row(tiles: List[Image.Image]) -> Image.Image:
    if not tiles:
        return Image.new("RGB", (8, 8), (22, 22, 26))
    w = sum(t.width for t in tiles) + 6 * (len(tiles) + 1)
    h = max(t.height for t in tiles) + 12
    out = Image.new("RGB", (w, h), (16, 16, 20))
    x = 6
    for t in tiles:
        out.paste(t, (x, 6))
        x += t.width + 6
    return out


def probe(name: str) -> None:
    rig = Rig.load(name)
    if rig.head is None:
        print(f"{name}: no v3 head plate")
        return
    plate = Image.open(rig.head_plate_path()).convert("RGBA")

    for side, left in (("l", True), ("r", False)):
        geo = rig.head.eye_dict(left)
        ap = [tuple(map(float, p)) for p in geo.get("aperture") or ()]
        if len(ap) < 3:
            print(f"{name} eye_{side}: no measured aperture")
            continue
        x0 = int(min(p[0] for p in ap))
        x1 = int(max(p[0] for p in ap))
        y0 = int(min(p[1] for p in ap))
        y1 = int(max(p[1] for p in ap))
        pad = int(max(x1 - x0, y1 - y0) * 0.75)
        box = (max(0, x0 - pad), max(0, y0 - pad),
               min(plate.width, x1 + pad), min(plate.height, y1 + pad))

        over = plate.crop(box).convert("RGBA")
        d = ImageDraw.Draw(over)
        d.polygon([(p[0] - box[0], p[1] - box[1]) for p in ap],
                  outline=(60, 255, 90, 255))
        ic = geo.get("iris") or (0, 0, 0)
        axes = geo.get("iris_axes") or (ic[2], ic[2])
        ang = math.radians(float(geo.get("iris_angle") or 0.0))
        pts = []
        for i in range(72):
            t = 2 * math.pi * i / 72
            px, py = axes[0] * math.cos(t), axes[1] * math.sin(t)
            pts.append((ic[0] - box[0] + px * math.cos(ang) - py * math.sin(ang),
                        ic[1] - box[1] + px * math.sin(ang) + py * math.cos(ang)))
        d.line(pts + pts[:1], fill=(80, 230, 255, 255))

        tiles = [_tile(over, f"{name} eye_{side}: aperture/iris")]
        lid = _sprite(name, str(geo.get("lid_img") or ""))
        if lid is not None:
            lo = geo.get("lid_origin") or (0, 0)
            d.rectangle([lo[0] - box[0], lo[1] - box[1],
                         lo[0] - box[0] + lid.width - 1,
                         lo[1] - box[1] + lid.height - 1],
                        outline=(255, 220, 60, 255))
            tiles[0] = _tile(over, f"{name} eye_{side}: aperture/iris/lid src")
        for key, label in (("eyeball", "eyeball"), ("socket", "socket"),
                           ("lid_img", "lid strip")):
            s = _sprite(name, str(geo.get(key) or ""))
            if s is not None:
                flat = Image.new("RGB", s.size, (255, 0, 255))
                flat.paste(s, (0, 0), s)
                tiles.append(_tile(flat, f"{label} {s.width}x{s.height}"))
        os.makedirs(OUT, exist_ok=True)
        dest = f"{OUT}/{name}_geom_{side}.png"
        _row(tiles).save(dest)
        print(f"{name} eye_{side}: aperture {x1 - x0}x{y1 - y0} "
              f"iris_axes={tuple(round(float(a), 1) for a in axes)} "
              f"gaze_box={geo.get('gaze_box')} → {dest}")


if __name__ == "__main__":
    for c in (sys.argv[1:] or CHARS):
        probe(c)
