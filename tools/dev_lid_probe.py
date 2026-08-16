"""Row-by-row diagnostic for the eyelid band the blink samples.

`lid_sprite` walks up from the aperture, skips the lash, then collects rows
while they are still eyelid skin. When that walk stops early the strip
collapses to a flat fallback and a blink loses the artist's crease, so this
prints the evidence the walk acts on — per row, the fraction of pixels that
are ink, that are far from the skin reference, and the row's own median tone.

    python -m tools.dev_lid_probe [character ...]

Developer tool: read-only, renders nothing, ships no assets.
"""
from __future__ import annotations

import sys

import numpy as np
from PIL import Image

from engine.rig import Rig
from tools.art_eyes import (LASH_LUM_MAX, LASH_SAT_MAX, LID_BAND_FRAC,
                            LID_LASH_SKIP_FRAC, LID_ROW_DIRT_MAX,
                            LID_SKIN_DELTA, SAMPLE_MIN_PX, _luma)

CHARS = ("chintu", "gudiya")


def probe(name: str) -> None:
    rig = Rig.load(name)
    head = np.asarray(Image.open(rig.head_plate_path()).convert("RGB"),
                      dtype=np.float32)
    print(f"\n=== {name} · head plate {head.shape[1]}x{head.shape[0]} ===")

    for side in ("l", "r"):
        geo = rig.eye_dict(side == "l")
        ap = np.asarray(geo["aperture"], dtype=np.float64)
        if len(ap) < 3:
            print(f"  eye_{side}: no measured aperture")
            continue
        x0 = max(0, int(np.floor(ap[:, 0].min())) - 3)
        x1 = min(head.shape[1], int(np.ceil(ap[:, 0].max())) + 3)
        top = int(np.floor(ap[:, 1].min()))
        bot = int(np.ceil(ap[:, 1].max()))
        ap_h = max(3, bot - top)

        ref = head[max(0, top - ap_h):top, x0:x1].reshape(-1, 3)
        lum, sat = _luma(ref), ref.max(axis=1) - ref.min(axis=1)
        clean = ref[~((lum <= LASH_LUM_MAX) & (sat <= LASH_SAT_MAX))]
        skin = (np.median(clean, axis=0) if len(clean) >= SAMPLE_MIN_PX
                else np.median(ref, axis=0))

        skip = max(2, int(round(ap_h * LID_LASH_SKIP_FRAC)))
        cap = int(round(ap_h * LID_BAND_FRAC))
        print(f"  eye_{side}: x={x0}..{x1} top={top} bot={bot} h={ap_h} "
              f"skin={skin.round(0)} skip_cap={skip} band_cap={cap}")
        print(f"    {'dy':>4} {'ink%':>6} {'far%':>6} {'dirt%':>6}  median")
        for dy in range(1, min(top, ap_h) + 1):
            row = head[top - dy, x0:x1]
            lum = _luma(row)
            sat = row.max(axis=1) - row.min(axis=1)
            ink = (lum <= LASH_LUM_MAX) & (sat <= LASH_SAT_MAX)
            far = np.abs(row - skin).max(axis=1) > LID_SKIN_DELTA
            dirt = ink | far
            flag = "" if dirt.mean() <= LID_ROW_DIRT_MAX else "  <-- rejected"
            print(f"    {dy:>4} {ink.mean() * 100:6.1f} {far.mean() * 100:6.1f} "
                  f"{dirt.mean() * 100:6.1f}  {np.median(row, axis=0).round(0)}"
                  f"{flag}")


if __name__ == "__main__":
    for c in (sys.argv[1:] or CHARS):
        probe(c)
