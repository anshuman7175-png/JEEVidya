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
from tools.art_eyes import (LID_BAND_FRAC, LID_LASH_SKIP_FRAC,
                            LID_REF_EMA, LID_ROW_INK_MAX, LID_SKIN_DELTA,
                            _ink, _row_tone)

CHARS = ("chintu", "gudiya")


def probe(name: str) -> None:
    rig = Rig.load(name)
    if rig.head is None:
        print(f"\n=== {name}: no v3 head plate — run `jvmake.py rig` first")
        return
    head = np.asarray(Image.open(rig.head_plate_path()).convert("RGB"),
                      dtype=np.float32)
    print(f"\n=== {name} · head plate {head.shape[1]}x{head.shape[0]} ===")

    for side in ("l", "r"):
        geo = rig.head.eye_dict(side == "l")
        ap = np.asarray(geo["aperture"], dtype=np.float64)
        if len(ap) < 3:
            print(f"  eye_{side}: no measured aperture")
            continue
        x0 = max(0, int(np.floor(ap[:, 0].min())) - 3)
        x1 = min(head.shape[1], int(np.ceil(ap[:, 0].max())) + 3)
        top = int(np.floor(ap[:, 1].min()))
        bot = int(np.ceil(ap[:, 1].max()))
        ap_h = max(3, bot - top)

        skip = max(2, int(round(ap_h * LID_LASH_SKIP_FRAC)))
        cap = int(round(ap_h * LID_BAND_FRAC))
        print(f"  eye_{side}: x={x0}..{x1} top={top} bot={bot} h={ap_h} "
              f"skip_cap={skip} band_cap={cap}")
        print(f"    {'dy':>4} {'ink%':>6}  {'tone':<18} {'Δref':>6}  verdict")

        # Mirror `lid_sprite`: lash by ink alone, then a tracking reference.
        ref = None
        seen = 0
        state = "lash"
        for dy in range(1, min(top, ap_h) + 1):
            row = head[top - dy, x0:x1]
            ink = float(np.mean(_ink(row)))
            tone = _row_tone(row)
            if state == "lash":
                if ink <= LID_ROW_INK_MAX:
                    state = "band"
                    ref = tone
                else:
                    print(f"    {dy:>4} {ink * 100:6.1f}  {'':<18} {'':>6}  lash skip")
                    continue
            d = (float(np.abs(tone - ref).max())
                 if (tone is not None and ref is not None) else float("inf"))
            if state == "band" and (tone is None or ink > LID_ROW_INK_MAX
                                    or d > LID_SKIN_DELTA):
                state = "done"
            verdict = ("accepted" if state == "band" and seen < cap
                       else "STOP" if state == "done" else "over cap")
            if state == "band" and seen < cap:
                seen += 1
                ref = (1.0 - LID_REF_EMA) * ref + LID_REF_EMA * tone
            t = "—" if tone is None else str(tone.round(0))
            print(f"    {dy:>4} {ink * 100:6.1f}  {t:<18} {d:6.1f}  {verdict}")
            if state != "band":
                break
        print(f"    → band {seen} rows")


if __name__ == "__main__":
    for c in (sys.argv[1:] or CHARS):
        probe(c)
