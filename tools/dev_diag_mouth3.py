"""Diagnostic: WHERE does the mouth's constant downward offset come from?

`dev_diag_mouth2` established that the offset is a near-constant dy
(chintu ≈ 5.4px, gudiya ≈ 12.8px) that the ring-centroid correction only
partly explains. Two suspects remain, and they live in different places:

  A. the RENDERER — the art viseme sprite is quad-warped into the outer
     contour's bbox at weight 0.55 (`MouthRasterizer._warp_art`). Nothing
     registers that sprite's own lips to the contour, so if the artist's
     lips sit low inside their plate, the painted body really is lower
     than the parametric contour claims.
  B. the DETECTOR — the ROI is the mouth's variation bbox grown by 5% of
     face height, and the plate's own chin/lip shadow falls inside it, so
     static artwork could be dragging the centroid down.

Both are measurable, separately, with no affine and no plate involved:
render the mouth PATCH alone and compare its lip-colour centroid against
the analytic ring centroid in patch coordinates.

  proc  — patch with art_weight = 0   (pure parametric mouth)
  art   — patch as the renderer actually composites it (art_weight 0.55)

Usage:  python3 -m tools.dev_diag_mouth3 [chintu gudiya]
"""
from __future__ import annotations

import sys
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from engine.bone_engine import BoneEngine, PuppetPose
from engine.mouth_model import lip_contour
from engine.rig import Rig
from tools.dev_diag_mouth2 import ring_centroid
from tools.face_qc import color_mask, mask_centroid
from tools.verify_face import LIP_TOL, _baked_viseme_classes


def patch_centroid(patch: Image.Image, lip: Tuple[int, int, int],
                   shadow: Optional[Tuple[int, int, int]]
                   ) -> Optional[Tuple[float, float]]:
    """Lip-colour centroid of a mouth patch on transparent background.

    The patch is RGBA with real transparency, so it is flattened onto a
    colour NOTHING in the palette matches before the colour mask runs —
    otherwise every transparent pixel reads as whatever RGB happens to sit
    under alpha 0.
    """
    flat = Image.new("RGB", patch.size, (0, 255, 0))
    flat.paste(patch, (0, 0), patch)
    m = color_mask(flat, lip, tol=LIP_TOL)
    if shadow is not None:
        m |= color_mask(flat, shadow, tol=LIP_TOL)
    a = np.asarray(patch.getchannel("A")) > 8
    return mask_centroid(m & a)


def alpha_centroid(patch: Image.Image) -> Optional[Tuple[float, float]]:
    a = np.asarray(patch.getchannel("A"), dtype=np.float64)
    tot = a.sum()
    if tot <= 0:
        return None
    ys, xs = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    return (float((a * xs).sum() / tot), float((a * ys).sum() / tot))


def diag(character: str) -> None:
    rig = Rig.load(character)
    eng = BoneEngine(rig)
    asm = eng.assembly
    mr = asm.mouth
    pal = rig.head.palette or {}
    lip = tuple(pal.get("lip", (170, 80, 80)))
    shadow = tuple(pal["lip_shadow"]) if "lip_shadow" in pal else None
    ox, oy = mr.origin()
    print(f"\n═══ {character}  patch origin=({ox:.2f},{oy:.2f}) "
          f"center={mr.center} scale={mr.scale:.2f}")
    print(f"    {'viseme':15s} {'dy_proc':>8s} {'dy_art':>8s} "
          f"{'dx_proc':>8s} {'dx_art':>8s}   ring(plate)")
    for v in _baked_viseme_classes(rig):
        pose = PuppetPose(viseme=v, viseme_to=v, mouth_open=1.0).clamped()
        ch, _, _, _, _ = eng._channels(pose, None)
        p = ch.mouth
        outer_n, inner_n = lip_contour(p)
        cx, cy = mr.center
        outer = [(cx + x * mr.scale, cy + y * mr.scale) for x, y in outer_n]
        inner = [(cx + x * mr.scale, cy + y * mr.scale) for x, y in inner_n]
        rx, ry = ring_centroid(outer, inner)
        # analytic ring centroid in PATCH pixel coordinates
        exp = (rx - ox, ry - oy)

        art = asm.art.get(v)
        rows = []
        for w in (0.0, 0.55 if art is not None else 0.0):
            patch = mr.render(p, v, art if w > 0 else None, w)
            got = patch_centroid(patch, lip, shadow)
            rows.append(None if got is None
                        else (got[0] - exp[0], got[1] - exp[1]))
        (dxp, dyp) = rows[0] or (float("nan"),) * 2
        (dxa, dya) = rows[1] or (float("nan"),) * 2
        print(f"    {v:15s} {dyp:8.2f} {dya:8.2f} {dxp:8.2f} {dxa:8.2f}"
              f"   ({rx:7.2f},{ry:7.2f})  art={'yes' if art else 'no'}")


def main(argv: Sequence[str]) -> int:
    for c in (argv or ["chintu", "gudiya"]):
        diag(c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
