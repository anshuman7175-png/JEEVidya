"""Dev diagnostic 2: prove WHY registration:iris_r fails.

Hypothesis: `_iris_center` is fed a whole half-frame in the registration
path, so `largest_component` can lock onto iris-toned material that is
not the iris (lash/brow/hair touching the eye). `_iris_datum` feeds the
same estimator a tight crop around the known centre and passes on all
four eyes.

This script measures the same eye three ways and writes a zoomed
visualisation so the contamination can be seen:

  A) half-frame  ∩ blink-ROI      (what the gate does today)
  B) tight window around pred     (what the datum does, re-centred)
  C) raw component stats          (area vs the known ellipse area)

Not a gate. Never imported by the engine.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.bone_engine import BoneEngine, PuppetPose            # noqa: E402
from engine.rig import Rig, rig_dir                              # noqa: E402
from tools.face_qc import color_mask, largest_component          # noqa: E402
from tools.verify_face import (                                  # noqa: E402
    IRIS_TOL, _iris_center, _iris_datum, _roi_from_variation,
)

OUT = "output/diag_iris"


def _mask_stats(m: np.ndarray):
    n = int(m.sum())
    if not n:
        return 0, None
    ys, xs = np.nonzero(m)
    return n, (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def diag(character: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    rig = Rig.load(character)
    engine = BoneEngine(rig)
    S = float(engine.scale)

    neutral = PuppetPose()
    open_frame = engine.render(neutral)
    closed = engine.render(PuppetPose(blink=1.0))
    pred = engine.predict(neutral)
    face_h = max(1.0, rig.head.face_height * S)
    eye_roi = _roi_from_variation([open_frame, closed], face_h)

    palette = rig.head.palette or {}
    iris_rgb = tuple(palette.get("iris", (92, 58, 38)))
    geos = {"iris_l": engine.assembly.eyes.left.geo,
            "iris_r": engine.assembly.eyes.right.geo}
    arts = {"iris_l": rig.head.art_eye_l, "iris_r": rig.head.art_eye_r}

    budget = 0.6 / 100.0 * face_h
    print(f"\n═══ {character} ═══   budget={budget:.3f}px  face_h={face_h:.1f}")

    mid_x = int((pred["iris_l"][0] + pred["iris_r"][0]) / 2.0)
    m_all = color_mask(open_frame, iris_rgb, tol=IRIS_TOL)
    m_roi = m_all & eye_roi if eye_roi is not None else m_all

    for eye in ("iris_l", "iris_r"):
        geo = geos[eye]
        ax, ay = (float(v) for v in geo.iris_axes)
        axes = (ax, ay, float(geo.iris_angle))
        ell_area = math.pi * ax * ay
        d = _iris_datum(Image.open(
            os.path.join(rig_dir(character), "head_canonical.png")
        ).convert("RGBA"), arts.get(eye) or {})
        dat = (0.0, 0.0) if d is None else (d[0] * S, d[1] * S)
        exp = (pred[eye][0] + dat[0], pred[eye][1] + dat[1])

        # ── A) today's half-frame region ──
        mA = np.zeros_like(m_roi)
        if pred[eye][0] < mid_x:
            mA[:, :mid_x] = m_roi[:, :mid_x]
        else:
            mA[:, mid_x:] = m_roi[:, mid_x:]
        detA = _iris_center(mA, axes)
        blobA = largest_component(mA)
        nA, bbA = _mask_stats(blobA)

        # ── B) tight window around the PREDICTION (datum's own pad) ──
        pad = int(max(ax, ay) * 2.2)
        cx, cy = exp
        x0, y0 = max(0, int(cx - pad)), max(0, int(cy - pad))
        x1 = min(m_roi.shape[1], int(cx + pad))
        y1 = min(m_roi.shape[0], int(cy + pad))
        mB = np.zeros_like(m_roi)
        mB[y0:y1, x0:x1] = m_roi[y0:y1, x0:x1]
        detB = _iris_center(mB, axes)
        blobB = largest_component(mB)
        nB, bbB = _mask_stats(blobB)

        def err(det):
            if det is None:
                return None
            return float(np.hypot(det[0] - exp[0], det[1] - exp[1]))

        eA, eB = err(detA), err(detB)
        print(f"\n  ── {eye} ──  ellipse_area={ell_area:.0f}px "
              f"axes=({ax:.1f},{ay:.1f})")
        print(f"     expect            = ({exp[0]:8.2f}, {exp[1]:8.2f})")
        print(f"     A half-frame blob : n={nA:6d} ({nA/ell_area:5.2f}× ellipse) "
              f"bbox={bbA}")
        print(f"       det={None if detA is None else (round(detA[0],2), round(detA[1],2))}"
              f"  ERR={'None' if eA is None else f'{eA:8.2f}px'}"
              f"  {'PASS' if eA is not None and eA <= budget else 'FAIL'}")
        print(f"     B tight  blob     : n={nB:6d} ({nB/ell_area:5.2f}× ellipse) "
              f"bbox={bbB}")
        print(f"       det={None if detB is None else (round(detB[0],2), round(detB[1],2))}"
              f"  ERR={'None' if eB is None else f'{eB:8.2f}px'}"
              f"  {'PASS' if eB is not None and eB <= budget else 'FAIL'}")

        # ── visualisation: zoomed crop, mask overlay, markers ──
        vpad = pad * 2
        vx0, vy0 = max(0, int(cx - vpad)), max(0, int(cy - vpad))
        vx1 = min(open_frame.width, int(cx + vpad))
        vy1 = min(open_frame.height, int(cy + vpad))
        crop = open_frame.crop((vx0, vy0, vx1, vy1)).convert("RGBA")
        ov = Image.new("RGBA", crop.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(ov)
        sub = blobA[vy0:vy1, vx0:vx1]
        ys, xs = np.nonzero(sub)
        for x, y in zip(xs, ys):
            od.point((int(x), int(y)), fill=(255, 0, 255, 110))
        comp = Image.alpha_composite(crop, ov)
        Z = 5
        comp = comp.resize((comp.width * Z, comp.height * Z), Image.NEAREST)
        cd = ImageDraw.Draw(comp)

        def mark(p, col, r=5):
            if p is None:
                return
            X = (p[0] - vx0) * Z
            Y = (p[1] - vy0) * Z
            cd.line([(X - r * 3, Y), (X + r * 3, Y)], fill=col, width=3)
            cd.line([(X, Y - r * 3), (X, Y + r * 3)], fill=col, width=3)

        mark(exp, (0, 255, 0, 255))       # green  = expected
        mark(detA, (255, 0, 0, 255))      # red    = today's detection
        mark(detB, (0, 160, 255, 255))    # blue   = tight-window detection
        p = os.path.join(OUT, f"{character}_{eye}.png")
        comp.convert("RGB").save(p)
        print(f"     viz → {p}   (green=expect red=half-frame blue=tight)")


if __name__ == "__main__":
    for c in (sys.argv[1:] or ["chintu", "gudiya"]):
        diag(c)
