"""dev probe: does letting SCALE float remove the arc-coverage bias?

Measured finding this probe exists to test (see dev_diag_iris6): on all four
eyes the iris-coloured mask keeps a mostly ONE-SIDED arc (reach_v ~= 0), and
a FIXED-AXES fit of a one-sided arc is biased along the axis the arc does
not straddle. The bias is not noise: it is the colour tolerance eroding the
crescent INWARD by a pixel or two, which a fixed-size ellipse can only
explain by moving its centre.

If that reading is right, then fitting (cx, cy, s) with the axis RATIO and
angle held fixed should absorb the inset into `s` and leave the centre
unbiased on every eye, whatever arc survived. That would make the plate
datum ~0 and let both sides of the gate agree without calibration.

Run:  .venv/bin/python -m tools.dev_diag_iris7
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.bone_engine import BoneEngine, PuppetPose      # noqa: E402
from engine.rig import Rig, rig_dir                        # noqa: E402
from tools import verify_face as VF                        # noqa: E402
from tools.face_qc import (ARC_MIN_POINTS, ARC_R_HI,       # noqa: E402
                          ARC_R_LO, boundary_points,
                          color_mask, label_components)


def _resid(pts, c, ax, ay, ct, st):
    dx = pts[:, 0] - c[0]
    dy = pts[:, 1] - c[1]
    u = (dx * ct + dy * st) / ax
    v = (-dx * st + dy * ct) / ay
    return np.sqrt(u * u + v * v)


def fit_scaled(mask, axes, angle_deg):
    """Fit (cx, cy, s): axis RATIO and angle fixed, overall size free."""
    ax, ay = float(axes[0]), float(axes[1])
    pts = boundary_points(mask)
    if len(pts) < ARC_MIN_POINTS:
        return None
    th = math.radians(angle_deg)
    ct, st = math.cos(th), math.sin(th)
    xs, ys = pts[:, 0].astype(float), pts[:, 1].astype(float)
    cx = 0.5 * (xs.min() + xs.max())
    cy = 0.5 * (ys.min() + ys.max())
    s = 1.0
    for _ in range(80):
        r = _resid(pts, (cx, cy), ax * s, ay * s, ct, st)
        keep = (r >= ARC_R_LO) & (r <= ARC_R_HI)
        if keep.sum() < ARC_MIN_POINTS:
            return None
        p = pts[keep]
        dx = p[:, 0] - cx
        dy = p[:, 1] - cy
        A, B = ax * s, ay * s
        u = (dx * ct + dy * st) / A
        v = (-dx * st + dy * ct) / B
        f = u * u + v * v - 1.0
        jx = 2.0 * (u * (-ct / A) + v * (st / B))
        jy = 2.0 * (u * (-st / A) + v * (-ct / B))
        js = -2.0 * (u * u + v * v) / s
        J = np.stack([jx, jy, js], axis=1)
        H = J.T @ J
        g = -(J.T @ f)
        try:
            step = np.linalg.solve(H + 1e-9 * np.eye(3), g)
        except np.linalg.LinAlgError:
            return None
        cx += float(step[0])
        cy += float(step[1])
        s += float(step[2])
        if s <= 0.05:
            return None
        if max(abs(float(step[0])), abs(float(step[1]))) < 1e-5:
            break
    r = _resid(pts, (cx, cy), ax * s, ay * s, ct, st)
    keep = (r >= ARC_R_LO) & (r <= ARC_R_HI)
    if keep.sum() < ARC_MIN_POINTS:
        return None
    err = float(np.mean(np.abs(r[keep] - 1.0)))
    return (cx, cy), err, s, int(keep.sum())


def best_scaled(mask, axes, angle_deg):
    ax, ay = float(axes[0]), float(axes[1])
    lab, n = label_components(mask)
    if n == 0:
        return None
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    ell = math.pi * ax * ay
    best = None
    tried = 0
    for lb in np.argsort(counts)[::-1]:
        a = float(counts[lb])
        if a <= 0:
            break
        if a < max(24.0, 0.04 * ell) or a > 1.60 * ell:
            continue
        got = fit_scaled(lab == lb, (ax, ay), angle_deg)
        tried += 1
        if got is not None and (best is None or got[1] < best[1]):
            best = got
        if tried >= 8:
            break
    return best


def main() -> int:
    for ch in ("chintu", "gudiya"):
        rig = Rig.load(ch)
        eng = BoneEngine(rig)
        S = float(eng.scale)
        face_h = max(1.0, rig.head.face_height * eng.scale)
        neutral = PuppetPose()
        of = eng.render(neutral)
        cf = eng.render(PuppetPose(blink=1.0))
        pred = eng.predict(neutral)
        roi = VF._roi_from_variation([of, cf], face_h)
        iris_rgb = tuple((rig.head.palette or {}).get("iris", (92, 58, 38)))
        mi = color_mask(of, iris_rgb, tol=VF.IRIS_TOL)
        if roi is not None:
            mi = mi & roi
        mid = int((pred["iris_l"][0] + pred["iris_r"][0]) / 2.0)
        plate = Image.open(
            os.path.join(rig_dir(ch), "head_canonical.png")).convert("RGBA")
        budget = VF.REG_TOL_FRAC * face_h
        print(f"=== {ch}  scale={S:.4f} budget={budget:.2f}px")
        for eye, geo, art in (
                ("iris_l", eng.assembly.eyes.left.geo, rig.head.art_eye_l),
                ("iris_r", eng.assembly.eyes.right.geo, rig.head.art_eye_r)):
            gax, gay = geo.iris_axes
            gang = float(geo.iris_angle)
            mm = np.zeros_like(mi)
            if pred[eye][0] < mid:
                mm[:, :mid] = mi[:, :mid]
            else:
                mm[:, mid:] = mi[:, mid:]
            r_got = best_scaled(mm, (gax, gay), gang)
            # plate datum through the SAME estimator
            pax, pay = (float(v) for v in art["iris_axes"])
            pcx, pcy = (float(v) for v in art["iris_c"])
            pang = float(art.get("iris_angle") or 0.0)
            prgb = tuple(art["colors"]["iris"])
            pad = int(max(pax, pay) * 2.2)
            x0, y0 = max(0, int(pcx - pad)), max(0, int(pcy - pad))
            crop = plate.crop((x0, y0, int(pcx + pad), int(pcy + pad)))
            p_got = best_scaled(color_mask(crop, prgb, tol=VF.IRIS_TOL),
                                (pax, pay), pang)
            if r_got is None or p_got is None:
                print(f"  {eye}: render={r_got is not None} "
                      f"plate={p_got is not None}  -> NOT MEASURABLE")
                continue
            (rx, ry), rerr, rs, rn = r_got
            (px, py), perr, ps, pn = p_got
            raw = math.dist((rx, ry), pred[eye])
            dxp = (px + x0 - pcx) * S
            dyp = (py + y0 - pcy) * S
            cdx = rx - pred[eye][0] - dxp
            cdy = ry - pred[eye][1] - dyp
            print(f"  {eye}: render c=({rx:8.2f},{ry:8.2f}) s={rs:.3f} "
                  f"rim={rerr:.4f} n={rn:3d}")
            print(f"        raw dx={rx - pred[eye][0]:+6.2f} "
                  f"dy={ry - pred[eye][1]:+6.2f}  raw_err={raw:5.2f}")
            print(f"        plate datum (canvas px) "
                  f"dx={dxp:+6.2f} dy={dyp:+6.2f} s={ps:.3f}")
            print(f"        CORRECTED err={math.hypot(cdx, cdy):5.2f} "
                  f"({'PASS' if math.hypot(cdx, cdy) <= budget else 'FAIL'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
