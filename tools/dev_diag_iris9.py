"""dev probe: does letting a uniform SCALE float (axis ratio + angle still
fixed) remove the reach-dependent centre bias that dev_diag_iris8 measured?

iris8 established that the residual chintu iris_r error is not a
misplacement but an UNCANCELLED estimator artefact: plate and render
fitted differently-clipped arcs (reach_v 0.337 vs 0.000) and the datum,
which is only valid for similarly-clipped arcs, could not cancel it.
Error tracked Δreach_v monotonically across all four eyes.

WHY A FLOATING SCALE SHOULD FIX IT
The mask boundary sits slightly INSIDE the true rim: colour tolerance and
antialiasing eat the outermost pixels, so the visible arc is inset by some
δ. With the axes pinned, the only way the fit can push the arc's radius
back to 1.0 is to move the CENTRE away from whichever side the arc is on —
a bottom crescent drags the centre up. The bias is therefore a function of
which arc survived, i.e. of reach. Let one scale float and the same inset
is absorbed by s ≈ 1 − δ/a with the centre left where it is, so the
centre stops depending on which arc survived.

An arc with curvature still determines (cx, cy, s): three points fix a
circle. reach_v = 0 does not make it unmeasurable, only less
well-conditioned.

Prints, per eye, fixed vs scaled on BOTH sides, the fitted s, and the
registration error WITH and WITHOUT the datum — so it is visible whether
the scaled fit is unbiased enough to need no datum at all.

Nothing here is a gate; it only prints.

    .venv/bin/python -m tools.dev_diag_iris9
"""
from __future__ import annotations

import math
import os
import sys
from typing import Optional, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.bone_engine import BoneEngine, PuppetPose
from engine.rig import Rig, rig_dir
from tools.face_qc import (ARC_ITERS, ARC_MIN_POINTS, ARC_R_HI, ARC_R_LO,
                           CAND_AREA_HI_FRAC, CAND_AREA_LO_FRAC,
                           MIN_COMPONENT_PX, REG_POS_TOL_FRAC, boundary_points,
                           color_mask, fit_ellipse_best_component,
                           label_components)
from tools.verify_face import IRIS_TOL, _roi_from_variation

# A real eyeball's visible arc is inset by antialiasing, never inflated,
# so s should land just under 1. A fit that needs to grow or shrink the
# eyeball a lot has found something that is not the eyeball.
S_LO, S_HI = 0.72, 1.15


def fit_scaled(mask: np.ndarray, axes: Tuple[float, float],
               angle_deg: float = 0.0):
    """((cx,cy), rim_err, n, (reach_u,reach_v), s) with ratio+angle fixed."""
    ax, ay = float(axes[0]), float(axes[1])
    if ax <= 0.0 or ay <= 0.0:
        return None
    pts = boundary_points(mask)
    if len(pts) < ARC_MIN_POINTS:
        return None
    th = math.radians(float(angle_deg))
    ct, st = math.cos(th), math.sin(th)
    xs, ys = pts[:, 0], pts[:, 1]

    def norm(c, s):
        dx = pts[:, 0] - c[0]
        dy = pts[:, 1] - c[1]
        u = (dx * ct + dy * st) / (ax * s)
        v = (-dx * st + dy * ct) / (ay * s)
        return u, v, np.sqrt(u * u + v * v)

    mid_x = 0.5 * (float(xs.min()) + float(xs.max()))
    seeds = [(mid_x, float(ys.max()) - ay),
             (mid_x, float(ys.min()) + ay),
             (mid_x, 0.5 * (float(ys.min()) + float(ys.max())))]

    def cost(c):
        _, _, r = norm(c, 1.0)
        keep = (r >= ARC_R_LO) & (r <= ARC_R_HI)
        if keep.sum() < ARC_MIN_POINTS:
            return math.inf
        return float(np.mean(np.abs(r[keep] - 1.0)))

    best = min(seeds, key=cost)
    if not math.isfinite(cost(best)):
        return None
    cx, cy = best
    s = 1.0
    for _ in range(ARC_ITERS):
        u, v, r = norm((cx, cy), s)
        keep = (r >= ARC_R_LO) & (r <= ARC_R_HI)
        if keep.sum() < ARC_MIN_POINTS:
            return None
        uu, vv = u[keep], v[keep]
        f = uu * uu + vv * vv - 1.0
        # residual r²−1 in coords already divided by s
        jx = 2.0 * (uu * (-ct / (ax * s)) + vv * (st / (ay * s)))
        jy = 2.0 * (uu * (-st / (ax * s)) + vv * (-ct / (ay * s)))
        js = -2.0 * (uu * uu + vv * vv) / s
        J = np.stack([jx, jy, js], axis=1)
        A = J.T @ J
        b = -(J.T @ f)
        try:
            step = np.linalg.solve(A + 1e-12 * np.eye(3), b)
        except np.linalg.LinAlgError:
            break
        cx += float(step[0])
        cy += float(step[1])
        s += float(step[2])
        if not (S_LO * 0.5 < s < S_HI * 1.5):
            return None
        if max(abs(step[0]), abs(step[1]), abs(step[2])) < 1e-4:
            break
    u, v, r = norm((cx, cy), s)
    keep = (r >= ARC_R_LO) & (r <= ARC_R_HI)
    if keep.sum() < ARC_MIN_POINTS:
        return None
    rim = float(np.mean(np.abs(r[keep] - 1.0)))
    uu, vv = u[keep], v[keep]
    ru = min(abs(float(uu.max())), abs(float(uu.min()))) \
        if (uu.max() > 0.0 > uu.min()) else 0.0
    rv = min(abs(float(vv.max())), abs(float(vv.min()))) \
        if (vv.max() > 0.0 > vv.min()) else 0.0
    return ((float(cx), float(cy)), rim, int(keep.sum()), (ru, rv), float(s))


def best_scaled(mask, axes, angle_deg=0.0, min_px=MIN_COMPONENT_PX):
    """Same candidate window + least-rim-residual selection as the
    committed fixed-axes selector, so only the FIT differs."""
    ax, ay = float(axes[0]), float(axes[1])
    lab, n = label_components(mask)
    if n == 0:
        return None
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    ell = math.pi * ax * ay
    lo = max(float(min_px), CAND_AREA_LO_FRAC * ell)
    hi = CAND_AREA_HI_FRAC * ell
    best = None
    for lb in np.argsort(counts)[::-1]:
        a = float(counts[lb])
        if a <= 0.0:
            break
        if a < lo or a > hi:
            continue
        got = fit_scaled(lab == lb, (ax, ay), angle_deg)
        if got is None:
            continue
        if not (S_LO <= got[4] <= S_HI):
            continue
        if best is None or got[1] < best[1]:
            best = ((got[0][0], got[0][1]), got[1], int(a), got[3], got[4])
    return best


def main() -> int:
    print("fixed = committed fit_ellipse_best_component (axes pinned)")
    print("scaled = prototype, axis ratio + angle pinned, one scale free\n")
    rows = []
    for character in ("chintu", "gudiya"):
        rig = Rig.load(character)
        engine = BoneEngine(rig)
        S = float(engine.scale)
        face_h = max(1.0, rig.head.face_height * S)
        budget = REG_POS_TOL_FRAC * face_h
        open_frame = engine.render(PuppetPose())
        closed = engine.render(PuppetPose(blink=1.0))
        pred = engine.predict(PuppetPose())
        eye_roi = _roi_from_variation([open_frame, closed], face_h)
        plate = Image.open(
            os.path.join(rig_dir(character), "head_canonical.png")
        ).convert("RGBA")
        palette = rig.head.palette or {}
        iris_rgb = tuple(palette.get("iris", (92, 58, 38)))
        m_iris = color_mask(open_frame, iris_rgb, tol=IRIS_TOL)
        if eye_roi is not None:
            m_iris = m_iris & eye_roi
        mid_x = int((pred["iris_l"][0] + pred["iris_r"][0]) / 2.0)
        arts = {"iris_l": rig.head.art_eye_l, "iris_r": rig.head.art_eye_r}
        geos = {"iris_l": engine.assembly.eyes.left.geo,
                "iris_r": engine.assembly.eyes.right.geo}
        print(f"═══ {character}  budget={budget:.4f}px ═══")
        for eye in ("iris_l", "iris_r"):
            art = arts[eye] or {}
            ax_p, ay_p = (float(v) for v in art["iris_axes"])
            cx_p, cy_p = (float(v) for v in art["iris_c"])
            ang_p = float(art.get("iris_angle") or 0.0)
            rgb_p = tuple(art["colors"]["iris"])
            pad = int(max(ax_p, ay_p) * 2.2)
            x0, y0 = max(0, int(cx_p - pad)), max(0, int(cy_p - pad))
            crop = plate.crop((x0, y0, int(cx_p + pad), int(cy_p + pad)))
            pmask = color_mask(crop, rgb_p, tol=IRIS_TOL)

            gax, gay = geos[eye].iris_axes
            gang = float(geos[eye].iris_angle)
            m = np.zeros_like(m_iris)
            if pred[eye][0] < mid_x:
                m[:, :mid_x] = m_iris[:, :mid_x]
            else:
                m[:, mid_x:] = m_iris[:, mid_x:]

            print(f"\n  ── {eye} ──")
            for tag in ("fixed", "scaled"):
                if tag == "fixed":
                    pf = fit_ellipse_best_component(pmask, (ax_p, ay_p), ang_p)
                    rf = fit_ellipse_best_component(m, (float(gax), float(gay)),
                                                    gang)
                    ps = rs = None
                else:
                    pf = best_scaled(pmask, (ax_p, ay_p), ang_p)
                    rf = best_scaled(m, (float(gax), float(gay)), gang)
                    ps = pf[4] if pf else None
                    rs = rf[4] if rf else None
                if pf is None or rf is None:
                    print(f"    {tag:6s} NOT MEASURABLE "
                          f"(plate={pf is not None} render={rf is not None})")
                    continue
                dat = (pf[0][0] + x0 - cx_p, pf[0][1] + y0 - cy_p)
                det = rf[0]
                raw = math.dist(pred[eye], det)
                exp = (pred[eye][0] + dat[0] * S, pred[eye][1] + dat[1] * S)
                err = math.dist(exp, det)
                ss = "" if ps is None else f" s_p={ps:.3f} s_r={rs:.3f}"
                print(f"    {tag:6s} reach_v p={pf[3][1]:.3f} r={rf[3][1]:.3f}"
                      f"  |datum|={math.hypot(*dat):5.2f}{ss}")
                print(f"           err_nodatum={raw:7.4f}  "
                      f"err_datum={err:7.4f}  "
                      f"{'PASS' if err <= budget else 'FAIL'}")
                rows.append((character, eye, tag, raw, err, budget))
    print("\n═══ summary (err_datum vs budget) ═══")
    for c, e, t, raw, err, b in rows:
        print(f"  {c:7s} {e:7s} {t:6s} nodatum={raw:8.4f} "
              f"datum={err:8.4f} budget={b:.4f} "
              f"{'PASS' if err <= b else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
