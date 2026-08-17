"""dev probe: is chintu's residual iris_r error an UNCANCELLED estimator
artefact rather than a renderer misplacement?

The registration gate compares

    detected_on_render   vs   predicted + datum_on_plate

where the datum is the same estimator's reading at rest. That
cancellation is only valid when both sides fitted a SIMILARLY CLIPPED
arc — `fit_fixed_axes_ellipse_ex` returns `arc_reach` precisely so a
caller can check that. `_iris_center` currently discards it, so nothing
does.

This prints the two fits side by side for all four eyes: chosen
component, rim residual, retained rim points, reach_u/reach_v, and the
axes each side used. If the eyes that PASS show matching reach across
plate/render while chintu_r does not, the residual is an estimator
artefact and the fix belongs in the estimator.

Nothing here is a gate; it only prints.

    .venv/bin/python -m tools.dev_diag_iris8
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.bone_engine import BoneEngine, PuppetPose
from engine.rig import Rig, rig_dir
from tools.face_qc import (REG_POS_TOL_FRAC, color_mask,
                           fit_fixed_axes_ellipse_ex, label_components)
from tools.verify_face import IRIS_TOL, _iris_center, _roi_from_variation


def _census(mask: np.ndarray, axes, label: str, top: int = 4) -> None:
    """Print each plausible component's fit so the SELECTION is visible."""
    ax, ay, ang = axes
    lab, n = label_components(mask)
    ell = math.pi * ax * ay
    print(f"      {label:6s} axes={ax:6.2f}x{ay:6.2f} ang={ang:6.2f} "
          f"ell_area={ell:7.1f}  ({n} comps)")
    if n == 0:
        return
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    for lb in np.argsort(counts)[::-1][:top]:
        a = float(counts[lb])
        if a <= 0:
            continue
        sub = lab == lb
        ys, xs = np.nonzero(sub)
        w = int(xs.max() - xs.min() + 1)
        h = int(ys.max() - ys.min() + 1)
        got = fit_fixed_axes_ellipse_ex(sub, (ax, ay), ang)
        if got is None:
            print(f"        area={a:6.0f} {w:3d}x{h:3d} fit=None")
            continue
        (cx, cy), err, kept, (ru, rv) = got
        print(f"        area={a:6.0f} {w:3d}x{h:3d} err={err:7.4f} "
              f"kept={kept:4d} reach_u={ru:.3f} reach_v={rv:.3f} "
              f"frac_ell={a / ell:5.2f} c=({cx:7.2f},{cy:7.2f})")


def main() -> int:
    for character in ("chintu", "gudiya"):
        print(f"\n═══════════ {character} ═══════════")
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
        print(f"  scale={S:.4f} face_h={face_h:.2f} budget={budget:.4f} "
              f"iris_rgb={iris_rgb}")

        for eye in ("iris_l", "iris_r"):
            art = arts[eye] or {}
            print(f"\n    ── {eye} ──")
            ax_p, ay_p = (float(v) for v in art["iris_axes"])
            cx_p, cy_p = (float(v) for v in art["iris_c"])
            ang_p = float(art.get("iris_angle") or 0.0)
            rgb_p = tuple(art["colors"]["iris"])

            # ---- PLATE side (produces the datum) ----
            pad = int(max(ax_p, ay_p) * 2.2)
            x0, y0 = max(0, int(cx_p - pad)), max(0, int(cy_p - pad))
            crop = plate.crop((x0, y0, int(cx_p + pad), int(cy_p + pad)))
            pmask = color_mask(crop, rgb_p, tol=IRIS_TOL)
            _census(pmask, (ax_p, ay_p, ang_p), "PLATE")
            pfit = _iris_center(pmask, (ax_p, ay_p, ang_p))
            if pfit is None:
                dat = (0.0, 0.0)
                print("        datum: NOT MEASURABLE")
            else:
                dat = (pfit[0] + x0 - cx_p, pfit[1] + y0 - cy_p)
                print(f"        datum(plate) dx={dat[0]:+.2f} "
                      f"dy={dat[1]:+.2f} |d|={math.hypot(*dat):.2f}"
                      f"   ->canvas dx={dat[0]*S:+.2f} dy={dat[1]*S:+.2f}")

            # ---- RENDER side ----
            gax, gay = geos[eye].iris_axes
            gang = float(geos[eye].iris_angle)
            m = np.zeros_like(m_iris)
            if pred[eye][0] < mid_x:
                m[:, :mid_x] = m_iris[:, :mid_x]
            else:
                m[:, mid_x:] = m_iris[:, mid_x:]
            _census(m, (float(gax), float(gay), gang), "RENDER")
            det = _iris_center(m, (float(gax), float(gay), gang))
            exp = (pred[eye][0] + dat[0] * S, pred[eye][1] + dat[1] * S)
            if det is None:
                print("        detected: NOT MEASURABLE")
                continue
            err = math.dist(exp, det)
            print(f"        expect=({exp[0]:8.2f},{exp[1]:8.2f})  "
                  f"detect=({det[0]:8.2f},{det[1]:8.2f})")
            print(f"        ERR={err:7.4f} budget={budget:.4f}  "
                  f"{'PASS' if err <= budget else 'FAIL'}   "
                  f"(dx={det[0]-exp[0]:+.2f} dy={det[1]-exp[1]:+.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
