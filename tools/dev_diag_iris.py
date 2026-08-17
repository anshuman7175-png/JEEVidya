"""Dev diagnostic: is registration:iris_r a PREDICTION bug or a DETECTION bug?

Prints, per character and per eye:
  * pred[]        — engine's analytic prediction (canvas px)
  * geo.iris_c    — assembly eyeball centre (canvas px, already scaled)
  * art iris_c*S  — bake's measured art centre, scaled to canvas
  * det           — what verify_face's detector reads off the render
  * mask stats    — pixel count + bbox of the half-plane mask actually used

Not a gate. Never imported by the engine.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.bone_engine import BoneEngine, PuppetPose            # noqa: E402
from engine.rig import Rig, rig_dir                              # noqa: E402
from tools.face_qc import color_mask                             # noqa: E402
from tools.verify_face import (                                  # noqa: E402
    IRIS_TOL, _iris_center, _iris_datum, _roi_from_variation,
)


def diag(character: str) -> None:
    rig = Rig.load(character)
    engine = BoneEngine(rig)
    S = float(engine.scale)

    neutral = PuppetPose()
    open_frame = engine.render(neutral)
    closed = engine.render(PuppetPose(blink=1.0))
    pred = engine.predict(neutral)

    face_h = max(1.0, rig.head.face_height * S)
    eye_roi = _roi_from_variation([open_frame, closed], face_h)

    plate_path = os.path.join(rig_dir(character), "head_canonical.png")
    plate = Image.open(plate_path).convert("RGBA")

    palette = rig.head.palette or {}
    iris_rgb = tuple(palette.get("iris", (92, 58, 38)))

    geos = {"iris_l": engine.assembly.eyes.left.geo,
            "iris_r": engine.assembly.eyes.right.geo}
    arts = {"iris_l": rig.head.art_eye_l, "iris_r": rig.head.art_eye_r}

    print(f"\n═══ {character} ═══")
    print(f"  canvas={engine.width}x{engine.height} scale={S:.5f} "
          f"face_h={face_h:.1f} iris_rgb={iris_rgb}")

    mid_x = int((pred["iris_l"][0] + pred["iris_r"][0]) / 2.0)
    print(f"  mid_x={mid_x}")

    m_iris_full = color_mask(open_frame, iris_rgb, tol=IRIS_TOL)
    m_iris = m_iris_full & eye_roi if eye_roi is not None else m_iris_full
    print(f"  iris-colour px: raw={int(m_iris_full.sum())} "
          f"∩roi={int(m_iris.sum())}")

    for eye in ("iris_l", "iris_r"):
        geo = geos[eye]
        art = arts.get(eye) or {}
        ax, ay = geo.iris_axes
        axes = (float(ax), float(ay), float(geo.iris_angle))

        d = _iris_datum(plate, art)
        art_c = art.get("iris_c") or (0.0, 0.0)
        art_axes = art.get("iris_axes") or (0.0, 0.0)

        m = np.zeros_like(m_iris)
        if pred[eye][0] < mid_x:
            side = "LEFT-half"
            m[:, :mid_x] = m_iris[:, :mid_x]
        else:
            side = "RIGHT-half"
            m[:, mid_x:] = m_iris[:, mid_x:]

        det = _iris_center(m, axes)
        n = int(m.sum())
        if n:
            ys, xs = np.nonzero(m)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        else:
            bbox = None

        print(f"\n  ── {eye} ({side}) ──")
        print(f"     pred        = ({pred[eye][0]:8.2f}, {pred[eye][1]:8.2f})")
        print(f"     geo.iris_c  = ({geo.iris_c[0]:8.2f}, "
              f"{geo.iris_c[1]:8.2f})  axes=({ax:.2f},{ay:.2f}) "
              f"ang={geo.iris_angle:.3f}")
        print(f"     art.iris_c  = ({float(art_c[0]):8.2f}, "
              f"{float(art_c[1]):8.2f})  ×S=("
              f"{float(art_c[0])*S:8.2f}, {float(art_c[1])*S:8.2f}) "
              f"art_axes=({float(art_axes[0]):.2f},{float(art_axes[1]):.2f})")
        print(f"     datum(plate)= {None if d is None else (round(d[0],2), round(d[1],2))}")
        print(f"     mask px={n} bbox={bbox}")
        if det is None:
            print("     det         = None  ← detector found nothing")
            continue
        print(f"     det         = ({det[0]:8.2f}, {det[1]:8.2f})")
        dd = (0.0, 0.0) if d is None else (d[0] * S, d[1] * S)
        exp = (pred[eye][0] + dd[0], pred[eye][1] + dd[1])
        err = float(np.hypot(det[0] - exp[0], det[1] - exp[1]))
        print(f"     expect      = ({exp[0]:8.2f}, {exp[1]:8.2f})")
        print(f"     ERR         = {err:8.2f}px  "
              f"(dx={det[0]-exp[0]:+.2f}, dy={det[1]-exp[1]:+.2f})")

        # Is the detector simply landing on the raw centroid instead?
        if n:
            ys, xs = np.nonzero(m)
            print(f"     raw centroid= ({xs.mean():8.2f}, {ys.mean():8.2f})")


if __name__ == "__main__":
    for c in (sys.argv[1:] or ["chintu", "gudiya"]):
        diag(c)
