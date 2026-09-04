"""Measure the opaque-pixel envelope of every safe body pose, per puppet,
as fractions of the render canvas HEIGHT (the compositor scales by height
and anchors bottom-centre). These numbers are what config/brand.SHOT_PRESETS
is solved against — run this whenever a rig or pose sheet changes.

    python tools/measure_envelope.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from engine.bone_engine import PuppetPose  # noqa: E402
from pipeline.puppet import PuppetActor  # noqa: E402


def main() -> None:
    for name in ("gudiya", "chintu"):
        a = PuppetActor(name)
        W, H = a.engine.width, a.engine.height
        print(f"\n== {name}: canvas {W}x{H}  aspect w/h={W / H:.3f}")
        poses = sorted(a._safe_poses() & set(a.pose_lib.pose_names)
                       & set(a.rig.poses))
        worst_l = worst_r = 0.0
        rows = []
        for p in ["neutral"] + [q for q in poses if q != "neutral"]:
            pose = PuppetPose()
            if p != "neutral":
                pose.body_pose = p
                pose.body_pose_to = p
                pose.body_pose_blend = 1.0
            img = a.engine.render(pose, a.engine.step_physics(pose))
            al = np.asarray(img.getchannel("A")) > 24
            ys, xs = np.where(al)
            cx = W / 2.0
            left = (cx - xs.min()) / H      # reach LEFT of anchor, in H units
            right = (xs.max() - cx) / H     # reach RIGHT of anchor
            top = (H - ys.min()) / H        # head top above canvas bottom
            bot = (H - ys.max()) / H        # content bottom → canvas bottom
            span = max(1, ys.max() - ys.min())
            head_band = al[ys.min(): ys.min() + int(0.16 * span)]
            _, hx = np.where(head_band)
            head_l = (cx - hx.min()) / H
            head_r = (hx.max() - cx) / H
            rows.append((p, left, right, top, bot, head_l, head_r))
            worst_l = max(worst_l, left)
            worst_r = max(worst_r, right)
        for r in rows:
            print("  pose %-8s L=%.3f R=%.3f top=%.3f footgap=%.3f  "
                  "head L=%.3f R=%.3f" % r)
        print(f"  WORST reach: L={worst_l:.3f}·H  R={worst_r:.3f}·H")


if __name__ == "__main__":
    main()
