"""
JEEVidya — Newton's Mountain Thought Experiment Simulation (Visual Bible v1 §6.3)
══════════════════════════════════════════════════════════════════════════════
Renders broadcast-grade 60fps orbital mechanics thought experiment (1687):
  • Central Body: Photorealistic high-resolution Earth with atmospheric corona glow.
  • Mountain: Pinnacle at the North Pole with a mountain summit and launcher platform.
  • Trajectories:
      1. 'drop': Ball dropped from rest ($v = 0$), falls vertically to the base.
      2. 'throw_short': Moderate impulse ($v = 3.5\\text{ km/s}$), arcs and hits ground nearby.
      3. 'throw_far': High impulse ($v = 6.5\\text{ km/s}$), curves over horizon onto far hemisphere.
      4. 'orbital_loop': Orbital velocity ($v = 7.9\\text{ km/s}$), closed perpetual free-fall loop!
  • Kinetic Comet Particles & Vectors:
      - Tangential velocity vector (Cyan #00E5FF)
      - Inward gravitational pull (Coral #FF5252)
  • Typography: Clean shadows, no clunky opaque boxes, Shorts safe-zone compliance.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import settings


def _load_font(font_name: str, size: int) -> ImageFont.FreeTypeFont:
    """Load bundled Google Fonts with graceful OS system font fallbacks."""
    paths = [
        os.path.join(settings.PROJECT_ROOT, "assets", "fonts", font_name),
        os.path.join(settings.PROJECT_ROOT, "assets", "fonts", "Mukta-ExtraBold.ttf"),
        os.path.join(settings.PROJECT_ROOT, "assets", "fonts", "Baloo2-Variable.ttf"),
        f"/System/Library/Fonts/Supplemental/{font_name}",
        f"/Library/Fonts/{font_name}",
        f"/System/Library/Fonts/{font_name}",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _draw_shadow_text(draw: ImageDraw.Draw, x: float, y: float, text: str,
                      font: ImageFont.FreeTypeFont, fill: Tuple[int, int, int],
                      anchor: str = "mm", rs: float = 1.0) -> None:
    """Draw broadcast typography with soft ambient drop shadow."""
    off = max(1, int(2 * rs))
    for ox in (-off, 0, off):
        for oy in (-off, 0, off):
            if ox != 0 or oy != 0:
                draw.text((x + ox, y + oy), text, font=font, fill=(5, 10, 25, 230), anchor=anchor)
    draw.text((x, y), text, font=font, fill=fill + (255,), anchor=anchor)


def _draw_hud_vector(draw: ImageDraw.Draw, x1: float, y1: float, x2: float, y2: float,
                     color: Tuple[int, int, int], label: str = "",
                     width: int = 4, rs: float = 1.0,
                     label_pos: Optional[Tuple[float, float]] = None) -> None:
    """Draw sleek HUD vector arrow with glowing trail and non-colliding text."""
    draw.line([(x1, y1), (x2, y2)], fill=color + (90,), width=width + int(4 * rs))
    draw.line([(x1, y1), (x2, y2)], fill=color + (255,), width=width)

    angle = math.atan2(y2 - y1, x2 - x1)
    head_len = 15 * rs
    p1 = (x2 - head_len * math.cos(angle - 0.36), y2 - head_len * math.sin(angle - 0.36))
    p2 = (x2 - head_len * math.cos(angle + 0.36), y2 - head_len * math.sin(angle + 0.36))
    draw.polygon([(x2, y2), p1, p2], fill=color + (255,))

    if label:
        font_main = _load_font("Mukta-ExtraBold.ttf", max(11, int(15 * rs)))
        if label_pos is not None:
            lx, ly = label_pos
        else:
            dist = 22 * rs
            lx = x2 + dist * math.cos(angle)
            ly = y2 + dist * math.sin(angle)
        _draw_shadow_text(draw, lx, ly, label, font_main, fill=color, anchor="mm", rs=rs)


class NewtonMountainSimulation:
    """Broadcast-grade 60fps simulation of Newton's Mountain thought experiment."""

    def __init__(self, mode: str = "throw_short", **kwargs: Any):
        self.mode = mode
        self.kwargs = kwargs
        self._earth_src: Optional[Image.Image] = None
        self._cached_earth: Optional[Tuple[int, Image.Image]] = None
        self._load_assets()

    def _load_assets(self) -> None:
        earth_p = os.path.join(settings.PROJECT_ROOT, "assets", "sims", "earth_photoreal.png")
        if os.path.exists(earth_p):
            try:
                self._earth_src = Image.open(earth_p).convert("RGBA")
            except Exception:
                self._earth_src = None

    def render(self, t_seconds: float, params: Dict[str, Any],
               width: int = 1080, height: int = 820, res_scale: float = 1.0) -> Image.Image:
        """Render one 60fps simulation frame on the Lab viewport canvas."""
        rs = res_scale
        w_px = int(width)
        h_px = int(height)
        canvas = Image.new("RGBA", (w_px, h_px), (0, 0, 0, 0))
        d = ImageDraw.Draw(canvas)

        mode = params.get("mode", self.mode).lower().strip()
        if mode in ("circular", "orbit", "newton_curvature"):
            mode = "orbital_loop"
        elif mode in ("short", "throw"):
            mode = "throw_short"
        elif mode in ("far", "miss"):
            mode = "throw_far"

        center_x = w_px / 2.0
        center_y = (h_px / 2.0) + 75.0 * rs

        earth_r = 160.0 * rs
        mtn_h = 42.0 * rs
        r_top = earth_r + mtn_h

        # ── 0. Deep Space Starfield ──
        for sx, sy, s_rad, s_seed in [
            (0.10 * w_px, 0.20 * h_px, 1.8 * rs, 1.1),
            (0.20 * w_px, 0.70 * h_px, 1.2 * rs, 2.3),
            (0.88 * w_px, 0.26 * h_px, 2.0 * rs, 3.5),
            (0.90 * w_px, 0.65 * h_px, 1.4 * rs, 4.2),
            (0.08 * w_px, 0.50 * h_px, 1.0 * rs, 5.7),
            (0.80 * w_px, 0.82 * h_px, 1.5 * rs, 6.1),
            (0.32 * w_px, 0.16 * h_px, 1.2 * rs, 7.8),
            (0.68 * w_px, 0.18 * h_px, 1.6 * rs, 8.4),
        ]:
            twinkle = 0.55 + 0.45 * math.sin(t_seconds * 2.5 + s_seed)
            alpha = int(180 * twinkle)
            d.ellipse((sx - s_rad, sy - s_rad, sx + s_rad, sy + s_rad), fill=(210, 235, 255, alpha))

        # ── 1. Top Title & Kicker ──
        font_kicker = _load_font("Mukta-ExtraBold.ttf", max(11, int(15 * rs)))  # Top kicker banner

        kicker_text = "NEWTON'S MOUNTAIN - 1687 THOUGHT EXPERIMENT"
        _draw_shadow_text(d, center_x, 34 * rs, kicker_text, font_kicker,
                          fill=(255, 230, 80), anchor="mm", rs=rs)

        # ── 2. Atmospheric Corona Glow ──
        d.ellipse((center_x - (earth_r + 26 * rs), center_y - (earth_r + 26 * rs),
                   center_x + (earth_r + 26 * rs), center_y + (earth_r + 26 * rs)),
                  fill=(0, 180, 255, 25))
        d.ellipse((center_x - (earth_r + 14 * rs), center_y - (earth_r + 14 * rs),
                   center_x + (earth_r + 14 * rs), center_y + (earth_r + 14 * rs)),
                  fill=(30, 200, 255, 45))
        d.ellipse((center_x - (earth_r + 5 * rs), center_y - (earth_r + 5 * rs),
                   center_x + (earth_r + 5 * rs), center_y + (earth_r + 5 * rs)),
                  fill=(60, 220, 255, 75))

        # ── 3. Central Earth ──
        if self._earth_src is not None:
            scale = (earth_r / 432.0)
            target_dim = int(1144 * scale)
            if self._cached_earth is None or self._cached_earth[0] != target_dim:
                resized = self._earth_src.resize((target_dim, target_dim), Image.Resampling.LANCZOS)
                self._cached_earth = (target_dim, resized)
            else:
                resized = self._cached_earth[1]
            ex = int(center_x - target_dim / 2.0)
            ey = int(center_y - target_dim / 2.0)
            canvas.alpha_composite(resized, (ex, ey))
        else:
            d.ellipse(
                (center_x - earth_r, center_y - earth_r, center_x + earth_r, center_y + earth_r),
                fill=(12, 38, 78, 255), outline=(62, 224, 255, 240), width=int(3 * rs)
            )
            font_earth = _load_font("Mukta-ExtraBold.ttf", max(12, int(26 * rs)))
            _draw_shadow_text(d, center_x, center_y, "EARTH", font_earth, fill=(210, 235, 255), rs=rs)

        # ── 4. Mountain Pinnacle at Top Pole ──
        # Pinnacle geometry: triangular summit rising from Earth curve
        top_apex_x = center_x
        top_apex_y = center_y - r_top
        base_half_w = 26.0 * rs
        base_left = (center_x - base_half_w, center_y - earth_r + 4 * rs)
        base_right = (center_x + base_half_w, center_y - earth_r + 4 * rs)
        apex = (top_apex_x, top_apex_y)

        # Mountain silhouette with glowing crest
        d.polygon([base_left, apex, base_right], fill=(30, 48, 75, 255), outline=(120, 190, 255, 200))
        # Golden Cannon Platform at summit
        cannon_box = (
            top_apex_x - 7 * rs, top_apex_y - 6 * rs,
            top_apex_x + 7 * rs, top_apex_y + 4 * rs
        )
        d.rectangle(cannon_box, fill=(255, 215, 0, 255), outline=(255, 255, 255, 255))
        d.line([(top_apex_x, top_apex_y - 1 * rs), (top_apex_x + 12 * rs, top_apex_y - 1 * rs)],
               fill=(255, 235, 120, 255), width=max(2, int(3 * rs)))

        # Mountain Label
        font_mtn = _load_font("Mukta-ExtraBold.ttf", max(9, int(12 * rs)))
        _draw_shadow_text(d, top_apex_x - 42 * rs, top_apex_y + 4 * rs, "NEWTON'S MOUNTAIN",
                          font_mtn, fill=(255, 235, 140), anchor="rm", rs=rs)

        # ── 5. Trajectory Paths Computation ──
        # Origin of projectile launch:
        p0_x = top_apex_x + 12 * rs
        p0_y = top_apex_y - 1 * rs

        # Trajectory 1: Vertical Drop (x = const, y goes from apex to surface)
        drop_pts = [
            (top_apex_x + 4 * rs, top_apex_y + (i / 30.0) * mtn_h)
            for i in range(31)
        ]

        # Trajectory 2: Short Throw (ballistic arc landing ~45° along Earth surface)
        short_pts: List[Tuple[float, float]] = []
        for i in range(41):
            frac = i / 40.0
            theta = frac * (math.pi * 0.28)  # ~50 degrees
            r = r_top - frac * frac * (mtn_h + 8 * rs)
            # convert polar to cartesian centered at Earth center (launch from top = -pi/2)
            angle_from_top = -math.pi / 2.0 + theta
            short_pts.append((center_x + r * math.cos(angle_from_top),
                              center_y + r * math.sin(angle_from_top)))

        # Trajectory 3: Far Throw (curves over Earth horizon to ~115°)
        far_pts: List[Tuple[float, float]] = []
        for i in range(51):
            frac = i / 50.0
            theta = frac * (math.pi * 0.64)  # ~115 degrees
            r = r_top - frac * frac * (mtn_h + 5 * rs)
            angle_from_top = -math.pi / 2.0 + theta
            far_pts.append((center_x + r * math.cos(angle_from_top),
                            center_y + r * math.sin(angle_from_top)))

        # Trajectory 4: Full Closed Orbit Loop (radius r_top, continuous circle)
        orbit_pts: List[Tuple[float, float]] = [
            (center_x + r_top * math.cos(-math.pi / 2.0 + (i / 64.0) * 2.0 * math.pi),
             center_y + r_top * math.sin(-math.pi / 2.0 + (i / 64.0) * 2.0 * math.pi))
            for i in range(65)
        ]

        # Helper to draw smooth polyline
        def draw_path(pts: List[Tuple[float, float]], color: Tuple[int, int, int, int], width: int = 2):
            for j in range(len(pts) - 1):
                d.line([pts[j], pts[j + 1]], fill=color, width=width)

        # ── 6. Mode-Specific Progressive Animation ──
        loop_period = 2.4
        progress = (t_seconds % loop_period) / loop_period  # 0.0 -> 1.0

        if mode == "drop":
            # Trajectory: Vertical Drop
            draw_path(drop_pts, (255, 80, 80, 120), width=max(1, int(2 * rs)))
            idx = int(progress * (len(drop_pts) - 1))
            bx, by = drop_pts[idx]
            # Falling ball & glow
            d.ellipse((bx - 6 * rs, by - 6 * rs, bx + 6 * rs, by + 6 * rs), fill=(255, 80, 80, 255), outline=(255, 255, 255, 255))
            # Vector: Downward gravity
            _draw_hud_vector(d, bx, by, bx, by + 34 * rs, (255, 82, 82), label="v = 0 (Falls Straight)", rs=rs)
            badge_text = "TRIAL 1: RELEASE FROM REST - FALLS TO BASE"
            badge_color = (255, 100, 100)
            expl_text = "Gravity pulls directly toward Earth's centre."

        elif mode == "throw_short":
            # Ghost drop path
            draw_path(drop_pts, (150, 150, 150, 70), width=1)
            # Active short throw
            draw_path(short_pts, (0, 220, 255, 140), width=max(1, int(2.5 * rs)))
            idx = int(progress * (len(short_pts) - 1))
            bx, by = short_pts[idx]
            d.ellipse((bx - 6 * rs, by - 6 * rs, bx + 6 * rs, by + 6 * rs), fill=(0, 220, 255, 255), outline=(255, 255, 255, 255))
            # Horizontal tangent vector at launch / ball
            _draw_hud_vector(d, bx, by, bx + 28 * rs, by + 12 * rs, (0, 229, 255), label="v = 3.5 km/s", rs=rs)
            badge_text = "TRIAL 2: THROWN HORIZONTALLY - ARCS & HITS"
            badge_color = (0, 220, 255)
            expl_text = "Moves forward, but gravity curves it into the ground."

        elif mode == "throw_far":
            # Ghost paths
            draw_path(drop_pts, (150, 150, 150, 60), width=1)
            draw_path(short_pts, (0, 180, 220, 70), width=1)
            # Active far throw
            draw_path(far_pts, (255, 215, 0, 160), width=max(1, int(2.5 * rs)))
            idx = int(progress * (len(far_pts) - 1))
            bx, by = far_pts[idx]
            d.ellipse((bx - 6 * rs, by - 6 * rs, bx + 6 * rs, by + 6 * rs), fill=(255, 215, 0, 255), outline=(255, 255, 255, 255))
            # Velocity vector
            _draw_hud_vector(d, bx, by, bx + 32 * rs, by + 16 * rs, (255, 215, 0), label="v = 6.5 km/s", rs=rs)
            badge_text = "TRIAL 3: THROWN HARDER - CURVES AROUND EARTH"
            badge_color = (255, 215, 0)
            expl_text = "Earth curves away beneath it — travels much further!"

        else:  # "orbital_loop" / "orbit"
            # Ghost trajectories
            draw_path(drop_pts, (120, 120, 120, 50), width=1)
            draw_path(short_pts, (0, 180, 220, 50), width=1)
            draw_path(far_pts, (255, 215, 0, 60), width=1)
            # Full glowing orbit path
            d.ellipse((center_x - r_top, center_y - r_top, center_x + r_top, center_y + r_top),
                      outline=(0, 229, 255, 90), width=max(1, int(2.5 * rs)))

            # Moving orbiting projectile with glowing comet tail
            orbit_speed = 0.28  # rev / s
            angle = -math.pi / 2.0 + t_seconds * orbit_speed * (2.0 * math.pi)
            bx = center_x + r_top * math.cos(angle)
            by = center_y + r_top * math.sin(angle)

            # Comet tail
            for step in range(24):
                a1 = angle - (step / 24.0) * (math.pi / 3.0)
                a2 = angle - ((step + 1) / 24.0) * (math.pi / 3.0)
                p1 = (center_x + r_top * math.cos(a1), center_y + r_top * math.sin(a1))
                p2 = (center_x + r_top * math.cos(a2), center_y + r_top * math.sin(a2))
                alpha_t = int(200 * (1.0 - step / 24.0))
                d.line([p1, p2], fill=(0, 229, 255, alpha_t), width=max(1, int(3 * rs * (1.0 - step / 24.0))))

            # Projectile bead
            d.ellipse((bx - 7 * rs, by - 7 * rs, bx + 7 * rs, by + 7 * rs),
                      fill=(255, 255, 255, 255), outline=(0, 229, 255, 255), width=int(2 * rs))

            # Forward Velocity Vector (+Tangent)
            v_len = 54 * rs
            vx = bx - v_len * math.sin(angle)
            vy = by + v_len * math.cos(angle)
            v_lx = bx - (v_len + 16 * rs) * math.sin(angle) + 16 * rs * math.cos(angle)
            v_ly = by + (v_len + 16 * rs) * math.cos(angle) + 16 * rs * math.sin(angle)
            _draw_hud_vector(d, bx, by, vx, vy, (0, 229, 255), label="v = 7.9 km/s", rs=rs, label_pos=(v_lx, v_ly))

            # Inward Gravity Vector (-Radius)
            g_len = 42 * rs
            gx = bx - g_len * math.cos(angle)
            gy = by - g_len * math.sin(angle)
            g_lx = bx - (g_len * 0.5) * math.cos(angle) + 36 * rs * math.sin(angle)
            g_ly = by - (g_len * 0.5) * math.sin(angle) - 36 * rs * math.cos(angle)
            _draw_hud_vector(d, bx, by, gx, gy, (255, 82, 82), label="g (Gravity)", rs=rs, label_pos=(g_lx, g_ly))

            badge_text = "ORBITAL INSERTION: 7.9 km/s (28,000 km/h) - NEVER HITS!"
            badge_color = (60, 255, 160)
            expl_text = "Curvature of fall matches curvature of Earth | PERPETUAL ORBIT!"

        # ── 7. Clean Editorial Footer Telemetry ──
        font_h1 = _load_font("Mukta-ExtraBold.ttf", max(12, int(17 * rs)))
        font_h2 = _load_font("Mukta-ExtraBold.ttf", max(10, int(13.5 * rs)))

        _draw_shadow_text(d, center_x, h_px - 44 * rs, badge_text, font_h1, fill=badge_color, anchor="mm", rs=rs)
        _draw_shadow_text(d, center_x, h_px - 20 * rs, expl_text, font_h2, fill=(200, 235, 255), anchor="mm", rs=rs)

        return canvas


__all__ = ["NewtonMountainSimulation"]
