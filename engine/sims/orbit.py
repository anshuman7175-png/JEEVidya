"""
JEEVidya — Orbit Simulation Engine (Visual Bible v1 §6.1)
════════════════════════════════════════════════════════
Broadcast-grade pedagogical physics simulation:
  • Central Body: Photorealistic high-resolution Earth with Rayleigh atmospheric glow,
    realistic oceans, clouds, and continents.
  • Altitude Dimension: Visual bracket showing h = 400 km (Low Earth Orbit).
  • Orbiter: Authentic 3D ISS spacecraft with golden photovoltaic arrays and modules.
  • Dual HUD Telemetry Cards (Frosted Glassmorphism):
      - Left Panel: Earth Gravity Pull (g = 8.7 m/s², 89% active at 400 km).
      - Right Panel: Orbital Speed (v = 7.7 km/s / 27,600 km/h, perpetual curvature).
  • Pill-Backed Vector Arrows: High-contrast, non-colliding vectors for
    tangent velocity (Electric Cyan) and inward gravity (Coral Red).
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from config import settings


def _load_font(font_name: str, size: int) -> ImageFont.FreeTypeFont:
    """Safely load TrueType font with fallbacks."""
    paths = [
        os.path.join(settings.PROJECT_ROOT, "assets", "fonts", font_name),
        os.path.join(settings.PROJECT_ROOT, "assets", "fonts", "Mukta-ExtraBold.ttf"),
        os.path.join(settings.PROJECT_ROOT, "assets", "fonts", "Baloo2-Variable.ttf"),
        f"/System/Library/Fonts/Supplemental/{font_name}",
        f"/Library/Fonts/{font_name}",
        f"/System/Library/Fonts/{font_name}",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf"
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
    """Draw broadcast typography with soft ambient drop shadow for legibility without clunky UI boxes."""
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
    """Draw sleek vector arrow with tapered stroke, glowing arrowhead, and clean drop-shadowed label."""
    # Outer glow
    draw.line([(x1, y1), (x2, y2)], fill=color + (90,), width=width + int(4 * rs))
    # Core arrow
    draw.line([(x1, y1), (x2, y2)], fill=color + (255,), width=width)

    # Arrowhead
    angle = math.atan2(y2 - y1, x2 - x1)
    head_len = 16 * rs
    p1 = (x2 - head_len * math.cos(angle - 0.38), y2 - head_len * math.sin(angle - 0.38))
    p2 = (x2 - head_len * math.cos(angle + 0.38), y2 - head_len * math.sin(angle + 0.38))
    draw.polygon([(x2, y2), p1, p2], fill=color + (255,))

    if label:
        font_main = _load_font("Mukta-ExtraBold.ttf", max(11, int(15 * rs)))
        if label_pos is not None:
            lx, ly = label_pos
        else:
            pill_dist = 22 * rs
            lx = x2 + pill_dist * math.cos(angle)
            ly = y2 + pill_dist * math.sin(angle)
        _draw_shadow_text(draw, lx, ly, label, font_main, fill=color, anchor="mm", rs=rs)


class OrbitSimulation:
    """Broadcast-grade 60fps orbital physics simulation with studio visual assets."""

    def __init__(self, mode: str = "circular", **kwargs: Any):
        self.mode = mode
        self.kwargs = kwargs
        self._earth_src: Optional[Image.Image] = None
        self._iss_src: Optional[Image.Image] = None
        self._moon_src: Optional[Image.Image] = None
        self._cached_earth: Optional[Tuple[int, Image.Image]] = None
        self._cached_iss: Optional[Tuple[int, Image.Image]] = None
        self._cached_moon: Optional[Tuple[int, Image.Image]] = None
        self._load_assets()

    def _load_assets(self) -> None:
        """Load photorealistic Earth, 3D ISS, and Moon assets if available."""
        sims_dir = os.path.join(settings.PROJECT_ROOT, "assets", "sims")
        earth_p = os.path.join(sims_dir, "earth_photoreal.png")
        iss_p = os.path.join(sims_dir, "iss_satellite.png")
        moon_p = os.path.join(sims_dir, "moon.png")

        if os.path.exists(earth_p):
            try:
                self._earth_src = Image.open(earth_p).convert("RGBA")
            except Exception:
                self._earth_src = None

        if os.path.exists(iss_p):
            try:
                self._iss_src = Image.open(iss_p).convert("RGBA")
            except Exception:
                self._iss_src = None

        if os.path.exists(moon_p):
            try:
                self._moon_src = Image.open(moon_p).convert("RGBA")
            except Exception:
                self._moon_src = None

    def render(self, t_seconds: float, params: Dict[str, Any],
               width: int = 1080, height: int = 820, res_scale: float = 1.0) -> Image.Image:
        """
        Render a broadcast-grade educational orbital mechanics frame.
        Dimensions: width x height (default 1080 x 820).
        """
        rs = res_scale
        w_px = int(width)
        h_px = int(height)
        canvas = Image.new("RGBA", (w_px, h_px), (0, 0, 0, 0))
        d = ImageDraw.Draw(canvas)

        mode = params.get("mode", self.mode)
        is_newton = (mode == "newton_curvature")
        is_moon = (mode == "moon_orbit" or params.get("target") == "moon")

        center_x = w_px / 2
        center_y = (h_px / 2) + 80 * rs

        earth_r = 165.0 * rs
        orbit_r = 255.0 * rs

        # Deep space subtle starry background with gentle twinkle
        for sx, sy, s_rad, s_seed in [
            (0.12 * w_px, 0.24 * h_px, 1.8 * rs, 1.1),
            (0.22 * w_px, 0.72 * h_px, 1.2 * rs, 2.3),
            (0.85 * w_px, 0.28 * h_px, 2.0 * rs, 3.5),
            (0.92 * w_px, 0.68 * h_px, 1.4 * rs, 4.2),
            (0.08 * w_px, 0.55 * h_px, 1.0 * rs, 5.7),
            (0.78 * w_px, 0.85 * h_px, 1.5 * rs, 6.1),
            (0.35 * w_px, 0.18 * h_px, 1.2 * rs, 7.8),
            (0.65 * w_px, 0.22 * h_px, 1.6 * rs, 8.4),
        ]:
            twinkle = 0.55 + 0.45 * math.sin(t_seconds * 2.5 + s_seed)
            alpha = int(180 * twinkle)
            d.ellipse((sx - s_rad, sy - s_rad, sx + s_rad, sy + s_rad), fill=(210, 235, 255, alpha))

        # ══════════════════════════════════════════════════════════
        # 1. TOP DOCUMENTARY KICKER (Clean typography, no clunky box)
        # ══════════════════════════════════════════════════════════
        font_tag = _load_font("Mukta-ExtraBold.ttf", max(11, int(15 * rs)))
        if is_moon:
            tag_text = "GRAVITATION - WHY THE MOON DOES NOT CRASH INTO EARTH"
            tag_color = (255, 235, 120)
        elif is_newton:
            tag_text = "NEWTON'S THOUGHT EXPERIMENT - FALLING AROUND EARTH"
            tag_color = (255, 214, 0)
        else:
            tag_text = "GRAVITATION - CLASS 11 NCERT"
            tag_color = (140, 215, 255)
        _draw_shadow_text(d, center_x, 32 * rs, tag_text, font_tag, fill=tag_color, anchor="mm", rs=rs)

        # ══════════════════════════════════════════════════════════
        # 2. EARTH ATMOSPHERIC CORONA GLOW (Limb Light Wrap)
        # ══════════════════════════════════════════════════════════
        d.ellipse((center_x - (earth_r + 28 * rs), center_y - (earth_r + 28 * rs),
                   center_x + (earth_r + 28 * rs), center_y + (earth_r + 28 * rs)),
                  fill=(0, 180, 255, 25))
        d.ellipse((center_x - (earth_r + 16 * rs), center_y - (earth_r + 16 * rs),
                   center_x + (earth_r + 16 * rs), center_y + (earth_r + 16 * rs)),
                  fill=(30, 200, 255, 45))
        d.ellipse((center_x - (earth_r + 6 * rs), center_y - (earth_r + 6 * rs),
                   center_x + (earth_r + 6 * rs), center_y + (earth_r + 6 * rs)),
                  fill=(60, 220, 255, 80))

        # ══════════════════════════════════════════════════════════
        # 3. CENTRAL EARTH (Photoreal Asset or Vector Fallback)
        # ══════════════════════════════════════════════════════════
        if self._earth_src is not None:
            scale = (earth_r / 432.0)
            target_dim = int(1144 * scale)
            if self._cached_earth is None or self._cached_earth[0] != target_dim:
                resized_earth = self._earth_src.resize((target_dim, target_dim), Image.Resampling.LANCZOS)
                self._cached_earth = (target_dim, resized_earth)
            else:
                resized_earth = self._cached_earth[1]

            ex = int(center_x - target_dim / 2)
            ey = int(center_y - target_dim / 2)
            canvas.alpha_composite(resized_earth, (ex, ey))
        else:
            d.ellipse(
                (center_x - earth_r, center_y - earth_r, center_x + earth_r, center_y + earth_r),
                fill=(12, 38, 78, 255), outline=(62, 224, 255, 240), width=int(3 * rs)
            )
            font_earth = _load_font("Mukta-ExtraBold.ttf", max(12, int(26 * rs)))
            _draw_shadow_text(d, center_x, center_y, "EARTH", font_earth, fill=(210, 235, 255), rs=rs)

        # ══════════════════════════════════════════════════════════
        # 4. ORBIT TRAJECTORY (Smooth Ring + Illuminated Velocity Arc)
        # ══════════════════════════════════════════════════════════
        # Subtle full orbital path
        d.ellipse((center_x - orbit_r, center_y - orbit_r, center_x + orbit_r, center_y + orbit_r),
                  outline=(0, 210, 255, 55), width=max(1, int(2 * rs)))

        speed_rad_per_sec = 0.16 if is_moon else 0.20
        angle = t_seconds * speed_rad_per_sec - (math.pi / 4)
        sat_x = center_x + orbit_r * math.cos(angle)
        sat_y = center_y + orbit_r * math.sin(angle)

        # Glowing comet velocity trail behind satellite/moon
        tail_steps = 32
        tail_angle_span = math.pi / 4.0 if is_moon else math.pi / 3.4
        tail_color = (255, 235, 140) if is_moon else (0, 230, 255)
        for s in range(tail_steps):
            a_start = angle - (s / tail_steps) * tail_angle_span
            a_end = angle - ((s + 1) / tail_steps) * tail_angle_span
            p1 = (center_x + orbit_r * math.cos(a_start), center_y + orbit_r * math.sin(a_start))
            p2 = (center_x + orbit_r * math.cos(a_end), center_y + orbit_r * math.sin(a_end))
            t_alpha = int(190 * (1.0 - s / tail_steps))
            t_width = max(1, int(3.5 * rs * (1.0 - s / tail_steps)))
            d.line([p1, p2], fill=tail_color + (t_alpha,), width=t_width)

        # ══════════════════════════════════════════════════════════
        # 5. ORBITER (Photoreal Moon, 3D ISS Spacecraft, or Procedural)
        # ══════════════════════════════════════════════════════════
        if is_moon:
            moon_dim = int(52 * rs)
            if self._moon_src is not None:
                if self._cached_moon is None or self._cached_moon[0] != moon_dim:
                    resized_moon = self._moon_src.resize((moon_dim, moon_dim), Image.Resampling.LANCZOS)
                    self._cached_moon = (moon_dim, resized_moon)
                else:
                    resized_moon = self._cached_moon[1]
                mx = int(sat_x - moon_dim / 2.0)
                my = int(sat_y - moon_dim / 2.0)
                canvas.alpha_composite(resized_moon, (mx, my))
            else:
                d.ellipse((sat_x - 14 * rs, sat_y - 14 * rs, sat_x + 14 * rs, sat_y + 14 * rs),
                          fill=(220, 225, 235, 255), outline=(255, 255, 255, 255), width=int(2 * rs))
        elif self._iss_src is not None:
            iss_w = int(76 * rs)
            iss_h = int(iss_w * self._iss_src.height / self._iss_src.width)
            if self._cached_iss is None or self._cached_iss[0] != iss_w:
                resized_iss = self._iss_src.resize((iss_w, iss_h), Image.Resampling.LANCZOS)
                self._cached_iss = (iss_w, resized_iss)
            else:
                resized_iss = self._cached_iss[1]

            rot_deg = -math.degrees(angle) - 45
            rot_iss = resized_iss.rotate(rot_deg, resample=Image.Resampling.BILINEAR, expand=True)

            ix = int(sat_x - rot_iss.width / 2)
            iy = int(sat_y - rot_iss.height / 2)
            canvas.alpha_composite(rot_iss, (ix, iy))
        else:
            d.ellipse((sat_x - 8 * rs, sat_y - 8 * rs, sat_x + 8 * rs, sat_y + 8 * rs),
                      fill=(255, 255, 255, 255), outline=(0, 225, 255, 255), width=int(2 * rs))

        # ══════════════════════════════════════════════════════════
        # 6. PHYSICAL VECTORS (Bold, Floating Documentary Style)
        # ══════════════════════════════════════════════════════════
        show_vectors = params.get("show_vectors", True)
        gravity_on = params.get("gravity_vector", True)

        if show_vectors:
            # Velocity Vector: Forward along tangent (+T)
            v_len = 64 * rs
            vx = sat_x - v_len * math.sin(angle)
            vy = sat_y + v_len * math.cos(angle)
            if is_moon:
                v_lbl = "Speed: 1.0 km/s (Sideways)"
            elif is_newton:
                v_lbl = "Speed: 7.7 km/s"
            else:
                v_lbl = "v = 7.7 km/s"
            # Forward Quadrant: Placed near forward arrow tip, offset radially outward (+R)
            v_lx = sat_x - (v_len + 18 * rs) * math.sin(angle) + 14 * rs * math.cos(angle)
            v_ly = sat_y + (v_len + 18 * rs) * math.cos(angle) + 14 * rs * math.sin(angle)
            _draw_hud_vector(d, sat_x, sat_y, vx, vy, (0, 229, 255), label=v_lbl,
                             width=int(3 * rs), rs=rs, label_pos=(v_lx, v_ly))

        if gravity_on:
            # Gravity Vector: Directed inwards towards Earth center (-R)
            g_len = 48 * rs
            gx = sat_x - g_len * math.cos(angle)
            gy = sat_y - g_len * math.sin(angle)
            if is_moon:
                g_lbl = "Fall: 1.4 mm/s"
            elif is_newton:
                g_lbl = "Fall: ~5 m/s"
            else:
                g_lbl = "g ≈ 8.7 m/s²"
            # Inward Quadrant: Offset inward (-R) and laterally into trailing side (-T)
            g_lx = sat_x - (g_len * 0.55) * math.cos(angle) + 48 * rs * math.sin(angle)
            g_ly = sat_y - (g_len * 0.55) * math.sin(angle) - 48 * rs * math.cos(angle)
            _draw_hud_vector(d, sat_x, sat_y, gx, gy, (255, 82, 82), label=g_lbl,
                             width=int(3 * rs), rs=rs, label_pos=(g_lx, g_ly))

        # ══════════════════════════════════════════════════════════
        # 7. SATELLITE CALLOUT & NEWTON TAKEAWAY
        # ══════════════════════════════════════════════════════════
        if is_moon:
            lbl = "MOON - 3,84,400 km"
            tag_txt = "MOON"
        else:
            raw_lbl = params.get("label", "ISS - 400 km")
            lbl = "ISS - 400 km" if raw_lbl in ("ISS Spacecraft", "ISS", "ISS · 400 km up", "ISS - 400 km up") else raw_lbl
            tag_txt = "ISS" if is_newton else lbl

        if lbl:
            tag_x = sat_x + 60 * rs * math.sin(angle) + 24 * rs * math.cos(angle)
            tag_y = sat_y - 60 * rs * math.cos(angle) + 24 * rs * math.sin(angle)
            font_sat = _load_font("Mukta-ExtraBold.ttf", max(11, int(15 * rs)))
            _draw_shadow_text(d, tag_x, tag_y, tag_txt, font_sat, fill=(255, 235, 120), anchor="mm", rs=rs)

        if is_moon:
            font_h1 = _load_font("Mukta-ExtraBold.ttf", max(13, int(18 * rs)))
            font_h2 = _load_font("Mukta-ExtraBold.ttf", max(10, int(14 * rs)))
            _draw_shadow_text(d, center_x, h_px - 44 * rs,
                              "MOON FALLS 1.4 mm EVERY SECOND",
                              font_h1, fill=(255, 220, 90), anchor="mm", rs=rs)
            _draw_shadow_text(d, center_x, h_px - 20 * rs,
                              "1 km sideways + 1.4 mm down = keeps missing Earth for 4.5 billion years!",
                              font_h2, fill=(180, 230, 255), anchor="mm", rs=rs)
        elif is_newton:
            # Editorial takeaway in clean typography (no UI box)
            font_h1 = _load_font("Mukta-ExtraBold.ttf", max(13, int(18 * rs)))
            font_h2 = _load_font("Mukta-ExtraBold.ttf", max(10, int(14 * rs)))
            _draw_shadow_text(d, center_x, h_px - 44 * rs,
                              "FALLING AROUND THE EARTH",
                              font_h1, fill=(255, 220, 90), anchor="mm", rs=rs)
            _draw_shadow_text(d, center_x, h_px - 20 * rs,
                              "7.7 km/s forward -> zameen curve ho jaati hai -> you MISS the ground!",
                              font_h2, fill=(180, 230, 255), anchor="mm", rs=rs)

        return canvas
