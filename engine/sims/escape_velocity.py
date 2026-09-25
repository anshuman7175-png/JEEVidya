"""
JEEVidya — Escape Velocity & Spacetime Simulation (Visual Bible v1 §6.4)
═════════════════════════════════════════════════════════════════════════
Simplified, high-impact science animations optimized for YouTube Shorts (1080x1920):
  • Shorts Safe Zones:
      - Top margin safe zone: Y in [0, 160] kept clear of critical text (YouTube search/sound bar).
      - Header title at Y ≈ 175 px (bold, uncluttered).
      - Action zone at Y in [230, 680] px (high-contrast physics visuals).
      - Bottom takeaway at Y ≈ 730 px (single clean punchline, cleared before Y=820 stage line).
      - Zero AI-style boxes or clunky pills.
"""
from __future__ import annotations

import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from config import settings


def _load_font(font_name: str, size: int) -> ImageFont.FreeTypeFont:
    """Load bundled Google Fonts with graceful OS system font fallbacks."""
    paths = [
        os.path.join(settings.PROJECT_ROOT, "assets", "fonts", font_name),
        os.path.join(settings.PROJECT_ROOT, "assets", "fonts", "Mukta-ExtraBold.ttf"),
        os.path.join(settings.PROJECT_ROOT, "assets", "fonts", "Baloo2-Variable.ttf"),
        os.path.join(settings.PROJECT_ROOT, "assets", "fonts", "Inter-Black.ttf"),
        os.path.join(settings.PROJECT_ROOT, "assets", "fonts", "Inter-Bold.ttf"),
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
    """Draw clean floating typography with native anti-aliased dark halo."""
    if not text:
        return
    stroke_w = max(1, int(2.5 * rs))
    draw.text((x, y), text, font=font, fill=fill + (255,), anchor=anchor,
              stroke_width=stroke_w, stroke_fill=(3, 6, 16, 240))


def _draw_hud_vector(draw: ImageDraw.Draw, x1: float, y1: float, x2: float, y2: float,
                     color: Tuple[int, int, int], label: str = "",
                     width: int = 3, rs: float = 1.0,
                     label_pos: Optional[Tuple[float, float]] = None) -> None:
    """Draw sleek HUD vector arrow with glowing trail and clean floating label."""
    draw.line([(x1, y1), (x2, y2)], fill=color + (80,), width=width + int(3 * rs))
    draw.line([(x1, y1), (x2, y2)], fill=color + (255,), width=width)

    angle = math.atan2(y2 - y1, x2 - x1)
    head_len = 12.0 * rs
    p1 = (x2 - head_len * math.cos(angle - 0.35), y2 - head_len * math.sin(angle - 0.35))
    p2 = (x2 - head_len * math.cos(angle + 0.35), y2 - head_len * math.sin(angle + 0.35))
    draw.polygon([(x2, y2), p1, p2], fill=color + (255,))

    if label:
        font_main = _load_font("Mukta-ExtraBold.ttf", max(11, int(15 * rs)))
        if label_pos is not None:
            lx, ly = label_pos
        else:
            dist = 18.0 * rs
            lx = x2 + dist * math.cos(angle)
            ly = y2 + dist * math.sin(angle)
        _draw_shadow_text(draw, lx, ly, label, font_main, fill=color, anchor="mm", rs=rs)


class EscapeVelocitySimulation:
    """
    Simplified, high-impact physics animations for Season 2 Episode 3:
      - 'earth_escape': Earth gravity well, suborbital fallback, and 11.2 km/s escape.
      - 'gravity_well_comparison': Clean 3D spacetime comparison (Earth vs Moon).
      - 'atmospheric_escape': Photorealistic Moon horizon with boiling gas molecules.
    """

    def __init__(self, mode: str = "earth_escape", **kwargs: Any):
        self.mode = mode
        self.kwargs = kwargs

        # Cached raw asset images
        self._earth_raw: Optional[Image.Image] = None
        self._moon_raw: Optional[Image.Image] = None

        # Texture cache: dim -> Image
        self._cached_earth: Optional[Tuple[int, Image.Image]] = None
        self._cached_moon_globe: Optional[Tuple[int, Image.Image]] = None
        self._cached_moon_horizon: Optional[Tuple[int, Image.Image]] = None

        # Persistent particles for Mode 3 (35 clean, uncluttered particles)
        self._particles: List[Dict[str, float]] = []
        self._init_particles()

        self._load_assets()

    def _load_assets(self) -> None:
        """Load photorealistic planetary textures."""
        earth_p = os.path.join(settings.PROJECT_ROOT, "assets", "sims", "earth_photoreal.png")
        if os.path.exists(earth_p):
            try:
                self._earth_raw = Image.open(earth_p).convert("RGBA")
            except Exception:
                self._earth_raw = None

        moon_p = os.path.join(settings.PROJECT_ROOT, "assets", "sims", "moon.png")
        if os.path.exists(moon_p):
            try:
                self._moon_raw = Image.open(moon_p).convert("RGBA")
            except Exception:
                self._moon_raw = None

    def _init_particles(self) -> None:
        """Initialize 35 clean, visible gas molecules for Mode 3."""
        rng = random.Random(42)
        self._particles = []
        for i in range(35):
            # ~20% are high-energy escaping molecules (v > 2.4 km/s)
            is_fast = (i % 5 == 0)
            speed = rng.uniform(2.6, 4.4) if is_fast else rng.uniform(0.7, 2.1)
            self._particles.append({
                "x_norm": rng.uniform(0.12, 0.88),
                "v_kms": speed,
                "vx_norm": rng.uniform(-0.08, 0.08),
                "phase": rng.uniform(0.0, 8.0),
                "lifetime": rng.uniform(2.8, 4.2),
                "size_scale": rng.uniform(0.9, 1.3),
            })

    def render(self, t_seconds: float, params: Dict[str, Any],
               width: int = 1080, height: int = 820, res_scale: float = 1.0) -> Image.Image:
        """Render 60fps simulation canvas with YouTube Shorts safe positioning."""
        rs = res_scale
        w_px = int(width)
        h_px = int(height)
        canvas = Image.new("RGBA", (w_px, h_px), (0, 0, 0, 0))
        d = ImageDraw.Draw(canvas)

        mode = params.get("mode", self.mode).lower().strip()
        if mode in ("escape", "earth", "escape_velocity", "earth_escape"):
            mode = "earth_escape"
        elif mode in ("gravity", "well", "comparison", "earth_vs_moon", "gravity_well_comparison"):
            mode = "gravity_well_comparison"
        elif mode in ("atmosphere", "atmospheric", "moon_atmosphere", "atmospheric_escape"):
            mode = "atmospheric_escape"

        # Subtle deep-space stars in background
        for sx, sy, s_rad, s_seed in [
            (0.12 * w_px, 0.22 * h_px, 1.8 * rs, 1.2),
            (0.85 * w_px, 0.25 * h_px, 2.0 * rs, 3.7),
            (0.90 * w_px, 0.65 * h_px, 1.4 * rs, 4.4),
            (0.08 * w_px, 0.55 * h_px, 1.2 * rs, 5.9),
            (0.35 * w_px, 0.20 * h_px, 1.4 * rs, 7.1),
            (0.70 * w_px, 0.22 * h_px, 1.6 * rs, 8.8),
        ]:
            twinkle = 0.6 + 0.4 * math.sin(t_seconds * 2.5 + s_seed)
            d.ellipse((sx - s_rad, sy - s_rad, sx + s_rad, sy + s_rad),
                      fill=(220, 240, 255, int(170 * twinkle)))

        if mode == "atmospheric_escape":
            return self._render_atmospheric_escape(canvas, d, t_seconds, w_px, h_px, rs)
        elif mode == "gravity_well_comparison":
            return self._render_gravity_well_comparison(canvas, d, t_seconds, w_px, h_px, rs)
        else:
            return self._render_earth_escape(canvas, d, t_seconds, w_px, h_px, rs)

    # ══════════════════════════════════════════════════════════════════════════
    # MODE 1: EARTH ESCAPE (Simple, Punchy, Shorts-Positioned)
    # ══════════════════════════════════════════════════════════════════════════
    def _render_earth_escape(self, canvas: Image.Image, d: ImageDraw.Draw,
                             t_seconds: float, w_px: int, h_px: int, rs: float) -> Image.Image:
        center_x = w_px / 2.0

        # ── YouTube Shorts Top Safe Title (Y ≈ 175 px) ──
        font_title = _load_font("Mukta-ExtraBold.ttf", max(16, int(24 * rs)))
        _draw_shadow_text(d, center_x, 175 * rs,
                          "ESCAPE VELOCITY: 11.2 km/s",
                          font_title, fill=(255, 220, 80), anchor="mm", rs=rs)

        # Earth centered comfortably in lower-middle sweet spot
        earth_cx = center_x - 140 * rs
        earth_cy = 490.0 * rs
        earth_r = 145.0 * rs

        # ── 1. Volumetric Gravitational Potential Well Corona ──
        d.ellipse((earth_cx - (earth_r + 32 * rs), earth_cy - (earth_r + 32 * rs),
                   earth_cx + (earth_r + 32 * rs), earth_cy + (earth_r + 32 * rs)),
                  fill=(60, 160, 255, 25))
        d.ellipse((earth_cx - (earth_r + 16 * rs), earth_cy - (earth_r + 16 * rs),
                   earth_cx + (earth_r + 16 * rs), earth_cy + (earth_r + 16 * rs)),
                  fill=(100, 200, 255, 50))

        # Equipotential boundary ring
        pulse = 0.5 + 0.5 * math.sin(t_seconds * 2.0)
        pot_r = earth_r * 1.55
        d.ellipse((earth_cx - pot_r, earth_cy - pot_r, earth_cx + pot_r, earth_cy + pot_r),
                  outline=(0, 210, 255, int(35 + 20 * pulse)), width=max(1, int(1.5 * rs)))

        # ── 2. Central Photorealistic Earth ──
        if self._earth_raw is not None:
            target_dim = int(earth_r * 2.55)
            if self._cached_earth is None or self._cached_earth[0] != target_dim:
                resized = self._earth_raw.resize((target_dim, target_dim), Image.Resampling.LANCZOS)
                self._cached_earth = (target_dim, resized)
            else:
                resized = self._cached_earth[1]
            canvas.alpha_composite(resized, (int(earth_cx - target_dim / 2), int(earth_cy - target_dim / 2)))
        else:
            d.ellipse((earth_cx - earth_r, earth_cy - earth_r, earth_cx + earth_r, earth_cy + earth_r),
                      fill=(14, 45, 95, 255), outline=(0, 210, 255, 200), width=2)

        # ── 3. Simple & Punchy Trajectory Comparison ──
        # Path 1: Suborbital Fallback (7.0 km/s) — Red dashed arc
        sub_pts = []
        for i in range(32):
            t_frac = i / 31.0
            th = -math.pi * 0.42 + t_frac * 0.70
            r_arc = (earth_r + 42 * rs) - 4.0 * ((t_frac - 0.5) ** 2) * (36 * rs)
            sub_pts.append((earth_cx + r_arc * math.cos(th), earth_cy + r_arc * math.sin(th)))
        for j in range(len(sub_pts) - 1):
            if j % 2 == 0:
                d.line([sub_pts[j], sub_pts[j + 1]], fill=(255, 80, 80, 200), width=max(1, int(2 * rs)))
        font_lbl = _load_font("Mukta-ExtraBold.ttf", max(11, int(15 * rs)))
        _draw_shadow_text(d, earth_cx + 40 * rs, earth_cy - earth_r - 36 * rs,
                          "7 km/s (Falls Back)", font_lbl, (255, 100, 100), anchor="mm", rs=rs)

        # Path 2: Closed Orbit (7.9 km/s) — Cyan loop
        orb_r = earth_r + 50 * rs
        d.ellipse((earth_cx - orb_r, earth_cy - orb_r, earth_cx + orb_r, earth_cy + orb_r),
                  outline=(0, 229, 255, 75), width=max(1, int(1.5 * rs)))
        _draw_shadow_text(d, earth_cx - orb_r - 20 * rs, earth_cy - 10 * rs,
                          "7.9 km/s (Orbit)", font_lbl, (0, 229, 255), anchor="mm", rs=rs)

        # Path 3: Hyperbolic Escape Trajectory (11.2 km/s) — Golden Soaring Arc
        escape_pts = []
        for i in range(65):
            t_frac = i / 64.0
            t_rad = t_frac * 1.85
            x_pos = earth_cx + (earth_r + 15 * rs) * math.cosh(t_rad * 0.52) * 0.65 + t_frac * 360 * rs
            y_pos = earth_cy - (earth_r + 15 * rs) * math.sinh(t_rad * 0.52) * 0.65 - t_frac * 290 * rs
            escape_pts.append((x_pos, y_pos))

        for j in range(len(escape_pts) - 1):
            d.line([escape_pts[j], escape_pts[j + 1]], fill=(255, 215, 70, 210), width=max(2, int(3 * rs)))

        # ── 4. Rocket Vehicle with Blazing Fire Exhaust ──
        loop_time = 3.0
        p_flight = (t_seconds % loop_time) / loop_time
        curr_idx = min(len(escape_pts) - 1, int(p_flight * (len(escape_pts) - 1)))
        curr_x, curr_y = escape_pts[curr_idx]

        if curr_idx < len(escape_pts) - 1:
            nxt_x, nxt_y = escape_pts[curr_idx + 1]
            angle_v = math.atan2(nxt_y - curr_y, nxt_x - curr_x)
        else:
            angle_v = -0.65

        flame_angle = angle_v + math.pi
        # White plasma nozzle
        d.ellipse((curr_x - 4 * rs, curr_y - 4 * rs, curr_x + 4 * rs, curr_y + 4 * rs),
                  fill=(255, 255, 255, 255))
        # Fire particle plume
        for p_idx in range(12):
            dist_p = (p_idx + 1) * 3.8 * rs
            spread = (random.Random(p_idx + int(t_seconds * 30)).uniform(-0.2, 0.2))
            fx = curr_x + dist_p * math.cos(flame_angle + spread)
            fy = curr_y + dist_p * math.sin(flame_angle + spread)
            rad_p = max(1.5, (4.5 - p_idx * 0.25) * rs)
            color_p = (255, 240, 160, 230) if p_idx < 4 else ((255, 150, 30, 190) if p_idx < 8 else (255, 60, 10, 110))
            d.ellipse((fx - rad_p, fy - rad_p, fx + rad_p, fy + rad_p), fill=color_p)

        # Directional rocket probe body
        p_len = 15.0 * rs
        p_w = 5.5 * rs
        nose = (curr_x + p_len * math.cos(angle_v), curr_y + p_len * math.sin(angle_v))
        tail1 = (curr_x + p_w * math.cos(angle_v + 2.2), curr_y + p_w * math.sin(angle_v + 2.2))
        tail2 = (curr_x + p_w * math.cos(angle_v - 2.2), curr_y + p_w * math.sin(angle_v - 2.2))
        d.polygon([nose, tail1, tail2], fill=(245, 250, 255, 255), outline=(0, 229, 255, 255))

        # Escape HUD vector
        v_hud_len = 50 * rs
        vx_end = curr_x + v_hud_len * math.cos(angle_v)
        vy_end = curr_y + v_hud_len * math.sin(angle_v)
        _draw_hud_vector(d, curr_x, curr_y, vx_end, vy_end, (255, 215, 0),
                         label="11.2 km/s (Escapes!)", width=int(3 * rs), rs=rs)

        # ── 5. Single Punchy Bottom Takeaway (Y ≈ 730 px) ──
        font_punch = _load_font("Mukta-ExtraBold.ttf", max(13, int(18 * rs)))
        _draw_shadow_text(d, center_x, 730 * rs,
                          "At 11.2 km/s, gravity can NEVER pull it back!",
                          font_punch, fill=(60, 255, 160), anchor="mm", rs=rs)

        return canvas

    # ══════════════════════════════════════════════════════════════════════════
    # MODE 2: GRAVITY WELL COMPARISON (Simple, High-Contrast 3D Funnels)
    # ══════════════════════════════════════════════════════════════════════════
    def _render_gravity_well_comparison(self, canvas: Image.Image, d: ImageDraw.Draw,
                                        t_seconds: float, w_px: int, h_px: int, rs: float) -> Image.Image:
        center_x = w_px / 2.0
        col_w = w_px / 2.0

        # ── YouTube Shorts Top Safe Title (Y ≈ 175 px) ──
        font_title = _load_font("Mukta-ExtraBold.ttf", max(16, int(24 * rs)))
        _draw_shadow_text(d, center_x, 175 * rs,
                          "EARTH vs MOON: GRAVITY WELLS",
                          font_title, fill=(255, 220, 80), anchor="mm", rs=rs)

        # ── 3D Spacetime Funnel Mesh ──
        left_cx = col_w * 0.50
        right_cx = col_w * 1.50
        base_cy = 460.0 * rs

        def draw_funnel(origin_x: float, origin_y: float, depth: float,
                        accent_color: Tuple[int, int, int]):
            # 5 clean concentric depth rings
            for r_idx in range(1, 6):
                r_val = r_idx * 30.0 * rs
                z_dip = depth / (1.0 + (r_val / (40.0 * rs)) ** 1.8)
                ring_cy = origin_y + z_dip
                rx = r_val * 1.15
                ry = r_val * 0.40
                alpha = int(45 + 30 * (r_idx / 5.0))
                d.ellipse((origin_x - rx, ring_cy - ry, origin_x + rx, ring_cy + ry),
                          outline=accent_color + (alpha,), width=max(1, int(1.5 * rs)))

            # 8 clean radial warped lines
            for angle_deg in range(0, 360, 45):
                rad = math.radians(angle_deg)
                pts = []
                for step in range(10):
                    frac = step / 9.0
                    r_curr = (20.0 + frac * 125.0) * rs
                    z_curr = depth / (1.0 + (r_curr / (40.0 * rs)) ** 1.8)
                    px = origin_x + r_curr * math.cos(rad) * 1.15
                    py = origin_y + r_curr * math.sin(rad) * 0.40 + z_curr
                    pts.append((px, py))
                for k in range(len(pts) - 1):
                    d.line([pts[k], pts[k + 1]], fill=accent_color + (50,), width=1)

        # Left: Earth Deep Spacetime Funnel
        draw_funnel(left_cx, base_cy, depth=135 * rs, accent_color=(0, 210, 255))

        # Right: Moon Shallow Spacetime Funnel
        draw_funnel(right_cx, base_cy, depth=32 * rs, accent_color=(255, 215, 80))

        # ── Place Photorealistic Spheres ──
        earth_r = 70.0 * rs
        earth_cy = base_cy + 55 * rs
        if self._earth_raw is not None:
            target_dim = int(earth_r * 2.3)
            if self._cached_earth is None or self._cached_earth[0] != target_dim:
                resized = self._earth_raw.resize((target_dim, target_dim), Image.Resampling.LANCZOS)
                self._cached_earth = (target_dim, resized)
            else:
                resized = self._cached_earth[1]
            canvas.alpha_composite(resized, (int(left_cx - target_dim / 2), int(earth_cy - target_dim / 2)))

        moon_r = 45.0 * rs
        moon_cy = base_cy + 15 * rs
        if self._moon_raw is not None:
            target_dim = int(moon_r * 2.2)
            if self._cached_moon_globe is None or self._cached_moon_globe[0] != target_dim:
                resized_m = self._moon_raw.resize((target_dim, target_dim), Image.Resampling.LANCZOS)
                self._cached_moon_globe = (target_dim, resized_m)
            else:
                resized_m = self._cached_moon_globe[1]
            canvas.alpha_composite(resized_m, (int(right_cx - target_dim / 2), int(moon_cy - target_dim / 2)))

        # ── Bold Floating Labels (Clean & Readable) ──
        font_body = _load_font("Mukta-ExtraBold.ttf", max(15, int(22 * rs)))
        font_num = _load_font("Inter-Black.ttf", max(18, int(28 * rs)))
        font_sub = _load_font("Inter-Bold.ttf", max(11, int(14 * rs)))

        # Earth column (Left)
        _draw_shadow_text(d, left_cx, 235 * rs, "EARTH", font_body, (140, 220, 255), anchor="mm", rs=rs)
        _draw_shadow_text(d, left_cx, 275 * rs, "11.2 km/s", font_num, (255, 215, 0), anchor="mm", rs=rs)
        _draw_shadow_text(d, left_cx, 308 * rs, "DEEP GRAVITY WELL", font_sub, (180, 205, 230), anchor="mm", rs=rs)

        # Moon column (Right)
        _draw_shadow_text(d, right_cx, 235 * rs, "MOON", font_body, (255, 235, 140), anchor="mm", rs=rs)
        _draw_shadow_text(d, right_cx, 275 * rs, "2.4 km/s", font_num, (0, 229, 255), anchor="mm", rs=rs)
        _draw_shadow_text(d, right_cx, 308 * rs, "4.7x WEAKER WELL", font_sub, (180, 205, 230), anchor="mm", rs=rs)

        # Arrows climbing the funnels
        _draw_hud_vector(d, left_cx, earth_cy - earth_r, left_cx, earth_cy - earth_r - 95 * rs,
                         (255, 215, 0), label="Requires 11.2 km/s", width=int(3 * rs), rs=rs)
        _draw_hud_vector(d, right_cx, moon_cy - moon_r, right_cx, moon_cy - moon_r - 60 * rs,
                         (0, 229, 255), label="Only 2.4 km/s!", width=int(3 * rs), rs=rs)

        # ── Single Punchy Bottom Takeaway (Y ≈ 730 px) ──
        font_punch = _load_font("Mukta-ExtraBold.ttf", max(13, int(18 * rs)))
        _draw_shadow_text(d, center_x, 730 * rs,
                          "Moon's gravity is 4.7x weaker — gas easily boils off!",
                          font_punch, fill=(255, 215, 90), anchor="mm", rs=rs)

        return canvas

    # ══════════════════════════════════════════════════════════════════════════
    # MODE 3: ATMOSPHERIC ESCAPE (Clean, High-Impact Gas Kinetics)
    # ══════════════════════════════════════════════════════════════════════════
    def _render_atmospheric_escape(self, canvas: Image.Image, d: ImageDraw.Draw,
                                   t_seconds: float, w_px: int, h_px: int, rs: float) -> Image.Image:
        center_x = w_px / 2.0

        # ── YouTube Shorts Top Safe Title (Y ≈ 175 px) ──
        font_title = _load_font("Mukta-ExtraBold.ttf", max(16, int(24 * rs)))
        _draw_shadow_text(d, center_x, 175 * rs,
                          "WHY THE MOON HAS NO AIR",
                          font_title, fill=(255, 220, 80), anchor="mm", rs=rs)

        # ── 1. Photorealistic NASA Lunar Horizon ──
        apex_y = 560.0 * rs
        moon_radius = 880.0 * rs
        moon_cx = center_x
        moon_cy = apex_y + moon_radius
        target_moon_dim = int(moon_radius * 2.0)

        if self._moon_raw is not None:
            if self._cached_moon_horizon is None or self._cached_moon_horizon[0] != target_moon_dim:
                resized_horizon = self._moon_raw.resize((target_moon_dim, target_moon_dim), Image.Resampling.LANCZOS)
                self._cached_moon_horizon = (target_moon_dim, resized_horizon)
            else:
                resized_horizon = self._cached_moon_horizon[1]
            canvas.alpha_composite(resized_horizon, (int(moon_cx - target_moon_dim / 2), int(moon_cy - target_moon_dim / 2)))

        # Soft blue limb highlight along the lunar horizon
        d.arc((moon_cx - (moon_radius + 4 * rs), moon_cy - (moon_radius + 4 * rs),
               moon_cx + (moon_radius + 4 * rs), moon_cy + (moon_radius + 4 * rs)),
              start=200, end=340, fill=(180, 230, 255, 120), width=int(2 * rs))

        # ── 2. Clean Escape Threshold Line (Y = 290 px) ──
        thresh_y = 290.0 * rs
        for dash_x in range(60, w_px - 60, int(30 * rs)):
            d.line([(dash_x, thresh_y), (dash_x + 16 * rs, thresh_y)],
                   fill=(255, 75, 75, 200), width=max(1, int(2 * rs)))

        font_thresh = _load_font("Mukta-ExtraBold.ttf", max(12, int(16 * rs)))
        _draw_shadow_text(d, w_px - 170 * rs, thresh_y - 14 * rs,
                          "ESCAPE SPEED: 2.4 km/s",
                          font_thresh, fill=(255, 95, 95), anchor="mm", rs=rs)

        # Earth contrast note (floating, clean, Y = 235 px)
        font_note = _load_font("Mukta-ExtraBold.ttf", max(11, int(15 * rs)))
        _draw_shadow_text(d, 180 * rs, 235 * rs,
                          "Earth: 11.2 km/s (Holds Air)",
                          font_note, fill=(60, 255, 160), anchor="mm", rs=rs)

        # ── 3. 35 Clean, Legible Gas Molecules ──
        g_accel = 120.0 * rs

        for p in self._particles:
            cycle_time = (t_seconds + p["phase"]) % p["lifetime"]
            frac_t = cycle_time / p["lifetime"]

            px = w_px * (p["x_norm"] + p["vx_norm"] * frac_t)
            dx = px - moon_cx
            if abs(dx) < moon_radius:
                y_surf = moon_cy - math.sqrt(moon_radius * moon_radius - dx * dx)
            else:
                y_surf = apex_y + 80 * rs

            v_kms = p["v_kms"]
            is_escaping = (v_kms >= 2.4)

            if is_escaping:
                # Fast escaping molecule (crimson/red with trail)
                v_pixel_speed = v_kms * 105.0 * rs
                py = y_surf - (v_pixel_speed * frac_t - 0.5 * g_accel * 0.25 * (frac_t ** 2))
                # Ionization streak trail
                d.line([(px, py), (px - p["vx_norm"] * 80 * rs, py + 18 * rs)],
                       fill=(255, 75, 75, 160), width=max(1, int(2 * rs)))
                p_col = (255, 75, 75)
                p_rad = 4.2 * rs * p["size_scale"]
            elif v_kms >= 1.4:
                # Medium molecule (amber): ballistic arc
                v_pixel_speed = v_kms * 85.0 * rs
                h_peak = (v_pixel_speed ** 2) / (2.0 * g_accel)
                tau = (frac_t * 2.0) - 1.0
                py = max(thresh_y + 12 * rs, y_surf - h_peak * (1.0 - tau * tau))
                p_col = (255, 210, 75)
                p_rad = 3.6 * rs * p["size_scale"]
            else:
                # Cold molecule (cyan): hopping near surface
                v_pixel_speed = v_kms * 60.0 * rs
                tau = (frac_t * 2.0) - 1.0
                h_peak = max(20 * rs, (v_pixel_speed ** 2) / (2.0 * g_accel))
                py = y_surf - h_peak * (1.0 - tau * tau)
                p_col = (0, 229, 255)
                p_rad = 3.0 * rs * p["size_scale"]

            # Glowing particle
            d.ellipse((px - p_rad * 1.5, py - p_rad * 1.5, px + p_rad * 1.5, py + p_rad * 1.5),
                      fill=p_col + (65,))
            d.ellipse((px - p_rad, py - p_rad, px + p_rad, py + p_rad),
                      fill=p_col + (240,), outline=(255, 255, 255, 230), width=1)

        # Regolith indicator
        font_reg = _load_font("Inter-Bold.ttf", max(10, int(13 * rs)))
        _draw_shadow_text(d, center_x, apex_y + 55 * rs,
                          "HOT LUNAR DAY: ~100°C",
                          font_reg, fill=(200, 215, 235), anchor="mm", rs=rs)

        # ── 4. Single Punchy Bottom Takeaway (Y ≈ 730 px) ──
        font_punch = _load_font("Mukta-ExtraBold.ttf", max(13, int(18 * rs)))
        _draw_shadow_text(d, center_x, 730 * rs,
                          "Thermal speed exceeds 2.4 km/s -> Atmosphere boiled off!",
                          font_punch, fill=(255, 110, 100), anchor="mm", rs=rs)

        return canvas


__all__ = ["EscapeVelocitySimulation"]
