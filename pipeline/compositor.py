"""
Gudiya & Chintu — Cinematic Compositor V2
The heart of the factory. Renders every frame of the video:
  - Animated gradient background + particles
  - Character compositing with expression swap + body animation
  - Camera shot management (6 shot types with smooth transitions)
  - Explanation overlays (chalkboard + diagrams)
  - Builds the final MP4 with layered audio
"""
import math
import os
import time as time_module
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import settings, brand
from engine.backgrounds import render_gradient_background, render_chalkboard_overlay
from engine.particles import ParticleSystem
from engine.glow import apply_glow, create_text_with_glow
from engine.transitions import scale_pop, fade_cut
from pipeline.lipsync import (
    ExpressionLibrary, analyze_audio, select_expression,
    compute_body_animation,
)
from pipeline.camera import CameraSystem


class CinematicCompositor:
    """
    Renders every frame of a Gudiya & Chintu video.
    Takes voice data + dialogue script → outputs PNG frame sequence.
    """

    def __init__(self):
        self.width = settings.WIDTH
        self.height = settings.HEIGHT
        self.fps = settings.FPS
        self.particles = ParticleSystem(self.width, self.height)
        self.camera = CameraSystem()
        self.chalkboard_bg = render_chalkboard_overlay(self.width, self.height)

        # Load character expression libraries
        self.gudiya = ExpressionLibrary("gudiya")
        self.chintu = ExpressionLibrary("chintu")

        # Font cache
        self._fonts: Dict[str, ImageFont.FreeTypeFont] = {}

    def _get_font(self, name: str, size: int) -> ImageFont.FreeTypeFont:
        """Get or create a cached font."""
        key = f"{name}_{size}"
        if key not in self._fonts:
            try:
                self._fonts[key] = ImageFont.truetype(name, size)
            except (IOError, OSError):
                try:
                    self._fonts[key] = ImageFont.truetype(brand.FONT_FALLBACK, size)
                except (IOError, OSError):
                    self._fonts[key] = ImageFont.load_default()
        return self._fonts[key]

    def render_all_frames(self, turn_data: List[Dict[str, Any]],
                          output_dir: str) -> int:
        """
        Render every frame for the entire video.
        
        Args:
            turn_data: List of processed turn dicts from VoiceEngine
            output_dir: Directory to write frame PNGs
            
        Returns:
            Total number of frames rendered
        """
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        # Pre-analyze audio for all turns
        turn_audio_data = {}
        for turn in turn_data:
            if turn.get("audio") and os.path.exists(turn["audio"]):
                turn_audio_data[turn["turn_id"]] = analyze_audio(turn["audio"], self.fps)

        # Calculate total frames
        total_frames = 0
        turn_frame_ranges = []
        for turn in turn_data:
            duration_ms = turn["duration_ms"]
            num_frames = int((duration_ms / 1000.0) * self.fps)
            turn_frame_ranges.append({
                "turn": turn,
                "start_frame": total_frames,
                "end_frame": total_frames + num_frames,
                "num_frames": num_frames,
            })
            total_frames += num_frames

        print(f"  [Compositor] Rendering {total_frames} frames ({total_frames / self.fps:.1f}s)...")
        render_start = time_module.time()

        # Render loop
        global_frame = 0
        prev_expression_g = "neutral"
        prev_expression_c = "neutral"

        for turn_range in turn_frame_ranges:
            turn = turn_range["turn"]
            speaker = turn["speaker"]
            emotion = turn.get("emotion", "neutral")
            shot_type = turn.get("shot_type", "two_shot")
            turn_id = turn["turn_id"]
            audio_frames = turn_audio_data.get(turn_id, [])

            # Set camera shot for this turn
            self.camera.cut_to(shot_type)

            for local_frame in range(turn_range["num_frames"]):
                # Get audio data for this frame (if available)
                frame_audio = {}
                if local_frame < len(audio_frames):
                    frame_audio = audio_frames[local_frame]
                else:
                    frame_audio = {"db": -80, "mouth_state": 0, "is_speaking": False}

                # Render the frame
                frame_img = self._render_single_frame(
                    global_frame=global_frame,
                    local_frame=local_frame,
                    speaker=speaker,
                    emotion=emotion,
                    frame_audio=frame_audio,
                    turn=turn,
                    prev_expr_g=prev_expression_g,
                    prev_expr_c=prev_expression_c,
                )

                # Save frame
                frame_path = os.path.join(frames_dir, f"frame_{global_frame:05d}.png")
                frame_img.save(frame_path, "PNG")

                # Update expression tracking
                if speaker == "girl":
                    prev_expression_g = select_expression(True, frame_audio, emotion)
                    prev_expression_c = select_expression(False, frame_audio, emotion)
                elif speaker == "boy":
                    prev_expression_c = select_expression(True, frame_audio, emotion)
                    prev_expression_g = select_expression(False, frame_audio, emotion)

                # Camera update
                self.camera.update()
                global_frame += 1

                # Progress logging every 100 frames
                if global_frame % 100 == 0:
                    elapsed = time_module.time() - render_start
                    fps_rate = global_frame / max(0.1, elapsed)
                    eta = (total_frames - global_frame) / max(0.1, fps_rate)
                    print(f"  [Compositor] Frame {global_frame}/{total_frames} "
                          f"({fps_rate:.1f} fps, ETA: {eta:.0f}s)")

        elapsed = time_module.time() - render_start
        print(f"  [Compositor] ✓ Rendered {total_frames} frames in {elapsed:.1f}s "
              f"({total_frames / max(0.1, elapsed):.1f} fps)")
        return total_frames

    def _render_single_frame(self, global_frame: int, local_frame: int,
                              speaker: str, emotion: str,
                              frame_audio: Dict,
                              turn: Dict[str, Any],
                              prev_expr_g: str, prev_expr_c: str) -> Image.Image:
        """Render a single composite frame."""
        # Layer 1: Background (animated gradient + particles)
        if speaker == "explanation":
            bg = self.chalkboard_bg.copy()
        else:
            bg = render_gradient_background(global_frame, self.width, self.height)

        # Add particles (even on chalkboard, but fewer)
        self.particles.update()
        frame = self.particles.render(bg)

        # Layer 2: Characters
        if speaker == "explanation":
            frame = self._render_explanation_frame(frame, turn, local_frame, global_frame)
        else:
            frame = self._render_character_frame(
                frame, global_frame, speaker, emotion,
                frame_audio, prev_expr_g, prev_expr_c
            )

        # Layer 3: Topic banner (top)
        # Topic banner text is intentionally not rendered per-frame — only
        # the scene indicator appears here (see _render_scene_indicator).

        # Layer 4: Caption text (bottom)
        if speaker != "explanation" and frame_audio.get("is_speaking", False):
            text = turn.get("text", "")
            if text:
                frame = self._render_caption(frame, text, global_frame)

        # Convert to RGB for final output
        if frame.mode == 'RGBA':
            rgb_frame = Image.new('RGB', frame.size, brand.BG_TOP)
            rgb_frame.paste(frame, mask=frame.split()[3])
            return rgb_frame
        return frame

    def _render_character_frame(self, frame: Image.Image, global_frame: int,
                                 speaker: str, emotion: str,
                                 frame_audio: Dict,
                                 prev_expr_g: str, prev_expr_c: str) -> Image.Image:
        """Composite both characters onto the frame."""
        gudiya_params, chintu_params = self.camera.get_both_characters(speaker)

        # Determine expressions
        is_gudiya_speaking = speaker == "girl"
        is_chintu_speaking = speaker == "boy"

        expr_g = select_expression(is_gudiya_speaking, frame_audio, emotion)
        expr_c = select_expression(is_chintu_speaking, frame_audio, emotion)

        # Compute body animation
        g_x, g_y, g_body_scale = compute_body_animation(
            global_frame, is_gudiya_speaking,
            gudiya_params["x"], gudiya_params["y"]
        )
        c_x, c_y, c_body_scale = compute_body_animation(
            global_frame, is_chintu_speaking,
            chintu_params["x"], chintu_params["y"]
        )

        # Get expression images
        # Cross-dissolve if expression is changing
        gudiya_img = self.gudiya.get(expr_g)
        chintu_img = self.chintu.get(expr_c)

        # Composite Gudiya
        if gudiya_img and gudiya_params["opacity"] > 0.01 and gudiya_params["scale"] > 0.01:
            frame = self._paste_character(
                frame, gudiya_img,
                int(g_x), int(g_y),
                gudiya_params["scale"] * g_body_scale,
                gudiya_params["opacity"]
            )

        # Composite Chintu
        if chintu_img and chintu_params["opacity"] > 0.01 and chintu_params["scale"] > 0.01:
            frame = self._paste_character(
                frame, chintu_img,
                int(c_x), int(c_y),
                chintu_params["scale"] * c_body_scale,
                chintu_params["opacity"]
            )

        return frame

    # ─── Safe framing (single source of truth) ─────────────
    # A character's HEAD must never leave the frame. The head lives at
    # the TOP of a bottom-anchored sprite, so:
    #   • vertical:   guarantee headroom — if the top would rise above
    #     the margin, slide the sprite down (the body extending past the
    #     bottom edge is natural close-up framing; a cropped skull is not)
    #   • horizontal: keep the character's VISIBLE body on-screen with a
    #     small slack so intended flanking compositions survive, gross
    #     cutoffs don't. The clamp is reach-aware: sprite canvases carry
    #     wide transparent margins (the girl's canvas is square — 1.0·H
    #     wide — while her opaque body is only 0.485·H), so clamping on
    #     the raw canvas half-width would drag every flanking preset toward
    #     center. Instead the half-extent is the WORST-CASE measured arm
    #     reach — a pose-independent constant, so the clamp never jitters
    #     with pose changes and only engages near the edge / during zooms.
    HEADROOM_FRAC = 0.02      # minimum clear space above the head
    EDGE_SLACK_FRAC = 0.06    # allowed horizontal overhang per side (~65 px)
    REACH_FRAC = 0.31         # worst-case opaque half-extent, in units of
                              # target_h (measured: girl 0.310 L / 0.297 R,
                              # boy 0.281 L/R — tools/pose_envelope)
    CHAR_H_FRAC = 0.55        # canvas-height fraction at scale=1.0
    CHAR_H_CEILING = 0.92     # hard size ceiling: never taller than this

    def _char_target_h(self, scale: float, extra: float = 1.0) -> int:
        return min(int(self.height * self.CHAR_H_FRAC * scale * extra),
                   int(self.height * self.CHAR_H_CEILING))

    def _safe_anchor(self, x: float, y: float, target_w: int,
                     target_h: int) -> Tuple[float, float]:
        """Clamp a bottom-center anchor so the sprite is naturally framed.

        Returns FLOATS: callers on the sub-pixel raster (paste_subpixel)
        must not see integer truncation here — that is exactly the motion
        judder the continuous placement path exists to avoid. Integer
        paste paths cast at the paste site.

        Preserves deliberate flanking layouts (e.g. x=250 left, x=775
        right) because the horizontal bound uses the visible reach
        envelope, never the transparent canvas width."""
        headroom = int(self.height * self.HEADROOM_FRAC)
        # Vertical: top = y - target_h must stay below the headroom line.
        # Sliding down is the natural fix — a body past the bottom edge is
        # close-up framing, a cropped skull is not.
        y = max(float(y), float(headroom + target_h))
        # Horizontal: keep the visible body within [-slack, W + slack].
        # Half-extent can never exceed the canvas half-width (tiny sprites,
        # narrow canvases), and never the worst-case reach (wide canvases).
        slack = int(self.width * self.EDGE_SLACK_FRAC)
        half = min(target_w / 2.0, self.REACH_FRAC * target_h)
        min_x = half - slack
        max_x = self.width + slack - half
        if min_x > max_x:                 # sprite wider than frame + slack
            x = self.width / 2.0
        else:
            x = max(min_x, min(float(x), max_x))
        return x, y

    def _paste_character(self, frame: Image.Image, char_img: Image.Image,
                          x: int, y: int, scale: float, opacity: float) -> Image.Image:
        """Paste a character (bottom-center anchored at x, y) with scale,
        opacity, and safe framing — the head can never be cut off."""
        if scale <= 0.01 or opacity <= 0.01:
            return frame

        orig_w, orig_h = char_img.size
        target_h = self._char_target_h(scale)
        target_w = int(target_h * orig_w / orig_h)
        if target_w <= 0 or target_h <= 0:
            return frame

        scaled = char_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

        # Apply opacity
        if opacity < 1.0 and scaled.mode == 'RGBA':
            r, g, b, a = scaled.split()
            a = a.point(lambda p: int(p * opacity))
            scaled = Image.merge('RGBA', (r, g, b, a))

        # Safe-framed bottom-center anchor → paste corner
        ax, ay = self._safe_anchor(x, y, target_w, target_h)
        paste_x = int(round(ax)) - target_w // 2
        paste_y = int(round(ay)) - target_h

        if frame.mode != 'RGBA':
            frame = frame.convert('RGBA')

        # PIL clips negative/overflow paste boxes safely
        temp = Image.new('RGBA', frame.size, (0, 0, 0, 0))
        temp.paste(scaled, (paste_x, paste_y),
                   scaled if scaled.mode == 'RGBA' else None)
        return Image.alpha_composite(frame, temp)

    def _render_explanation_frame(self, frame: Image.Image, turn: Dict,
                                   local_frame: int, global_frame: int) -> Image.Image:
        """Render an explanation overlay with diagrams and formulas."""
        visual_elements = turn.get("visual_elements", [])
        total_frames = int((turn["duration_ms"] / 1000.0) * self.fps)
        progress = min(1.0, local_frame / max(1, total_frames))

        draw = ImageDraw.Draw(frame)
        center_x = self.width // 2
        center_y = self.height // 2 - 100

        for i, elem in enumerate(visual_elements):
            action = elem.get("action", "")
            params = elem.get("params", {})

            # Each element appears progressively
            elem_progress = min(1.0, max(0.0,
                (progress - i * 0.2) / 0.3))

            if elem_progress <= 0:
                continue

            if action == "draw_circle":
                radius = int(params.get("radius", 100) * elem_progress)
                cx = center_x + params.get("x", 0)
                cy = center_y + params.get("y", 0)
                color = self._resolve_color(params.get("color", "primary"))
                draw.ellipse(
                    (cx - radius, cy - radius, cx + radius, cy + radius),
                    outline=color + (int(255 * elem_progress),),
                    width=3
                )

            elif action == "show_text":
                text = params.get("text", "")
                tx = center_x + params.get("x", 0)
                ty = center_y + params.get("y", 0)
                font = self._get_font(brand.FONT_MAIN, brand.FONT_SIZE_BODY)
                alpha = int(255 * elem_progress)
                draw.text((tx, ty), text, font=font,
                          fill=brand.CHALKBOARD_TEXT + (alpha,), anchor="mm")

            elif action == "show_formula":
                formula = params.get("latex", "")
                # Strip LaTeX markers for display
                formula = formula.replace("$", "").replace("\\", "")
                fy = center_y + params.get("y", 200)
                font = self._get_font(brand.FONT_MAIN, brand.FONT_SIZE_FORMULA)
                alpha = int(255 * elem_progress)

                # Write-on effect: show characters progressively
                chars_to_show = int(len(formula) * elem_progress)
                visible_text = formula[:chars_to_show]
                draw.text((center_x, fy), visible_text, font=font,
                          fill=brand.SECONDARY + (alpha,), anchor="mm")

            elif action == "draw_arrow":
                x1 = center_x + params.get("x1", 0)
                y1 = center_y + params.get("y1", 0)
                x2_full = center_x + params.get("x2", 0)
                y2_full = center_y + params.get("y2", 0)
                # Animate arrow length
                x2 = int(x1 + (x2_full - x1) * elem_progress)
                y2 = int(y1 + (y2_full - y1) * elem_progress)
                color = self._resolve_color(params.get("color", "accent"))
                draw.line([(x1, y1), (x2, y2)],
                         fill=color + (int(255 * elem_progress),), width=3)
                # Arrowhead
                if elem_progress > 0.8:
                    head_size = 10
                    draw.polygon([
                        (x2, y2),
                        (x2 - head_size, y2 - head_size),
                        (x2 + head_size, y2 - head_size),
                    ], fill=color + (255,))

        # Tiny characters in corners during explanation — same anchors as
        # brand.SHOT_PRESETS["fullscreen_explain"] (above the Shorts
        # metadata row, Chintu left of the action rail).
        gudiya_img = self.gudiya.get("neutral")
        chintu_img = self.chintu.get("neutral")
        if gudiya_img:
            frame = self._paste_character(frame, gudiya_img,
                                          150, 1580, 0.30, 0.4)
        if chintu_img:
            frame = self._paste_character(frame, chintu_img,
                                          870, 1580, 0.30, 0.4)

        return frame

    def _render_caption(self, frame: Image.Image, text: str,
                         global_frame: int) -> Image.Image:
        """Render caption text at the bottom of the frame."""
        draw = ImageDraw.Draw(frame)
        font = self._get_font(brand.FONT_CAPTION, settings.CAPTION_FONT_SIZE)

        # Break into lines
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            test = f"{current_line} {word}".strip()
            if len(test) > settings.CAPTION_MAX_CHARS_PER_LINE:
                if current_line:
                    lines.append(current_line)
                current_line = word
            else:
                current_line = test
        if current_line:
            lines.append(current_line)

        # Position
        y_start = int(self.height * settings.CAPTION_Y_POSITION)
        line_height = settings.CAPTION_FONT_SIZE + 8

        for i, line in enumerate(lines[-3:]):  # Show max 3 lines
            y = y_start + i * line_height

            # Black stroke (outline)
            stroke_color = (0, 0, 0, 255)
            for dx in range(-settings.CAPTION_STROKE_WIDTH, settings.CAPTION_STROKE_WIDTH + 1):
                for dy in range(-settings.CAPTION_STROKE_WIDTH, settings.CAPTION_STROKE_WIDTH + 1):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((self.width // 2 + dx, y + dy), line,
                             font=font, fill=stroke_color, anchor="mm")

            # White text
            draw.text((self.width // 2, y), line,
                     font=font, fill=brand.TEXT_CAPTION + (255,), anchor="mm")

        return frame

    def _resolve_color(self, color_name: str) -> Tuple[int, int, int]:
        """Resolve a color name to RGB tuple."""
        color_map = {
            "primary": brand.PRIMARY,
            "secondary": brand.SECONDARY,
            "accent": brand.ACCENT,
            "success": brand.SUCCESS,
            "white": brand.TEXT_WHITE,
        }
        return color_map.get(color_name, brand.PRIMARY)
