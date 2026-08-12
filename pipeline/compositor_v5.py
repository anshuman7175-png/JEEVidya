"""
JEEVidya V5 — Streaming Compositor
══════════════════════════════════
Renders the video against the drift-free global Timeline and streams every
frame straight into the ffmpeg encoder. No PNGs, no disk I/O, no moviepy
assembly pass.

Inherits character compositing + camera logic from CinematicCompositor and
replaces the hot paths:
  • Background: GradientBackdrop numpy LUT (was 1,920 draw.line calls/frame)
  • Particles:  pre-baked glow sprites (was full-canvas GaussianBlur/frame)
  • Captions:   word-accurate karaoke from edge-tts VTT timings
                (was static full-turn text with a 48-call stroke)
  • Formulas:   real LaTeX via MathRenderer (was mangled plain text)

Supports res_scale for fast half-resolution previews.
"""
from __future__ import annotations

import os
import time as time_module
from typing import Any, Callable, Dict, List, Optional

from PIL import Image, ImageDraw

from config import settings, brand
from engine.backgrounds import render_chalkboard_overlay
from engine.cinematics import (CameraDynamics, apply_frame_transform,
                               whip_blur)
from engine.light import (ambient_wrap, bloom, chromatic_aberration,
                          contact_shadow, god_rays, halation, rack_focus,
                          rim_light)
from engine.relight3d import relight_character_3d
from engine.lipsync_pro import LipSyncTrack, MouthBlender
from engine.math_renderer import MathRenderer
from engine.motion_pro import CharacterMotion, transform_sprite
from engine.particles import ParticleSystem
from engine.subpixel import gate_weave, motion_ghosts, paste_subpixel
from engine.render_fast import (
    FastParticleRenderer,
    GradientBackdrop,
    _is_emphasis_word,
    draw_karaoke_caption,
    draw_kinetic_caption,
    resolve_devanagari_font,
)
from pipeline.compositor import CinematicCompositor
from pipeline.lipsync import (analyze_audio, compute_body_animation,
                              select_expression)
from pipeline.timeline import Timeline, TurnSpan
from tools.face_qc import GateResult

_SILENT_FRAME = {"db": -80, "mouth_state": 0, "is_speaking": False}


class _ScaledCamera:
    """Proxy around CameraSystem that scales positions for preview renders."""

    def __init__(self, camera, res_scale: float):
        self._camera = camera
        self._rs = res_scale

    def get_both_characters(self, active_speaker: str):
        g, c = self._camera.get_both_characters(active_speaker)
        return self._scale(g), self._scale(c)

    def _scale(self, params: Dict) -> Dict:
        p = params.copy()
        p["x"] = p["x"] * self._rs
        p["y"] = p["y"] * self._rs
        return p

    def __getattr__(self, name):
        return getattr(self._camera, name)


class StreamingCompositor(CinematicCompositor):
    """V5 renderer: Timeline in, encoded frames out."""

    def __init__(self, res_scale: float = 1.0, dna=None,
                 overrides: Optional[Dict[str, float]] = None):
        super().__init__()
        self.res_scale = res_scale
        self.dna = dna
        self.overrides = overrides or {}

        if res_scale != 1.0:
            self.width = int(settings.WIDTH * res_scale) // 2 * 2   # even dims for yuv420p
            self.height = int(settings.HEIGHT * res_scale) // 2 * 2
            self.particles = ParticleSystem(self.width, self.height)
            self.chalkboard_bg = render_chalkboard_overlay(self.width, self.height)
            self.camera = _ScaledCamera(self.camera, res_scale)

        # ─── Tier 2: Visual DNA phenotype (palette, grade, motifs) ───
        self.grade = None
        self.physics = None
        if dna is not None:
            pal = dna.palette
            self.backdrop = GradientBackdrop(self.width, self.height,
                                             top=pal["bg_top"],
                                             bottom=pal["bg_bottom"])
            self.particles = ParticleSystem(self.width, self.height,
                                            palette=dna.particle_colors)
            from engine.post_production import ColorGrade
            self.grade = ColorGrade(dna, self.width, self.height)
            self._build_motif_field(seed=dna.seed)
            print(f"  [V5] {dna.describe()}")
        else:
            self.backdrop = GradientBackdrop(self.width, self.height)

        self.fast_particles = FastParticleRenderer()
        # Critic-agent overrides: caption size/position, hologram scale
        cap_scale = 1.0 + self.overrides.get("caption_font_scale", 0.0)
        self.caption_y_frac = min(0.90, settings.CAPTION_Y_POSITION
                                  + self.overrides.get("caption_y_position", 0.0))
        self.hologram_scale = max(0.4, 1.0
                                  + self.overrides.get("hologram_scale", 0.0))
        self.caption_font = resolve_devanagari_font(
            max(14, int(settings.CAPTION_FONT_SIZE * res_scale * cap_scale)))
        self._formula_cache: Dict[str, Image.Image] = {}
        self._last_word_ms = -1.0

        # ─── PRO PATH: cinematic camera, body dynamics, real lip sync ───
        energy = dna.genes["energy"] if dna else 0.6
        self.cine = CameraDynamics(self.width, self.height, seed=0,
                                   fps=self.fps, energy=energy)
        self._mouths = {"girl": MouthBlender(self.gudiya),
                        "boy": MouthBlender(self.chintu)}
        self._motion: Dict[str, CharacterMotion] = {}
        self._char_prev: Dict[str, tuple] = {}   # velocity tracking
        self._lip_track: Optional[LipSyncTrack] = None
        self._focus = 0.0                     # rack-focus state (smoothed)
        self._xform = {"dx": 0.0, "dy": 0.0, "zoom": 1.0,
                       "whip_blur": 0.0, "whip_dir": 0.0}
        self.ca_strength = 0.6 + 0.6 * energy  # lens fingerprint
        self.bloom_enabled = True

    def _build_motif_field(self, seed: int) -> None:
        """Physics-driven floating motif background (Tier 2)."""
        from engine.physics_world import PhysicsWorld
        self.physics = PhysicsWorld(self.width, self.height, seed=seed)
        if self.dna is None:
            return
        import random as _random
        rng = _random.Random(seed + 13)
        names = self.dna.motif_names
        pal = self.dna.palette
        colors = (pal["primary"], pal["secondary"], pal["accent"])
        for _ in range(self.dna.motif_count):
            size = rng.uniform(0.04, 0.11) * self.height
            self.physics.spawn_drifter(
                rng.choice(names), size,
                rng.choice(colors) + (rng.randint(26, 60),))

        # ─── Tier 1: Bone Engine puppets (per-character fallback) ───
        from engine.rig import has_rig
        from pipeline.puppet import PuppetActor
        self.actors: Dict[str, PuppetActor] = {}
        for char_name, key, side in (("gudiya", "girl", "left"),
                                     ("chintu", "boy", "right")):
            if has_rig(char_name):
                try:
                    self.actors[key] = PuppetActor(char_name, side=side)
                except Exception as e:
                    print(f"  [V5] Puppet '{char_name}' failed to load ({e}) "
                          f"— falling back to expression swap")
        if self.actors:
            print(f"  [V5] Bone Engine active: "
                  f"{', '.join(sorted(self.actors))} rigged")
        self._puppet_turn: Optional[int] = None

    # ═══════════════════════════════════════
    # MAIN STREAMING LOOP
    # ═══════════════════════════════════════

    def render_stream(self, timeline: Timeline, encoder,
                      progress_callback: Optional[Callable[[float, str], None]] = None,
                      max_frames: Optional[int] = None) -> int:
        """Render every frame of the timeline into the encoder."""
        # Pre-analyze amplitude for every speaking turn
        audio_frames: Dict[int, List[Dict]] = {}
        for span in timeline.spans:
            turn = span.turn
            if turn.get("audio") and os.path.exists(turn["audio"]):
                audio_frames[turn["turn_id"]] = analyze_audio(turn["audio"], self.fps)

        total = timeline.total_frames
        if max_frames is not None:
            total = min(total, max_frames)

        print(f"  [V5] {timeline.describe()}")
        print(f"  [V5] Streaming {total} frames @ {self.width}x{self.height} → ffmpeg")
        start = time_module.time()

        current_span: Optional[TurnSpan] = None
        for f in range(total):
            span = timeline.span_at_frame(f)
            if span is not current_span:
                default_shot = ("fullscreen_explain"
                                if span.turn["speaker"] == "explanation" else "two_shot")
                self.camera.cut_to(span.turn.get("shot_type", default_shot))
                current_span = span

            local = f - span.start_frame
            frames = audio_frames.get(span.turn["turn_id"], [])
            fa = frames[local] if 0 <= local < len(frames) else _SILENT_FRAME

            encoder.write_frame(self._render_frame_v5(f, local, span, fa, timeline))
            self.camera.update()

            if f and f % 100 == 0:
                elapsed = time_module.time() - start
                rate = f / max(0.1, elapsed)
                eta = (total - f) / max(0.1, rate)
                msg = f"frame {f}/{total} ({rate:.1f} fps, ETA {eta:.0f}s)"
                print(f"  [V5] {msg}")
                if progress_callback:
                    progress_callback(f / total, msg)

        elapsed = time_module.time() - start
        print(f"  [V5] ✓ {total} frames in {elapsed:.1f}s "
              f"({total / max(0.1, elapsed):.1f} fps render rate)")
        if progress_callback:
            progress_callback(1.0, "Frames complete")
        return total

    # ═══════════════════════════════════════
    # TIER 0 — jvmake SEGMENT RENDER (one DAG node)
    # ═══════════════════════════════════════

    def render_segment(self, timeline: Timeline, span_index: int, encoder,
                       seed_key: str, prev_shot: str = "two_shot",
                       max_frames: Optional[int] = None) -> int:
        """
        Render ONE turn as a standalone, deterministic video segment.

        Everything stochastic or phase-dependent (particle field, backdrop
        drift, body sway, camera transition) is derived from `seed_key` —
        the segment's content hash — never from its absolute position on
        the timeline. A segment's pixels therefore depend only on its own
        inputs: it can be cached forever, survive crashes, and stay valid
        when neighbouring turns are edited.
        """
        import random as _random

        from pipeline.camera import CameraSystem

        span = timeline.spans[span_index]
        turn = span.turn
        speaker = turn["speaker"]

        # ── Deterministic per-segment state ──
        rng = _random.Random(int(seed_key[:16], 16))
        phase = int(seed_key[16:24], 16) % 100_000   # visual phase offset

        cam = CameraSystem()
        if prev_shot in brand.SHOT_PRESETS:
            cam.current_shot = cam.prev_shot = prev_shot
        default_shot = ("fullscreen_explain" if speaker == "explanation"
                        else "two_shot")
        cam.cut_to(turn.get("shot_type", default_shot))
        self.camera = (_ScaledCamera(cam, self.res_scale)
                       if self.res_scale != 1.0 else cam)
        self.particles = ParticleSystem(
            self.width, self.height, rng=rng,
            palette=self.dna.particle_colors if self.dna else None)
        if self.physics is not None:
            self._build_motif_field(seed=int(seed_key[24:32], 16))
        self._last_word_ms = -1.0
        self._puppet_turn = None

        amp_frames: List[Dict] = []
        if turn.get("audio") and os.path.exists(turn["audio"]):
            amp_frames = analyze_audio(turn["audio"], self.fps)

        # ─── PRO PATH: seed the cinematic state for this segment ───
        seg_seed = int(seed_key[32:40], 16)
        # Springs start AT the previous shot's params so the cut animates
        prev_preset = brand.SHOT_PRESETS.get(prev_shot,
                                             brand.SHOT_PRESETS["two_shot"])
        cur_shot = turn.get("shot_type", default_shot)
        cur_preset = brand.SHOT_PRESETS.get(cur_shot,
                                            brand.SHOT_PRESETS["two_shot"])
        rs = self.res_scale

        def scaled(p):
            return {"x": p["x"] * rs, "y": p["y"] * rs,
                    "scale": p["scale"], "opacity": p["opacity"]}

        prev_params = {
            "girl": scaled(prev_preset["active"] if speaker == "girl"
                           else prev_preset["inactive"]),
            "boy": scaled(prev_preset["inactive"] if speaker == "girl"
                          else prev_preset["active"]),
        }
        self.cine.begin_segment(prev_params, seed=seg_seed)
        if prev_shot != cur_shot:
            self.cine.on_cut(prev_preset["active"]["x"],
                             cur_preset["active"]["x"])
        self._motion = {
            "girl": CharacterMotion(seg_seed ^ 0xA11CE, fps=self.fps,
                                    side=-1.0),
            "boy": CharacterMotion(seg_seed ^ 0xB0B, fps=self.fps,
                                   side=1.0),
        }
        self._char_prev = {}
        self._lip_track = LipSyncTrack(amp_frames, span.words,
                                       span.start_ms, self.fps)
        self._focus = 1.0 if cur_shot in ("extreme_closeup",
                                          "reaction_cut") else 0.0

        n = span.end_frame - span.start_frame
        if max_frames is not None:
            n = min(n, max_frames)

        for local in range(n):
            # Real timeline ms (captions/visemes) vs phased frame (motion)
            t_ms = timeline.frame_ms(span.start_frame + local)
            fa = amp_frames[local] if 0 <= local < len(amp_frames) else _SILENT_FRAME
            encoder.write_frame(self._render_frame_v5(
                phase + local, local, span, fa, timeline, t_ms=t_ms))
            self.camera.update()
        return n

    # ═══════════════════════════════════════
    # PER-FRAME COMPOSITION
    # ═══════════════════════════════════════

    def _render_frame_v5(self, global_frame: int, local_frame: int,
                         span: TurnSpan, frame_audio: Dict,
                         timeline: Timeline,
                         t_ms: Optional[float] = None) -> Image.Image:
        # t_ms: real timeline position for speech-locked lookups (captions,
        # visemes). May differ from global_frame when rendering cached
        # segments, where global_frame carries a content-derived phase.
        if t_ms is None:
            t_ms = timeline.frame_ms(global_frame)
        speaker = span.turn["speaker"]
        emotion = span.turn.get("emotion", "neutral")

        # Layer 1: background
        if speaker == "explanation":
            frame = self.chalkboard_bg.copy()
        else:
            frame = self.backdrop.get(global_frame)

        # Camera: one physical transform per frame (drift, push-in, punch)
        self._xform = self.cine.frame_transform()

        # Word-start events: physics reactions + emphasis beats
        word = timeline.word_at(t_ms)
        if word is not None and word.start_ms != self._last_word_ms:
            self._last_word_ms = word.start_ms
            if self.physics is not None:
                self.physics.react(word.text)      # real physics, word-locked
            if _is_emphasis_word(word.text):
                # Numbers land: camera shake + speaker recoil + spark burst
                self.cine.impulse(0.8)
                m = self._motion.get(speaker)
                if m:
                    m.hit(0.7)
                if self.physics is not None:
                    self.physics.burst(self.width * 0.5, self.height * 0.42,
                                       count=14, speed=6.0)

        # Layer 1.5: unified physics — motif field
        if self.physics is not None:
            self.physics.step()
            if speaker != "explanation":
                frame = self._render_physics_layer(frame)

        # Layer 2: particles (sprite-stamped glow)
        self.particles.update()
        frame = self.fast_particles.render(frame, self.particles.particles)

        # Depth: rack-focus pull on the background plane in close-ups
        target_focus = 1.0 if span.turn.get("shot_type") in (
            "extreme_closeup", "reaction_cut") else 0.0
        self._focus += (target_focus - self._focus) * 0.18
        if speaker != "explanation" and self._focus > 0.02:
            frame = rack_focus(frame, self._focus)

        # Layer 3: characters / explanation content
        if speaker == "explanation":
            frame = self._render_explanation_v5(frame, span, local_frame)
        else:
            if self.actors:
                frame = self._render_puppet_frame(
                    frame, global_frame, span, frame_audio, timeline,
                    t_ms=t_ms)
            else:
                frame = self._render_character_frame_pro(
                    frame, local_frame, speaker, emotion, span, t_ms)

            # Layer 4: kinetic captions (word pops, live-word glow)
            chunk, active = timeline.active_caption(span, t_ms)
            if chunk:
                accent = (self.dna.palette["secondary"] if self.dna
                          else brand.TEXT_CAPTION_ACTIVE)
                emph = (self.dna.palette["accent"] if self.dna
                        else brand.ACCENT)
                if frame.mode != "RGBA":
                    frame = frame.convert("RGBA")
                draw_kinetic_caption(
                    frame,
                    [w.text for w in chunk.words],
                    [w.start_ms for w in chunk.words],
                    active, t_ms,
                    self.caption_font,
                    y=int(self.height * getattr(self, "caption_y_frac",
                                                settings.CAPTION_Y_POSITION)),
                    accent=accent, emphasis=emph,
                )

        # Flatten to RGB for the encoder
        if frame.mode == "RGBA":
            rgb = Image.new("RGB", frame.size, brand.BG_TOP)
            rgb.paste(frame, mask=frame.split()[3])
            frame = rgb

        # ─── LENS STACK (order matters, like a real post chain) ───
        # 1. whip-pan motion blur on cut frames
        if self._xform["whip_blur"] > 0.0:
            frame = whip_blur(frame, self._xform["whip_blur"],
                              self._xform["whip_dir"])
        # 2. bloom: true highlights halo (captions, formulas, sparks)
        if self.bloom_enabled:
            frame = bloom(frame, threshold=200, strength=0.45)
            # 2b. halation: film's warm bleed around the same highlights
            frame = halation(frame, threshold=225, strength=0.35)
        # 3. chromatic aberration at the edges (subliminal lens truth)
        frame = chromatic_aberration(frame, self.ca_strength)
        # 4. gate weave: 0.35px mechanical wobble — the film transport
        frame = gate_weave(frame, global_frame,
                           self.dna.seed if self.dna else 29)
        # 5. per-DNA film grade + grain, always last
        if self.grade is not None:
            frame = self.grade.apply(frame, global_frame)
        return frame

    # ═══════════════════════════════════════
    # PRO CHARACTER PATH (no rig required)
    # ═══════════════════════════════════════

    def _render_character_frame_pro(self, frame: Image.Image,
                                    local_frame: int, speaker: str,
                                    emotion: str, span: TurnSpan,
                                    t_ms: float) -> Image.Image:
        """World-class 2D character compositing:
        spring camera → body dynamics (squash/lean/breath) → continuous
        viseme mouth blend → ambient wrap + rim light → contact shadow.
        """
        g_target, c_target = self.camera.get_both_characters(speaker)
        cx, cy = self.width / 2, self.height / 2
        pal = self.dna.palette if self.dna else None
        wrap_color = pal["bg_bottom"] if pal else brand.BG_BOTTOM
        rim_color = pal["glow"] if pal else (150, 210, 255)

        # Lip-sync signal for the active speaker (built once per segment)
        openness, width_bias = (0.0, 0.0)
        if self._lip_track is not None:
            openness, width_bias = self._lip_track.at(local_frame, t_ms)

        for key, target in (("girl", g_target), ("boy", c_target)):
            params = self.cine.smooth_char(key, target)
            if params["opacity"] <= 0.01 or params["scale"] <= 0.01:
                continue
            is_active = (key == speaker)
            lib = self.gudiya if key == "girl" else self.chintu

            # Body dynamics → placement + deformation
            motion = self._motion.get(key)
            if motion is None:
                motion = self._motion[key] = CharacterMotion(
                    7 if key == "girl" else 11, fps=self.fps,
                    side=-1.0 if key == "girl" else 1.0)
            state = motion.step(is_active, openness if is_active else 0.0,
                                emotion)

            # Mouth: continuous phoneme-shaped blend (speaker only)
            if is_active:
                base = "neutral" if emotion in ("explaining", "neutral") \
                    else select_expression(False, {}, emotion)
                img = self._mouths[key].frame(base, openness, width_bias)
            else:
                img = lib.get(select_expression(False, {}, emotion))
            if img is None:
                continue

            # Light: scene-color wrap + key-side rim (both cached)
            side = 1.0 if key == "girl" else -1.0   # lit from center
            img = ambient_wrap(img, wrap_color, amount=0.16)
            img = rim_light(img, side=side, color=rim_color, strength=0.7)

            # Camera transform + body offsets
            x, y, scale = apply_frame_transform(
                params["x"] + state.dx * self.res_scale,
                params["y"] + state.dy * self.res_scale,
                params["scale"], self._xform, cx, cy)

            # Safe framing: the push-in/punch zoom scales positions away
            # from center — without this clamp, close-ups slowly crop
            # the head out of frame over the shot
            target_h = self._char_target_h(scale)
            target_w = int(target_h * img.width / img.height)
            x, y = self._safe_anchor(x, y, target_w, target_h)

            # Ground the character: soft contact shadow (lifts on rises)
            if target_w > 4 and params["opacity"] > 0.15:
                lift = min(1.0, abs(state.dy) / 26.0)
                shadow = contact_shadow(int(target_w * 0.72),
                                        opacity=int(70 * (1 - lift * 0.6)))
                if frame.mode != "RGBA":
                    frame = frame.convert("RGBA")
                frame.alpha_composite(
                    shadow, (int(x - shadow.width / 2),
                             int(y - shadow.height * 0.55)))

            # Squash & stretch + lean
            img = transform_sprite(img, state.sx, state.sy, state.lean)

            # Scale to final size ONCE, apply opacity
            th = max(2, self._char_target_h(scale, extra=state.sy))
            tw = max(2, int(th * img.width / img.height))
            sprite = img.resize((tw, th), Image.Resampling.LANCZOS)
            if params["opacity"] < 0.99:
                a = sprite.split()[3].point(
                    lambda p, o=params["opacity"]: int(p * o))
                sprite.putalpha(a)

            # Velocity (real, from the smoothed camera+body position)
            px, py = self._char_prev.get(key, (x, y))
            vx, vy = x - px, y - py
            self._char_prev[key] = (x, y)

            if frame.mode != "RGBA":
                frame = frame.convert("RGBA")
            # 2-sample motion blur along the true velocity vector…
            frame = motion_ghosts(frame, sprite, x, y, vx, vy)
            # …then the sharp pass on a continuous sub-pixel raster
            frame = paste_subpixel(frame, sprite, x, y)
        return frame

    def _render_physics_layer(self, frame: Image.Image) -> Image.Image:
        """Draw the physics world: rotating motifs behind, sparks above."""
        from tools.motif_forge import render_motif
        if frame.mode != "RGBA":
            frame = frame.convert("RGBA")
        overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        draw = None
        for b in self.physics.bodies:
            if b.kind == "spark":
                if draw is None:
                    draw = ImageDraw.Draw(overlay)
                s = b.size
                draw.ellipse((b.x - s, b.y - s, b.x + s, b.y + s),
                             fill=b.color)
            elif b.kind != "dot":
                sprite = render_motif(b.kind, max(24, int(b.size)),
                                      b.color[:3], alpha=b.color[3])
                if abs(b.angle) > 0.02:
                    sprite = sprite.rotate(
                        -b.angle * 57.2958,
                        resample=Image.Resampling.BILINEAR)
                # Parallax: camera drift displaces each depth plane
                # differently — the background becomes a SPACE
                par = (1.0 / b.depth - 1.0) * 6.0
                overlay.alpha_composite(
                    sprite,
                    (int(b.x - sprite.width / 2 + self._xform["dx"] * par),
                     int(b.y - sprite.height / 2 + self._xform["dy"] * par)))
        return Image.alpha_composite(frame, overlay)

    # ═══════════════════════════════════════
    # TIER 1 — BONE ENGINE PUPPET FRAME
    # ═══════════════════════════════════════

    def _render_puppet_frame(self, frame: Image.Image, global_frame: int,
                             span: TurnSpan, frame_audio: Dict,
                             timeline: Timeline,
                             t_ms: Optional[float] = None) -> Image.Image:
        """Skeletal puppets: visemes, gestures, head turns, spring physics.
        Characters without a rig keep the V2 expression-swap path."""
        speaker = span.turn["speaker"]
        emotion = span.turn.get("emotion", "neutral")
        if t_ms is None:
            t_ms = timeline.frame_ms(global_frame)

        # New turn → (re)build viseme + gesture tracks on both actors
        tid = span.turn.get("turn_id")
        if tid != self._puppet_turn:
            self._puppet_turn = tid
            for key, actor in self.actors.items():
                actor.begin_span(span, is_my_turn=(key == speaker),
                                 emotion=emotion)

        # Listener coupling (§XVII): every non-speaking actor tracks the
        # speaker's affect with ~400 ms lag at 0.4 gain. This runs BEFORE
        # any pose is built, so the reactive listening is part of the same
        # frame it reacts to — and it is what kills the dead-eyed listener.
        talker = self.actors.get(speaker)
        if talker is not None:
            for key, actor in self.actors.items():
                if key != speaker:
                    actor.track_speaker(t_ms, talker.affect)

        g_raw, c_raw = self.camera.get_both_characters(speaker)
        # Spring-smoothed camera + global drift/push-in (pro path)
        cx, cy = self.width / 2, self.height / 2
        g_params = self.cine.smooth_char("girl", g_raw)
        c_params = self.cine.smooth_char("boy", c_raw)
        for p in (g_params, c_params):
            p["x"], p["y"], p["scale"] = apply_frame_transform(
                p["x"], p["y"], p["scale"], self._xform, cx, cy)
        layout = (("girl", g_params, c_params), ("boy", c_params, g_params))

        for key, params, other in layout:
            if params["opacity"] <= 0.01 or params["scale"] <= 0.01:
                continue
            is_active = (key == speaker)
            actor = self.actors.get(key)

            # Ground every character (lifts subtly while it moves)
            th = self._char_target_h(params["scale"])
            if th > 8 and params["opacity"] > 0.15:
                shadow = contact_shadow(int(th * 0.5), opacity=64)
                if frame.mode != "RGBA":
                    frame = frame.convert("RGBA")
                frame.alpha_composite(
                    shadow, (int(params["x"] - shadow.width / 2),
                             int(params["y"] - shadow.height * 0.55)))

            if actor is None:                       # V2 fallback, per character
                lib = self.gudiya if key == "girl" else self.chintu
                expr = select_expression(is_active, frame_audio
                                         if is_active else _SILENT_FRAME,
                                         emotion)
                x, y, bscale = compute_body_animation(
                    global_frame, is_active and frame_audio.get(
                        "is_speaking", False), params["x"], params["y"])
                img = lib.get(expr)
                if img:
                    frame = self._paste_character(
                        frame, img, int(x), int(y),
                        params["scale"] * bscale, params["opacity"])
                continue

            # Head turns toward the current speaker; speaker plays slightly toward camera
            dirn = 1.0 if (other["x"] - params["x"]) >= 0 else -1.0
            look = dirn * (0.15 if is_active else 0.5)
            pose = actor.pose_at(
                global_frame, t_ms, is_active, emotion,
                frame_audio if is_active else None, look)
            img, dxr, dyr = actor.render(pose)

            # High-quality LANCZOS scaling to target resolution
            target_h = self._char_target_h(params["scale"])
            target_w = max(2, int(target_h * img.width / img.height))
            scaled_puppet = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

            if params["opacity"] < 0.99 and scaled_puppet.mode == "RGBA":
                r, g, b, a = scaled_puppet.split()
                a = a.point(lambda p, o=params["opacity"]: int(p * o))
                scaled_puppet = Image.merge("RGBA", (r, g, b, a))

            # Studio Light Pass: 3D Volumetric Relighting + scene bounce wrap + key-side rim light
            side = 1.0 if key == "girl" else -1.0
            scaled_puppet = relight_character_3d(
                scaled_puppet,
                light_dir=(-side * 0.45, -0.55, 0.77),
                key_intensity=0.45,
                specular_strength=0.25)
            scaled_puppet = ambient_wrap(scaled_puppet, (20, 25, 45), amount=0.14)
            scaled_puppet = rim_light(scaled_puppet, side=side, color=(160, 220, 255), strength=0.65)

            # Continuous sub-pixel placement (prevents integer motion judder)
            px = params["x"] + dxr * target_h
            py = params["y"] + dyr * target_h
            ax, ay = self._safe_anchor(px, py, target_w, target_h)
            frame = paste_subpixel(frame, scaled_puppet, ax, ay)
        return frame

    # ═══════════════════════════════════════
    # AFFECT QC (§XVII) — performance-level gates
    # ═══════════════════════════════════════

    def affect_gates(self) -> List[GateResult]:
        """One gate per rigged actor, from the performance just rendered.

        These are the gates decoded pixels CANNOT catch: a nervous system
        that snapped, or a face whose expression contradicts the state
        driving it. They ride in the same QCReport the publisher checks,
        so an incoherent performance cannot be uploaded.
        """
        gates: List[GateResult] = []
        for key in sorted(self.actors):
            violations = self.actors[key].affect_violations()
            gates.append(GateResult(
                name=f"affect_{key}", passed=not violations,
                value=float(len(violations)), threshold=0.0,
                detail="; ".join(violations) if violations
                else "state continuous, channels coherent"))
        return gates

    # ═══════════════════════════════════════
    # EXPLANATION SCENES (real LaTeX)
    # ═══════════════════════════════════════

    def _formula_image(self, latex: str) -> Image.Image:
        img = self._formula_cache.get(latex)
        if img is None:
            img = MathRenderer.render(latex, color=brand.SECONDARY, font_size=30)
            max_w = int(self.width * 0.82)
            if img.width > max_w:
                ratio = max_w / img.width
                img = img.resize((max_w, max(1, int(img.height * ratio))),
                                 Image.Resampling.LANCZOS)
            self._formula_cache[latex] = img
        return img

    def _render_explanation_v5(self, frame: Image.Image, span: TurnSpan,
                               local_frame: int) -> Image.Image:
        turn = span.turn
        rs = self.res_scale
        total_frames = max(1, span.end_frame - span.start_frame)
        progress = min(1.0, local_frame / total_frames)

        if frame.mode != "RGBA":
            frame = frame.convert("RGBA")

        # Cinematic reveal light: soft god rays fade in behind the content
        ray_color = (self.dna.palette["glow"] if self.dna
                     else (255, 235, 180))
        rays = god_rays((self.width, self.height), color=ray_color)
        if progress < 0.25:                    # fade the shafts in
            faded = rays.copy()
            faded.putalpha(faded.split()[3].point(
                lambda p: int(p * progress * 4)))
            frame.alpha_composite(faded)
        else:
            frame.alpha_composite(rays)

        draw = ImageDraw.Draw(frame)
        center_x = self.width // 2
        center_y = self.height // 2 - int(100 * rs)

        for i, elem in enumerate(turn.get("visual_elements", [])):
            action = elem.get("action", "")
            params = elem.get("params", {})
            ep = min(1.0, max(0.0, (progress - i * 0.2) / 0.3))
            if ep <= 0:
                continue

            if action == "draw_circle":
                radius = int(params.get("radius", 100) * ep * rs)
                cx = center_x + int(params.get("x", 0) * rs)
                cy = center_y + int(params.get("y", 0) * rs)
                color = self._resolve_color(params.get("color", "primary"))
                draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                             outline=color + (int(255 * ep),), width=max(2, int(3 * rs)))

            elif action == "show_text":
                font = self._get_font(brand.FONT_MAIN, max(12, int(brand.FONT_SIZE_BODY * rs)))
                draw.text((center_x + int(params.get("x", 0) * rs),
                           center_y + int(params.get("y", 0) * rs)),
                          params.get("text", ""), font=font,
                          fill=brand.CHALKBOARD_TEXT + (int(255 * ep),), anchor="mm")

            elif action == "show_formula":
                # V5 Tier 2: HOLOGRAPHIC LaTeX — glowing glass panel that
                # materializes with a scanline and bobs while on screen
                latex = params.get("latex", "")
                if latex:
                    from engine.holographic import formula_panel
                    pal = self.dna.palette if self.dna else None
                    holo_scale = getattr(self, "hologram_scale", 1.0)
                    img, bob = formula_panel(
                        latex,
                        t_seconds=local_frame / self.fps,
                        reveal=ep,
                        color=(pal["secondary"] if pal else brand.SECONDARY),
                        glow_color=(pal["glow"] if pal else brand.PRIMARY),
                        max_width=int(self.width * 0.82 * holo_scale / max(rs, 1e-6) * rs),
                        font_size=30)
                    if img is not None:
                        if rs != 1.0:
                            img = img.resize((max(1, int(img.width * rs)),
                                              max(1, int(img.height * rs))),
                                             Image.Resampling.LANCZOS)
                        fy = center_y + int(params.get("y", 200) * rs) + bob
                        frame.alpha_composite(
                            img, dest=(center_x - img.width // 2,
                                       fy - img.height // 2))
                        draw = ImageDraw.Draw(frame)

            elif action == "draw_arrow":
                x1 = center_x + int(params.get("x1", 0) * rs)
                y1 = center_y + int(params.get("y1", 0) * rs)
                x2f = center_x + int(params.get("x2", 0) * rs)
                y2f = center_y + int(params.get("y2", 0) * rs)
                x2 = int(x1 + (x2f - x1) * ep)
                y2 = int(y1 + (y2f - y1) * ep)
                color = self._resolve_color(params.get("color", "accent"))
                draw.line([(x1, y1), (x2, y2)],
                          fill=color + (int(255 * ep),), width=max(2, int(3 * rs)))
                if ep > 0.8:
                    hs = max(5, int(10 * rs))
                    draw.polygon([(x2, y2), (x2 - hs, y2 - hs), (x2 + hs, y2 - hs)],
                                 fill=color + (255,))

        # Tiny corner characters (scaled)
        for lib, x in ((self.gudiya, 150), (self.chintu, 930)):
            img = lib.get("neutral")
            if img:
                frame = self._paste_character(frame, img, int(x * rs),
                                              int(1650 * rs), 0.30, 0.4)
        return frame
