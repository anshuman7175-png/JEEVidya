"""
JEEVidya V5 — Puppet Actor (Tier 1)
═══════════════════════════════════
The performance layer between the Timeline and the Bone Engine.

For every frame, a PuppetActor fuses:
  • an EMOTION baseline (curious tilts, confident chin-up, amazed recoil…)
  • breathing (always) + speech bounce/sway/pulse (scaled by energy)
  • the GESTURE track (keyword-triggered at exact VTT word timestamps)
  • the VISEME track (text → mouth shapes, gated by live amplitude)
  • speech-beat micro-nods on word onsets
  • a Poisson blink scheduler (seeded per character → deterministic)
  • head turn toward whoever is speaking (2.5D yaw)
into one PuppetPose, then advances the spring-chain physics and renders.

Falls back cleanly: if a character has no rig, the compositor keeps the
V2 expression-swap path.
"""
from __future__ import annotations

import math
import random
from typing import Dict, Optional, Tuple

from PIL import Image

from config import settings
from engine.bone_engine import BoneEngine, PuppetPose
from engine.gestures import GestureTrack
from engine.pose_library import PoseLibrary, PoseState, DEFAULT_POSE
from engine.rig import Rig, has_rig  # noqa: F401  (has_rig re-exported)
from engine.visemes import (AmplitudeEnvelope, VisemeTrack,
                            mouth_openness_blend)

# Character name → dialogue speaker key
SPEAKER_OF = {"gudiya": "girl", "chintu": "boy"}

# Emotion → pose baseline (all additive, gently applied)
EMOTION_BASELINE: Dict[str, Dict[str, float]] = {
    "neutral":      {},
    "curious":      {"head_tilt": 5, "brow": 0.45, "lean": 2, "energy": 1.0},
    "enthusiastic": {"brow": 0.35, "lean": 2, "energy": 1.45},
    "confident":    {"head_nod": -0.18, "lean": -1.5, "energy": 1.15},
    "amazed":       {"brow": 0.9, "lean": 3, "energy": 1.3},
    "thinking":     {"head_tilt": 7, "head_yaw": -0.25, "head_nod": -0.12,
                     "brow": 0.3, "energy": 0.8},
    "happy":        {"head_tilt": -3, "brow": 0.25, "energy": 1.2},
    "explaining":   {"head_yaw": 0.1, "energy": 1.05},
    "dramatic":     {"brow": 0.7, "lean": 3.5, "energy": 1.6},
}

# Emotion → one-shot gesture fired when the actor STARTS speaking a turn
EMOTION_ENTRY_GESTURE: Dict[str, str] = {
    "curious": "lean_in", "amazed": "recoil", "enthusiastic": "excited_bounce",
    "dramatic": "recoil", "thinking": "think_tilt", "happy": "excited_bounce",
    "confident": "nod",
}

_BLINK_FRAMES = 6         # close(2) hold(1) open(3)
_SILENT = {"db": -80.0, "mouth_state": 0, "is_speaking": False}


class PuppetActor:
    """One rigged character performing on the global timeline."""

    def __init__(self, character: str, side: str = "left"):
        self.character = character
        self.speaker_key = SPEAKER_OF.get(character, character)
        self.side = side                       # screen side in two_shot
        self.rig = Rig.load(character)
        self.engine = BoneEngine(self.rig)

        # Deterministic per-character randomness (bit-identical re-renders)
        self._rng = random.Random(character)
        self._next_blink_ms = self._rng.uniform(800, 2600)
        self._blink_start_ms = -1e9
        self._double_blink_pending = False     # 10% chance second blink

        # Micro-saccade state (random eye jitter between fixations)
        self._saccade_dx: float = 0.0
        self._saccade_dy: float = 0.0
        self._next_saccade_ms: float = self._rng.uniform(400, 1200)

        # Listener reactive-nod tracking (reactive only, not periodic)
        self._last_listener_nod_ms: float = -5000.0

        # Head motion asymmetry: alternate tilt direction on emphasis beats
        self._tilt_sign: float = 1.0
        # Question brow hold: sustained raise for question turns
        self._question_brow: float = 0.0

        self.gestures = GestureTrack()
        self._viseme_track: Optional[VisemeTrack] = None
        self._current_turn_id: Optional[int] = None
        self._was_active = False

        # Attack/release amplitude follower: the jaw moves like muscle,
        # not like a VU meter (kills mouth flutter on sustained vowels)
        self._env = AmplitudeEnvelope(settings.FPS)

        # Speaking body-pose rotation (hands!): cycle natural talking
        # poses on phrase-length intervals instead of freezing in one
        self._speak_pose: str = "explaining"
        self._next_speak_pose_ms: float = 0.0

        # Pose library integration
        self.pose_lib = PoseLibrary(character, rig_scale=self.engine.scale)
        self.engine.pose_lib = self.pose_lib
        self.pose_state = PoseState(rng=self._rng)
        if self.pose_lib.has_poses:
            self._init_alt_torsos()

        # Smoothed channels (critically-damped) so cuts never snap the head
        # NOTE: tilt + brow rates slowed from v1 (0.25→0.18, 0.3→0.20)
        # for more natural, less reactive feel
        self._smooth = {"head_yaw": 0.0, "lean": 0.0, "head_tilt": 0.0,
                        "brow": 0.0}

    def _init_alt_torsos(self) -> None:
        """Crop torso regions from each pose image and register with BoneEngine."""
        torsos = {}
        for name in self.pose_lib.pose_names:
            torso = self.pose_lib.get_torso(
                name,
                neck_y=self.engine.skel.neck[1],
                torso_offset=self.engine.torso_off,
                original_torso=self.engine.torso)
            if torso is not self.engine.torso:
                torsos[name] = torso
        if torsos:
            self.engine.set_alt_torsos(torsos)
            print(f"  [Puppet] {self.character}: {len(torsos)} alt torsos registered")

    # ─── span lifecycle ───────────────────────────────────

    def begin_span(self, span, is_my_turn: bool, emotion: str) -> None:
        """Called once when the timeline enters a new turn."""
        self._current_turn_id = span.turn.get("turn_id")
        energy = EMOTION_BASELINE.get(emotion, {}).get("energy", 1.0)

        if is_my_turn:
            self._viseme_track = VisemeTrack.from_words(span.words,
                                                         turn_end_ms=span.end_ms)
            # Keyword gestures at the exact word timestamps
            self.gestures.schedule_from_words(span.words, energy=min(1.2, energy))
            # Beat gestures fill the keyword-less stretches (hands alive)
            self.gestures.schedule_beats(span.words, self._rng,
                                         energy=min(1.2, energy))
            # Entry gesture for the emotion
            entry = EMOTION_ENTRY_GESTURE.get(emotion)
            if entry:
                self.gestures.schedule(entry, span.start_ms + 60, scale=0.9)
            # Question turns end with an inquisitive tilt
            text = span.turn.get("text", "")
            if text.rstrip().endswith(("?", "?!")) and span.words:
                self.gestures.schedule("think_tilt",
                                       max(span.start_ms,
                                           span.end_ms - 900), scale=0.55)
                # Sustained brow raise for the whole question turn
                self._question_brow = 0.35
            else:
                self._question_brow = 0.0
        else:
            self._viseme_track = None
            # Listener reacts to what they hear
            if emotion in ("amazed", "dramatic"):
                self.gestures.schedule("recoil", span.start_ms + 350, scale=0.5)
            elif emotion == "curious":
                self.gestures.schedule("lean_in", span.start_ms + 250, scale=0.6)
        self.gestures.clear_before(span.start_ms)
        # Cut-masking blink: humans blink at attention shifts, and a
        # blink just after the cut makes the edit itself feel smoother
        self._next_blink_ms = min(self._next_blink_ms,
                                  span.start_ms + self._rng.uniform(120, 300))
        self._was_active = is_my_turn

    # ─── blink scheduler (log-normal + double-blink) ──────

    def _next_blink_interval(self) -> float:
        """Log-normal distribution: median ~3s, range 1.5–7s.
        Real human blink statistics, not uniform random."""
        return max(1400.0, min(7500.0,
                               math.exp(self._rng.gauss(8.0, 0.4))))

    def _blink_amount(self, t_ms: float, fps: int) -> float:
        if t_ms >= self._next_blink_ms:
            self._blink_start_ms = t_ms
            # 10% double-blink: second blink 150ms later
            if self._double_blink_pending:
                self._double_blink_pending = False
                self._next_blink_ms = t_ms + self._next_blink_interval()
            elif self._rng.random() < 0.10:
                self._double_blink_pending = True
                self._next_blink_ms = t_ms + 150.0  # tight second blink
            else:
                self._next_blink_ms = t_ms + self._next_blink_interval()
        dt = t_ms - self._blink_start_ms
        dur = _BLINK_FRAMES * 1000.0 / fps
        if 0 <= dt <= dur:
            t = dt / dur
            return math.sin(t * math.pi) ** 0.7   # fast close, ease open
        return 0.0

    # ─── micro-saccade generator ─────────────────────────────

    def _update_saccades(self, t_ms: float) -> Tuple[float, float]:
        """Generate small random eye position jitter every 0.3–1.5s.
        The brain reads motionless eyes as 'doll' — this breaks that."""
        if t_ms >= self._next_saccade_ms:
            # New fixation point: small random offset (max ±3px head-local)
            self._saccade_dx = self._rng.uniform(-2.5, 2.5)
            self._saccade_dy = self._rng.uniform(-1.5, 1.5)
            self._next_saccade_ms = t_ms + self._rng.uniform(300, 1500)
        return self._saccade_dx, self._saccade_dy

    # ─── speaking pose rotation ──────────────────────────

    _SPEAK_POSE_ROTATION = ("explaining", "presenting", "both_hands_wide",
                            "hand_on_heart", "counting")

    def _speaking_pose(self, t_ms: float) -> str:
        """Deterministically cycle natural talking poses every 2.4–4.2 s
        so the hands gesticulate through the whole turn (only poses the
        character actually has are eligible)."""
        if t_ms >= self._next_speak_pose_ms:
            options = [p for p in self._SPEAK_POSE_ROTATION
                       if p in self.pose_lib.pose_names
                       and p != self._speak_pose]
            if options:
                self._speak_pose = options[self._rng.randrange(len(options))]
            self._next_speak_pose_ms = t_ms + self._rng.uniform(2400, 4200)
        return self._speak_pose

    # --- pose synthesis ---

    def pose_at(self, global_frame: int, t_ms: float, is_speaking: bool,
                emotion: str, frame_audio: dict,
                look_dir: float, fps: int = settings.FPS) -> PuppetPose:
        """Build the full pose for one frame."""
        fa = frame_audio or _SILENT
        base = EMOTION_BASELINE.get(emotion, {})
        energy = base.get("energy", 1.0)
        pose = PuppetPose(energy=energy)

        # 1 · Emotion baseline (listeners get a MUCH softened version)
        soft = 1.0 if is_speaking else 0.35   # was 0.55 → quieter listener
        pose.lean += base.get("lean", 0.0) * soft
        pose.head_tilt += base.get("head_tilt", 0.0) * soft
        pose.head_nod += base.get("head_nod", 0.0) * soft
        pose.brow += base.get("brow", 0.0) * soft
        yaw_target = base.get("head_yaw", 0.0) * soft + look_dir * 0.45

        # 1b · Question brow hold: sustained raise for question turns
        pose.brow += self._question_brow

        # 2 · Breathing (always) + speech motion (reuses V2 tuning constants)
        #     Breathing amplitude modulated by a slow swell so it's never
        #     metronomically regular (AI tell #1).
        f = global_frame
        breath_mod = 0.7 + 0.3 * math.sin(f * 0.007)  # slow amplitude swell
        pose.bounce += math.sin(f * settings.BODY_BREATHE_SPEED) \
            * settings.BODY_BREATHE_AMPLITUDE * breath_mod
        if is_speaking and fa.get("is_speaking", False):
            pose.bounce += math.sin(f * settings.BODY_SPEAK_SPEED) \
                * settings.BODY_SPEAK_BOUNCE * energy
            pose.sway += math.sin(f * 0.25) * settings.BODY_SPEAK_SWAY * energy
            pose.squash += math.sin(f * 0.3) \
                * settings.BODY_SPEAK_SCALE_PULSE * energy
            # Speech energy → brow: louder = slightly raised brow
            amp_norm = max(0.0, min(1.0, (fa.get("db", -80.0) + 50.0) / 35.0))
            pose.brow += amp_norm * 0.28 * energy

        # 2b · Listener reactive nods: NOT periodic — only react to
        #      speaker emphasis peaks. Visual hierarchy: listener is calm.
        if not is_speaking:
            amp_now = max(0.0, min(1.0, (fa.get("db", -80.0) + 50.0) / 35.0))
            if amp_now > 0.65 and (t_ms - self._last_listener_nod_ms) > 2000:
                self.gestures.schedule("micro_nod", t_ms, scale=0.30)
                self._last_listener_nod_ms = t_ms

        # 3 · Gesture track (keyword triggers, entry gestures, reactions)
        g = self.gestures.sample(t_ms)
        pose.lean += g["lean"]
        pose.head_nod += g["head_nod"]
        pose.bounce += g["bounce"]
        pose.sway += g["sway"]
        pose.squash += g["squash"]
        pose.brow += g["brow"]
        yaw_target += g["head_yaw"]

        # 3a · Head tilt: alternate direction on emphasis beats so the
        #      head traces ARCS instead of bouncing on one axis
        gesture_tilt = g["head_tilt"]
        if abs(gesture_tilt) > 1.0:
            gesture_tilt *= self._tilt_sign
            self._tilt_sign *= -1.0   # flip for next emphasis
        pose.head_tilt += gesture_tilt

        # 3b · Pose library: gesture → body pose with anti-ping-pong
        if self.pose_lib.has_poses:
            gesture_pose = self.gestures.active_pose(t_ms)
            if gesture_pose:
                if not self.pose_state.would_pingpong(gesture_pose):
                    self.pose_state.set_target(gesture_pose, displacement=0.7)
            elif is_speaking:
                # Rotate through natural talking poses
                next_sp = self._speaking_pose(t_ms)
                if not self.pose_state.would_pingpong(next_sp):
                    self.pose_state.set_target(next_sp, displacement=0.4)
            else:
                # Listener: cycle calm poses at half the speaker rate
                self.pose_state.set_target("listening", displacement=0.3)
            _from, _to, _bt = self.pose_state.step()
            pose.body_pose = _from
            pose.body_pose_to = _to
            pose.body_pose_blend = _bt

        # 4 · Mouth: coarticulated viseme glide × enveloped amplitude.
        #     The envelope steps EVERY frame so speech decays smoothly
        #     into silence instead of snapping shut.
        amp_level = self._env.step(fa.get("db", -80.0))
        if is_speaking and self._viseme_track is not None:
            wt_dict, jaw = self._viseme_track.weights_at(t_ms, energy)
            # Pick the dominant viseme for the sprite render path
            sorted_v = sorted(wt_dict.items(), key=lambda x: x[1], reverse=True)
            primary = sorted_v[0][0].value if sorted_v else "REST"
            secondary = sorted_v[1][0].value if len(sorted_v) > 1 else primary
            blend = sorted_v[1][1] if len(sorted_v) > 1 else 0.0
            pose.viseme = primary
            pose.viseme_to = secondary
            pose.viseme_blend = blend
            # Jaw controls mouth openness, gated by amplitude envelope
            pose.mouth_open = jaw * (0.22 + 0.78 * max(0.0, min(1.0, amp_level)))
            # Speech-beat nod: only on amplitude peaks (> 0.6)
            if self._viseme_track.word_started_within(t_ms) \
                    and fa.get("is_speaking", False) \
                    and amp_level > 0.6:
                pose.head_nod += 0.12 * energy
        else:
            pose.viseme, pose.viseme_to = "REST", "REST"
            pose.viseme_blend, pose.mouth_open = 0.0, 0.0

        # 5 · Blink (log-normal intervals + double-blink)
        pose.blink = self._blink_amount(t_ms, fps)

        # 5b · Phrase-boundary blink boost: speakers blink more at pauses
        if is_speaking and self._viseme_track is not None:
            if not self._viseme_track.word_started_within(t_ms, 300) \
                    and (t_ms - self._blink_start_ms) > 2000 \
                    and self._rng.random() < 0.003:  # ~10% per pause
                self._next_blink_ms = min(self._next_blink_ms, t_ms + 50)

        # 5c · Micro-saccades (eye jitter between fixation points)
        sdx, sdy = self._update_saccades(t_ms)
        pose.eye_dx = sdx
        pose.eye_dy = sdy

        # 6 · Critically-damped smoothing — TUNED rates:
        #     tilt/brow slowed for natural, less snappy feel
        pose.head_yaw = self._chase("head_yaw", yaw_target, 0.18)
        pose.lean = self._chase("lean", pose.lean, 0.22)
        pose.head_tilt = self._chase("head_tilt", pose.head_tilt, 0.18)  # was 0.25
        pose.brow = self._chase("brow", pose.brow, 0.20)  # was 0.3
        return pose

    def _chase(self, name: str, target: float, rate: float) -> float:
        cur = self._smooth[name]
        cur += (target - cur) * rate
        self._smooth[name] = cur
        return cur

    # ─── render ───────────────────────────────────────────

    def render(self, pose: PuppetPose) -> Tuple[Image.Image, float, float]:
        """Advance physics one frame and render.
        Returns (RGBA canvas, dx, dy) — dx/dy are the whole-body offsets
        the compositor adds to the camera position (canvas-relative)."""
        physics = self.engine.step_physics(pose)
        img = self.engine.render(pose, physics)
        # Convert puppet-space px offsets to canvas-fraction offsets
        rel = 1.0 / max(1, self.engine.height)
        return img, pose.sway * rel, pose.bounce * rel

    def neutral_still(self) -> Image.Image:
        """A single static neutral render (for corner cameos)."""
        return self.engine.render(PuppetPose())
