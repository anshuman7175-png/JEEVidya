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
import zlib
from typing import Dict, List, Optional, Tuple

from PIL import Image

from config import settings
from engine.affect import (AffectState, ListenerCoupling, coherence_audit,
                           map_channels, schedule_micro_expressions,
                           verify_state_continuity)
from engine.bone_engine import BoneEngine, PuppetPose
from engine.gestures import GestureTrack
from engine.pose_library import PoseLibrary, PoseState, DEFAULT_POSE
from engine.rig import Rig, has_rig  # noqa: F401  (has_rig re-exported)
from engine.visemes import (AmplitudeEnvelope, VisemeTrack,
                            mouth_openness_blend)

# Character name → dialogue speaker key
SPEAKER_OF = {"gudiya": "girl", "chintu": "boy"}

# Script emotion tags (EMOTION_BASELINE below) → the affect engine's
# vocabulary (engine.affect.EMOTION_TARGETS). The show's tags are a
# performance dialect; affect speaks valence/arousal, so the two
# vocabularies are reconciled ONCE, here, instead of at every read.
AFFECT_OF_EMOTION: Dict[str, str] = {
    "neutral": "neutral",
    "curious": "curious",
    "enthusiastic": "excited",
    "confident": "proud",
    "amazed": "surprised",
    "thinking": "thinking",
    "happy": "happy",
    "explaining": "serious",
    "dramatic": "excited",
}

# The affect matrix returns multipliers around 1.0 and biases around 0.0.
# These are the only places the puppet lets that state touch a channel,
# each scaled so affect COLOURS the performance instead of driving it.
_AFFECT_BROW_GAIN = 0.55       # brow_height bias → pose.brow
_AFFECT_TILT_GAIN = 14.0       # head_tilt_bias (±0.12) → degrees
_AFFECT_PULL_GAIN = 0.60       # mouth_pull bias → pose.mouth_pull
_AFFECT_PRESS_GAIN = 0.50      # mouth_press bias → pose.mouth_press
# Listeners get affect at a fraction of the speaker's authority (visual
# hierarchy: the speaker owns the frame), reusing the `soft` factor.

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


def _stable_seed(*parts) -> int:
    """A seed that is identical in every process (Law 4: bit-identical
    re-renders). `hash()` of a str is salted per interpreter, so it can
    never be used where determinism is a contract."""
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0xFFFF


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

        # ONE nervous system per character (§XVII): a continuous
        # valence/arousal state every channel below reads from, so the
        # whole body agrees about how the character feels.
        self.affect = AffectState(character, seed=_stable_seed(character))
        # Neutral channel biases until the first step() — never None, so
        # no read path needs a fallback branch.
        self.channels: Dict[str, float] = map_channels(
            self.affect.valence, self.affect.arousal)
        # Per-channel rendered tracks + the state trajectory, for the
        # affect-coherence audit (engine.affect.coherence_audit).
        self.channel_tracks: Dict[str, list] = {"brow_height": [],
                                                "gesture_gain": [],
                                                "lid_openness": []}
        # This actor's link to whoever is currently speaking. Owned per
        # listener (not per pair) so its lag buffer follows the character
        # even when the shot changes who is on screen.
        self.listen_link = ListenerCoupling()
        # The turn's own emotional target, re-asserted every frame before
        # the lagged speaker state is blended in — otherwise the coupling
        # would compound frame after frame and the listener would end up
        # simply BEING the speaker.
        self._affect_emotion = "neutral"
        # Breath phase accumulator (radians): affect scales the RATE, and a
        # rate change must bend the cycle, not jump it.
        self._breath_phase = 0.0

        # Attack/release amplitude follower: the jaw moves like muscle,
        # not like a VU meter (kills mouth flutter on sustained vowels)
        self._env = AmplitudeEnvelope(settings.FPS)

        # Speaking body-pose rotation (hands!): cycle natural talking
        # poses on phrase-length intervals instead of freezing in one
        self._speak_pose: str = "neutral"
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
        # Asymmetric mouth smoothing state: the jaw OPENS fast (muscle
        # snap) and CLOSES slow (relaxation). Without this filter the
        # per-frame viseme jaw target strobes at frame rate.
        self._mouth_smooth = 0.0

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

        # Affect target for the turn. Emotions TRANSITION through the
        # second-order filters inside AffectState — a cut never snaps the
        # nervous system, which is why the head keeps its momentum.
        prev_affect = self._affect_emotion
        self._affect_emotion = AFFECT_OF_EMOTION.get(emotion, "neutral")
        self.affect.set_emotion(self._affect_emotion)
        # Micro-expression grammar: the surprise onset BEFORE the smile.
        # Scheduled from the transition itself, so it is deterministic per
        # (character, turn) and survives a re-render bit-identically.
        self.affect.micro = schedule_micro_expressions(
            [(float(span.start_ms), prev_affect, self._affect_emotion)],
            seed=_stable_seed(self.character, self._current_turn_id))
        text = span.turn.get("text", "")
        is_question = text.rstrip().endswith(("?", "?!"))
        if is_my_turn:
            self.affect.push_event("question" if is_question else "reveal")
        else:
            # The listener is being addressed, which is its own arousal
            # bump — this is what stops the dead-eyed listener.
            self.affect.push_event("addressed")

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

    # ─── listener coupling (§XVII) ────────────────────────

    def track_speaker(self, t_ms: float, speaker: AffectState) -> None:
        """Blend the LAGGED speaker state into this (listening) actor.

        Call once per frame, before `pose_at`, for every actor that is not
        speaking. The turn's own emotional target is re-asserted first so
        the coupling stays a 0.4-gain blend forever instead of compounding
        into "the listener simply becomes the speaker" after ten frames.
        """
        if speaker is self.affect:
            return
        self.listen_link.observe(t_ms, speaker)
        self.affect.set_emotion(self._affect_emotion)
        self.listen_link.drive(t_ms, self.affect)

    # ─── affect QC (§XVII) ────────────────────────────────

    def affect_violations(self) -> List[str]:
        """State continuity + affect coherence over the whole performance.

        Three failures are representable and all three are caught here: a
        snapping nervous system, a face that contradicts its own state,
        and — the silent one — a channel that never moved at all because
        its wiring was refactored away.
        """
        out = list(verify_state_continuity(self.affect.trajectory))
        out += coherence_audit(self.affect.trajectory, self.channel_tracks)
        for name, track in self.channel_tracks.items():
            if len(track) >= 8 and (max(track) - min(track)) < 1e-6:
                out.append(f"affect: channel '{name}' never moved — its "
                           f"wiring into the pose is dead")
        return [f"{self.character}: {v}" for v in out]

    # ─── blink scheduler (log-normal + double-blink) ──────

    def _next_blink_interval(self) -> float:
        """Log-normal distribution: median ~3s, range 1.5–7s.
        Real human blink statistics, not uniform random.

        Divided by affect's `blink_rate_mult`: an aroused character blinks
        up to 2.2× as often, a calm one barely more than half as often —
        one of the strongest unconscious arousal cues there is."""
        base = max(1400.0, min(7500.0, math.exp(self._rng.gauss(8.0, 0.4))))
        rate = max(0.1, self.channels["blink_rate_mult"])
        return max(650.0, min(9000.0, base / rate))

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
        The brain reads motionless eyes as 'doll' — this breaks that.

        Fixation length scales with affect's `gaze_hold_mult`: a calm,
        positive character holds its gaze; an activated one checks away
        more often."""
        if t_ms >= self._next_saccade_ms:
            # New fixation point: small random offset (max ±3px head-local)
            self._saccade_dx = self._rng.uniform(-2.5, 2.5)
            self._saccade_dy = self._rng.uniform(-1.5, 1.5)
            hold = max(0.2, self.channels["gaze_hold_mult"])
            self._next_saccade_ms = t_ms + self._rng.uniform(300, 1500) * hold
        return self._saccade_dx, self._saccade_dy

    # ─── speaking pose rotation ──────────────────────────

    _SPEAK_POSE_ROTATION = ("explaining", "presenting", "both_hands_wide",
                            "hand_on_heart", "counting")

    def _speaking_pose(self, t_ms: float) -> str:
        """Deterministically cycle natural talking poses every 4.5–7.5 s
        so the hands gesticulate through the whole turn (only poses the
        character actually has are eligible). Rotation is slow on purpose:
        combined with keyword/beat gestures it used to produce a pose swap
        every 1–2 s, which reads as frantic channel-surfing."""
        if t_ms >= self._next_speak_pose_ms:
            options = [p for p in self._SPEAK_POSE_ROTATION
                       if p in self.pose_lib.pose_names
                       and p != self._speak_pose]
            if options:
                self._speak_pose = options[self._rng.randrange(len(options))]
            self._next_speak_pose_ms = t_ms + self._rng.uniform(6000, 10000)
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

        # 0 · Affect FIRST: the nervous system advances before any channel
        #     reads it, so every channel in this frame sees ONE consistent
        #     emotional state (that agreement is the whole point of §XVII).
        #     Loudness pushes arousal; the listener contributes none of its
        #     own (it is coupled to the speaker in track_speaker instead).
        rms_norm = 0.0
        if is_speaking and fa.get("is_speaking", False):
            rms_norm = max(0.0, min(1.0, (fa.get("db", -80.0) + 50.0) / 35.0))
        snap = self.affect.step(rms_norm, float(fa.get("f0_excursion", 0.0)),
                                dt=1.0 / max(1, fps))
        ch = self.affect.channels_with_micro(snap, t_ms)
        self.channels = ch

        # 1 · Emotion baseline (listeners get a MUCH softened version)
        soft = 1.0 if is_speaking else 0.35   # was 0.55 → quieter listener
        pose.lean += base.get("lean", 0.0) * soft
        pose.head_tilt += base.get("head_tilt", 0.0) * soft
        pose.head_nod += base.get("head_nod", 0.0) * soft
        pose.brow += base.get("brow", 0.0) * soft
        yaw_target = base.get("head_yaw", 0.0) * soft + look_dir * 0.45

        # 1b · Question brow hold: sustained raise for question turns
        pose.brow += self._question_brow

        # 1c · Affect biases. Brow, head-tilt and the mouth's expressive
        #      axes are the face's own emotional colour; the listener gets
        #      them at the same reduced authority as everything else.
        pose.brow += ch["brow_height"] * _AFFECT_BROW_GAIN * soft
        pose.head_tilt += ch["head_tilt_bias"] * _AFFECT_TILT_GAIN * soft
        pose.mouth_pull = ch["mouth_pull"] * _AFFECT_PULL_GAIN
        pose.mouth_press = ch["mouth_press"] * _AFFECT_PRESS_GAIN
        pose.lid = ch["lid_openness"]

        # 2 · Idle breathing & speaking bounce / sway (continuous)
        f = global_frame
        # Independent phase per actor prevents synchronized breathing
        self._breath_phase += 0.02
        if is_speaking:
            # Speaking bounce: speech energy drives frequency and amplitude
            pose.bounce += math.sin(f * settings.BODY_SPEAK_SPEED) \
                * settings.BODY_SPEAK_BOUNCE * energy
            pose.sway += math.sin(f * 0.25) * settings.BODY_SPEAK_SWAY * energy
            pose.squash += math.sin(f * 0.3) \
                * settings.BODY_SPEAK_SCALE_PULSE * energy
            # Speech-energy brow lift: excited speech raises brows
            amp_norm = max(0.0, min(1.0, (fa.get("db", -80.0) + 50.0) / 35.0))
            pose.brow += amp_norm * 0.28 * energy
        else:
            # Breathing cycle
            pose.bounce += math.sin(self._breath_phase) \
                * settings.BODY_BREATHE_AMPLITUDE * energy
            pose.squash += math.sin(self._breath_phase) \
                * 0.01 * energy
            # Attentive listener nodding: triggered by speaker's amplitude peaks
            amp_now = max(0.0, min(1.0, (fa.get("db", -80.0) + 50.0) / 35.0))
            if amp_now > 0.65 and (t_ms - self._last_listener_nod_ms) > 2000:
                self.gestures.schedule("micro_nod", t_ms, scale=0.30)
                self._last_listener_nod_ms = t_ms

        # 3 · Gesture track (keyword triggers, entry gestures, reactions),
        #     scaled by affect's gesture_gain: the SAME gesture library
        #     reads as reserved or animated depending on how the character
        #     feels, instead of needing a second set of "excited" gestures.
        g = self.gestures.sample(t_ms)
        gain = ch.get("gesture_gain", 1.0)
        pose.lean += g["lean"] * gain
        pose.head_nod += g["head_nod"] * gain
        pose.bounce += g["bounce"] * gain
        pose.sway += g["sway"] * gain
        pose.squash += g["squash"] * gain
        pose.brow += g["brow"] * gain
        yaw_target = g["head_yaw"] * gain

        # 3a · Head tilt: alternate direction on emphasis beats so the
        #      head traces ARCS instead of bouncing on one axis
        gesture_tilt = g["head_tilt"] * gain
        if abs(gesture_tilt) > 1.0:
            gesture_tilt *= self._tilt_sign
            self._tilt_sign *= -1.0   # flip for next emphasis
        pose.head_tilt += gesture_tilt

        # 3b · Pose library: gesture → body pose with anti-ping-pong
        if self.pose_lib.has_poses:
            gesture_pose = self.gestures.active_pose(t_ms)
            if gesture_pose and gesture_pose in self.pose_lib.pose_names and gesture_pose in self.rig.poses:
                if not self.pose_state.would_pingpong(gesture_pose):
                    self.pose_state.set_target(gesture_pose, displacement=0.7)
            elif is_speaking:
                # Rotate through natural talking poses on phrase-length intervals
                next_sp = self._speaking_pose(t_ms)
                if next_sp in self.pose_lib.pose_names and next_sp in self.rig.poses:
                    if not self.pose_state.would_pingpong(next_sp):
                        self.pose_state.set_target(next_sp, displacement=0.4)
            else:
                # Listener: keep calm neutral pose
                if "neutral" in self.pose_lib.pose_names and "neutral" in self.rig.poses:
                    self.pose_state.set_target("neutral", displacement=0.3)
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
            # Jaw controls mouth openness, gated by amplitude envelope,
            # then run through an asymmetric attack/release filter so the
            # mouth moves at MUSCLE rate (open ~50ms, close ~130ms) instead
            # of strobing to a new target every single frame.
            raw_open = jaw * (0.22 + 0.78 * max(0.0, min(1.0, amp_level)))
            k = 0.55 if raw_open > self._mouth_smooth else 0.28
            self._mouth_smooth += (raw_open - self._mouth_smooth) * k
            pose.mouth_open = self._mouth_smooth
            # Speech-beat nod: only on amplitude peaks (> 0.6)
            if self._viseme_track.word_started_within(t_ms) \
                    and fa.get("is_speaking", False) \
                    and amp_level > 0.6:
                pose.head_nod += 0.12 * energy
        else:
            pose.viseme, pose.viseme_to = "REST", "REST"
            # Release toward closed instead of snapping shut mid-frame
            self._mouth_smooth *= 0.72
            if self._mouth_smooth < 0.01:
                self._mouth_smooth = 0.0
            pose.viseme_blend, pose.mouth_open = 0.0, self._mouth_smooth

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

        # 6 · Critically-damped smoothing — organic, film-grade rate:
        #     tilt/brow/lean slowed for natural, organic feel (no snapping)
        pose.head_yaw = self._chase("head_yaw", yaw_target, 0.12)
        pose.lean = self._chase("lean", pose.lean, 0.14)
        pose.head_tilt = self._chase("head_tilt", pose.head_tilt, 0.12)
        pose.brow = self._chase("brow", pose.brow, 0.14)

        # 7 · Record what was actually RENDERED (not what the matrix said)
        #     so the coherence audit can catch a face whose expression
        #     disagrees with its own emotional state.
        self.channel_tracks["brow_height"].append(pose.brow)
        self.channel_tracks["gesture_gain"].append(gain)
        self.channel_tracks["lid_openness"].append(pose.lid)
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
