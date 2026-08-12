"""
Affect Engine wiring tests (Singularity Plan, Part XVII).

`engine/affect.py` is only worth anything if its ONE state actually
reaches every channel. These tests assert the wiring, not the theory:

  • the channel matrix agrees with the physiology it claims to model
  • emotions TRANSITION (bounded state derivative) — a cut never snaps
  • micro-expressions stay brief and sub-threshold
  • a listener tracks the speaker with lag and NEVER becomes the speaker
  • the state reaches the pose: brow, mouth pull/press, lids, gesture gain
  • arousal really does raise the blink rate and shorten fixations
  • a whole rendered performance passes its own §XVII QC gates

The actor is driven against the synthetic v3 rig from `conftest.py`, so
the affect→pose→render path is exercised end to end without any shipped
character assets.
"""
from __future__ import annotations

import pytest

from config import settings
from engine.affect import (AffectState, ListenerCoupling, EMOTION_TARGETS,
                           map_channels, schedule_micro_expressions,
                           verify_state_continuity)
from pipeline.puppet import AFFECT_OF_EMOTION, PuppetActor
from pipeline.timeline import TurnSpan, WordEvent

FPS = settings.FPS
LOUD = {"db": -18.0, "mouth_state": 2, "is_speaking": True,
        "f0_excursion": 0.4}
QUIET = {"db": -80.0, "mouth_state": 0, "is_speaking": False}


# ═══════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════

TEXT = "Dekho yeh force kaise kaam karta hai"


def _span(start_ms: int, dur_ms: int, turn_id: int) -> TurnSpan:
    words, t = [], start_ms
    for w in TEXT.split():
        words.append(WordEvent(w, t, t + 260))
        t += 300
    return TurnSpan(turn={"turn_id": turn_id, "text": TEXT, "speaker": "girl"},
                    start_ms=start_ms, end_ms=start_ms + dur_ms,
                    start_frame=round(start_ms * FPS / 1000),
                    end_frame=round((start_ms + dur_ms) * FPS / 1000),
                    words=words)


def _perform(actor: PuppetActor, plan, speaking: bool = True) -> list:
    """Drive a performance arc — `plan` is [(emotion, seconds), …] — and
    return every pose. Multi-turn on purpose: one flat emotion gives the
    affect trajectory almost no range, and a QC gate that only ever sees
    a flat driver is not being tested at all."""
    audio = LOUD if speaking else QUIET
    out, frame, t_ms = [], 0, 0.0
    for turn_id, (emotion, secs) in enumerate(plan):
        actor.begin_span(_span(int(t_ms), int(secs * 1000), turn_id),
                         is_my_turn=speaking, emotion=emotion)
        for _ in range(int(secs * FPS)):
            out.append(actor.pose_at(frame, frame * 1000.0 / FPS, speaking,
                                     emotion, audio, look_dir=0.0, fps=FPS))
            frame += 1
        t_ms += secs * 1000.0
    return out


# A performance arc with real emotional range, the way a scripted turn
# list actually arrives from the Director agent.
ARC = [("neutral", 2), ("dramatic", 3), ("thinking", 3), ("enthusiastic", 3)]


# ═══════════════════════════════════════════
# the matrix
# ═══════════════════════════════════════════

def test_channel_matrix_matches_the_physiology_it_models():
    """One number pair, every channel: activation blinks faster, holds
    gaze shorter, gestures bigger, breathes faster and shallower."""
    calm = map_channels(*EMOTION_TARGETS["sad"])
    hot = map_channels(*EMOTION_TARGETS["excited"])
    assert hot["blink_rate_mult"] > calm["blink_rate_mult"]
    assert hot["gaze_hold_mult"] < calm["gaze_hold_mult"]
    assert hot["gesture_gain"] > calm["gesture_gain"]
    assert hot["breath_rate_mult"] > calm["breath_rate_mult"]
    assert hot["breath_depth_mult"] < calm["breath_depth_mult"]
    # Valence owns the mouth's expressive axis, not arousal.
    assert hot["mouth_pull"] > calm["mouth_pull"]


def test_every_channel_stays_inside_its_declared_bounds():
    for v in (-1.0, -0.4, 0.0, 0.5, 1.0):
        for a in (0.0, 0.3, 0.7, 1.0):
            ch = map_channels(v, a)
            for name, value in ch.items():
                lo, hi = _bounds(name)
                assert lo - 1e-9 <= value <= hi + 1e-9, name


def _bounds(name: str):
    from engine.affect import CHANNEL_MATRIX
    _, _, _, lo, hi = CHANNEL_MATRIX[name]
    return lo, hi


# ═══════════════════════════════════════════
# state dynamics
# ═══════════════════════════════════════════

def test_emotions_transition_instead_of_snapping():
    st = AffectState("gudiya", seed=7)
    st.set_emotion("excited")
    for _ in range(2 * FPS):
        st.step(0.8, 0.5, dt=1.0 / FPS)
    st.set_emotion("sad")                 # the hardest cut there is
    for _ in range(2 * FPS):
        st.step(0.0, 0.0, dt=1.0 / FPS)
    assert verify_state_continuity(st.trajectory, fps=FPS) == []
    assert st.valence < 0.3               # it really did travel


def test_state_is_deterministic_per_seed():
    def traj():
        st = AffectState("chintu", seed=11)
        st.set_emotion("curious")
        for _ in range(90):
            st.step(0.5, 0.2, dt=1.0 / FPS)
        return st.trajectory
    assert traj() == traj()


def test_dialogue_events_decay_instead_of_latching():
    st = AffectState("gudiya")
    st.set_emotion("neutral")
    st.push_event("reveal")
    peak = max(st.step(0.0, 0.0, dt=1.0 / FPS).arousal for _ in range(30))
    for _ in range(4 * FPS):
        snap = st.step(0.0, 0.0, dt=1.0 / FPS)
    assert peak > EMOTION_TARGETS["neutral"][1]
    assert snap.arousal == pytest.approx(EMOTION_TARGETS["neutral"][1],
                                         abs=0.03)


def test_micro_expressions_are_brief_and_subthreshold():
    micro = schedule_micro_expressions(
        [(1000.0, "neutral", "surprised"), (3000.0, "sad", "happy")], seed=3)
    assert micro, "a rising-arousal transition must fire a flicker"
    for m in micro:
        assert 120.0 <= m.duration_ms <= 200.0
        assert abs(m.amplitude) <= 0.30            # sub-threshold
        assert m.value_at(m.start_ms - 1.0) == 0.0
        assert m.value_at(m.start_ms + m.duration_ms + 1.0) == 0.0
        assert abs(m.value_at(m.start_ms + m.duration_ms / 2)) > 0.0


def test_micro_expression_schedule_is_deterministic():
    args = ([(500.0, "neutral", "excited")],)
    assert (schedule_micro_expressions(*args, seed=5)
            == schedule_micro_expressions(*args, seed=5))


# ═══════════════════════════════════════════
# listener coupling
# ═══════════════════════════════════════════

def test_listener_coupling_respects_its_lag():
    link = ListenerCoupling(lag_ms=400.0, gain=0.4)
    speaker, listener = AffectState("a"), AffectState("b")
    speaker.set_emotion("excited")
    before = listener._target
    for i in range(int(0.2 * FPS)):          # only 200 ms of history
        speaker.step(0.9, 0.5, dt=1.0 / FPS)
        link.observe(i * 1000.0 / FPS, speaker)
        link.drive(i * 1000.0 / FPS, listener)
    assert listener._target == before, "coupling fired before its lag"


def test_listener_never_simply_becomes_the_speaker(v3_rig, char_name):
    """track_speaker re-asserts the turn's own target every frame, so the
    0.4 gain stays a blend forever instead of compounding to 1.0."""
    speaker = PuppetActor(char_name)
    listener = PuppetActor(char_name)
    speaker.affect.set_emotion("excited")
    listener._affect_emotion = "neutral"
    for i in range(6 * FPS):
        t = i * 1000.0 / FPS
        speaker.affect.step(0.9, 0.5, dt=1.0 / FPS)
        listener.track_speaker(t, speaker.affect)
        listener.affect.step(0.0, 0.0, dt=1.0 / FPS)

    neutral_a = EMOTION_TARGETS["neutral"][1]
    excited_a = EMOTION_TARGETS["excited"][1]
    # It moved toward the speaker …
    assert listener.affect.arousal > neutral_a + 0.05
    # … but stopped well short of being them.
    assert listener.affect.arousal < neutral_a + 0.6 * (excited_a - neutral_a)


def test_track_speaker_ignores_its_own_state(v3_rig, char_name):
    actor = PuppetActor(char_name)
    actor.track_speaker(100.0, actor.affect)
    assert actor.listen_link._buffer == []


# ═══════════════════════════════════════════
# affect → pose (the wiring that matters)
# ═══════════════════════════════════════════

def test_affect_reaches_every_pose_channel(v3_rig, char_name):
    actor = PuppetActor(char_name)
    poses = _perform(actor, [("enthusiastic", 3)])
    assert actor.channels["gesture_gain"] > 1.0
    for name, track in actor.channel_tracks.items():
        assert len(track) == len(poses), name
        assert max(track) - min(track) > 1e-6, f"{name} never moved"
    last = poses[-1]
    assert last.mouth_pull > 0.0        # positive valence pulls the corners
    assert 0.3 <= last.lid <= 1.0
    assert last.brow > 0.0


def test_pose_channels_stay_clamped(v3_rig, char_name):
    actor = PuppetActor(char_name)
    for pose in _perform(actor, ARC):
        assert -1.0 <= pose.mouth_pull <= 1.0
        assert 0.0 <= pose.mouth_press <= 1.0
        assert 0.3 <= pose.lid <= 1.0


def test_arousal_raises_the_blink_rate(v3_rig, char_name):
    """The strongest unconscious arousal cue there is — and it comes out
    of the same state that moved the brow.

    Measured while LISTENING: a speaker's own loudness pushes arousal
    hard enough to saturate both takes, which would hide the effect the
    matrix is responsible for.
    """
    assert _count_blinks(char_name, "enthusiastic") > \
        _count_blinks(char_name, "thinking")


def _count_blinks(char_name: str, emotion: str) -> int:
    """Rising edges of the rendered blink channel over a 30 s take."""
    poses = _perform(PuppetActor(char_name), [(emotion, 30)], speaking=False)
    prev, n = 0.0, 0
    for pose in poses:
        if prev <= 0.0 < pose.blink:
            n += 1
        prev = pose.blink
    return n


def test_arousal_shortens_gaze_fixations(v3_rig, char_name):
    def saccades(emotion: str) -> int:
        actor = PuppetActor(char_name)
        n, last = 0, (actor._saccade_dx, actor._saccade_dy)
        for pose in _perform(actor, [(emotion, 30)], speaking=False):
            now = (pose.eye_dx, pose.eye_dy)
            if now != last:
                n += 1
                last = now
        return n
    assert saccades("enthusiastic") > saccades("thinking")


def test_affect_scales_the_same_gesture_library(v3_rig, char_name,
                                                monkeypatch):
    """One library, two energies: the gain is what makes a reserved
    character and an animated one out of the same keyframes."""
    const = {k: 1.0 for k in ("lean", "head_nod", "head_yaw", "head_tilt",
                              "bounce", "sway", "squash", "brow")}
    calm = PuppetActor(char_name)
    hot = PuppetActor(char_name)
    for actor in (calm, hot):
        monkeypatch.setattr(actor.gestures, "sample", lambda t: dict(const))
    _perform(calm, [("thinking", 5)], speaking=False)
    _perform(hot, [("dramatic", 5)], speaking=False)
    assert hot.channels["gesture_gain"] > calm.channels["gesture_gain"] * 1.15
    assert max(hot.channel_tracks["gesture_gain"]) > \
        max(calm.channel_tracks["gesture_gain"])


def test_emotion_dialect_maps_onto_the_affect_vocabulary():
    """The show's tags and the affect vocabulary are reconciled ONCE."""
    for tag, affect_name in AFFECT_OF_EMOTION.items():
        assert affect_name in EMOTION_TARGETS, tag


# ═══════════════════════════════════════════
# the §XVII QC gates, over a real render
# ═══════════════════════════════════════════

@pytest.mark.parametrize("speaking", [True, False])
def test_rendered_performance_passes_its_own_affect_gates(v3_rig, char_name,
                                                          speaking):
    """Both halves of a two-shot: the speaker's own gesture track and the
    listener's quieter one must each still agree with the state."""
    actor = PuppetActor(char_name)
    _perform(actor, ARC, speaking=speaking)
    assert actor.affect_violations() == []


def test_a_dead_channel_is_caught(v3_rig, char_name):
    """The silent failure §XVII exists to catch: wiring refactored away,
    so a channel renders a constant and nobody notices."""
    actor = PuppetActor(char_name)
    _perform(actor, ARC)
    n = len(actor.channel_tracks["brow_height"])
    actor.channel_tracks["brow_height"] = [0.25] * n
    violations = actor.affect_violations()
    assert any("brow_height" in v for v in violations)


def test_incoherent_face_fails_the_coherence_audit(v3_rig, char_name):
    """A face that contradicts its own state: brows falling while the
    nervous system is winding up."""
    actor = PuppetActor(char_name)
    _perform(actor, ARC)
    n = len(actor.channel_tracks["brow_height"])
    actor.channel_tracks["brow_height"] = [1.0 - i / n for i in range(n)]
    assert any("coherence" in v and "brow_height" in v
               for v in actor.affect_violations())
