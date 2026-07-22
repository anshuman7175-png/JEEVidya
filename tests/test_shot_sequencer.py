"""Tier 2 — Shot sequencer: the retention-shape properties, pinned."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.shot_sequencer import SHOT_INTENSITY, ShotSequencer
from engine.visual_dna import VisualDNA


def _turns(n=10, with_explain=True):
    random.seed(4)
    turns = []
    for i in range(n):
        if with_explain and i == n // 2:
            turns.append({"speaker": "explanation", "emotion": "neutral"})
        else:
            turns.append({
                "speaker": "girl" if i % 2 == 0 else "boy",
                "text": f"line {i}" + ("?" if i % 3 == 0 else "!"),
                "emotion": "amazed" if i == n // 2 + 1 else "curious",
            })
    return turns


def test_never_three_in_a_row():
    """The property the plan explicitly demands, over many seeds."""
    for seed_title in ("A", "B", "C", "gravity", "optics", "waves", "heat"):
        seq = ShotSequencer(VisualDNA.from_title(seed_title))
        shots = seq.plan(_turns(16))
        for i in range(2, len(shots)):
            assert not (shots[i] == shots[i - 1] == shots[i - 2]), \
                f"triple {shots[i]} at {i} for '{seed_title}': {shots}"


def test_deterministic_per_dna():
    dna = VisualDNA.from_title("Doppler")
    assert ShotSequencer(dna).plan(_turns()) == ShotSequencer(dna).plan(_turns())


def test_hook_and_cliffhanger_peaks():
    shots = ShotSequencer(VisualDNA.from_title("X")).plan(_turns(12))
    assert SHOT_INTENSITY[shots[0]] >= 0.9      # hook peak
    assert SHOT_INTENSITY[shots[-1]] >= 0.8     # ending peak


def test_explanation_gets_fullscreen():
    turns = _turns(9)
    shots = ShotSequencer(VisualDNA.from_title("Y")).plan(turns)
    idx = next(i for i, t in enumerate(turns)
               if t["speaker"] == "explanation")
    assert shots[idx] == "fullscreen_explain"
    # the reveal after explanation must be intense
    assert SHOT_INTENSITY[shots[idx + 1]] >= 0.7


def test_explicit_shot_wins():
    turns = _turns(8)
    turns[3]["shot_type"] = "reveal"
    shots = ShotSequencer(VisualDNA.from_title("Z")).plan(turns)
    assert shots[3] == "reveal"


def test_apply_mutates_turns():
    turns = _turns(8)
    ShotSequencer(VisualDNA.from_title("W")).apply(turns)
    assert all(t.get("shot_type") for t in turns)
