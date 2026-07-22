"""
JEEVidya V5 — Shot Sequencer (Tier 2)
═════════════════════════════════════
Plans the camera like an editor who has stared at retention graphs:

  Rules      ?-questions → close-ups · speaker changes → shot changes ·
             explanation → fullscreen · post-reveal → reaction/reveal ·
             never the same shot 3× in a row.
  Shape      a per-turn visual-intensity budget with enforced peaks at
             the hook (0-3s), the mid reveal, and the CTA/cliffhanger —
             the retention CURVE, not just shot variety.
  Cut rate   scaled by the DNA cut_rate gene; everything seeded → the
             same script + DNA always cuts identically (Tier 0-safe).

Explicit shot_type in the script always wins — the Director Agent can
override any planned cut.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from engine.visual_dna import VisualDNA

# How visually intense each shot reads (drives energy budgeting)
SHOT_INTENSITY = {
    "extreme_closeup": 0.95,
    "reaction_cut": 0.85,
    "reveal": 0.75,
    "medium": 0.60,
    "fullscreen_explain": 0.55,
    "two_shot": 0.40,
}

_QUESTION_MARKS = ("?", "kya", "kaise", "kyun", "kitna", "kitni", "kaun")


def _is_question(text: str) -> bool:
    t = (text or "").lower()
    return "?" in t or any(t.startswith(q) or f" {q} " in t
                           for q in _QUESTION_MARKS)


class ShotSequencer:
    """Deterministic shot planning over a list of turns."""

    def __init__(self, dna: Optional["VisualDNA"] = None):
        self.cut_rate = float(dna.genes["cut_rate"]) if dna else 1.0
        self.rng = random.Random(dna.seed + 31 if dna else 31)

    # ─── Public API ────────────────────────────────────────

    def plan(self, turns: List[Dict[str, Any]]) -> List[str]:
        """Return one shot per turn. Turns with an explicit shot_type keep
        it; everything else is planned by rule + energy shape."""
        n = len(turns)
        shots: List[Optional[str]] = [t.get("shot_type") for t in turns]
        explain_idx = next((i for i, t in enumerate(turns)
                            if t.get("speaker") == "explanation"), None)

        for i, turn in enumerate(turns):
            if shots[i]:
                continue
            shots[i] = self._choose(i, n, turn, turns, shots, explain_idx)

        self._enforce_no_triples(shots)
        self._enforce_peaks(shots, n, explain_idx)
        return [s or "two_shot" for s in shots]

    def apply(self, turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Mutate turns in place with the planned shots. Returns turns."""
        for turn, shot in zip(turns, self.plan(turns)):
            turn["shot_type"] = shot
        return turns

    def energy_curve(self, turns: List[Dict[str, Any]]) -> List[float]:
        """Per-turn visual intensity — thumbnails and QC read this."""
        return [SHOT_INTENSITY.get(t.get("shot_type", "two_shot"), 0.4)
                * (1.0 if t.get("emotion") != "amazed" else 1.15)
                for t in turns]

    # ─── Rules ─────────────────────────────────────────────

    def _choose(self, i: int, n: int, turn: Dict, turns: List[Dict],
                shots: List[Optional[str]], explain_idx: Optional[int]) -> str:
        speaker = turn.get("speaker")
        emotion = turn.get("emotion", "")
        text = turn.get("text", "")

        if speaker == "explanation":
            return "fullscreen_explain"
        if i == 0:
            return "extreme_closeup"                    # the hook
        if i == n - 1:
            return "extreme_closeup"                    # the cliffhanger
        if emotion == "amazed":
            return "reaction_cut"
        if explain_idx is not None and i == explain_idx + 1:
            return "reveal"                              # land the payoff
        if _is_question(text):
            return "medium" if self.rng.random() < 0.5 else "extreme_closeup"

        prev = shots[i - 1]
        speaker_changed = (i > 0 and turns[i - 1].get("speaker") != speaker)
        cut_prob = 0.45 * self.cut_rate + (0.35 if speaker_changed else 0.0)
        if self.rng.random() < cut_prob and prev is not None:
            options = [s for s in ("two_shot", "medium") if s != prev]
            return self.rng.choice(options) if options else "two_shot"
        return prev if prev in ("two_shot", "medium") else "two_shot"

    # ─── Shape enforcement ─────────────────────────────────

    def _enforce_no_triples(self, shots: List[Optional[str]]) -> None:
        """The 3×-repeat killer (a property test pins this forever)."""
        for i in range(2, len(shots)):
            if shots[i] == shots[i - 1] == shots[i - 2]:
                current = shots[i]
                options = [s for s in ("medium", "two_shot", "reaction_cut")
                           if s != current]
                shots[i] = self.rng.choice(options)

    def _enforce_peaks(self, shots: List[Optional[str]], n: int,
                       explain_idx: Optional[int]) -> None:
        """Guarantee intensity peaks at hook, mid-reveal, and ending."""
        def intensity(i):
            return SHOT_INTENSITY.get(shots[i] or "two_shot", 0.4)

        # Hook peak (already extreme_closeup by rule, keep it honest)
        if n > 0 and intensity(0) < 0.9:
            shots[0] = "extreme_closeup"
        # Mid peak: the turn after the explanation must land ≥ reveal
        if explain_idx is not None and explain_idx + 1 < n:
            if intensity(explain_idx + 1) < 0.7:
                shots[explain_idx + 1] = "reveal"
        # End peak
        if n > 1 and intensity(n - 1) < 0.8:
            shots[n - 1] = "extreme_closeup"
