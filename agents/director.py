
JEEVidya V5 — Director Agent (Tier 3)
═════════════════════════════════════
Topic in → reviewed, revised, validated script out. Multi-pass:

  PASS 1  DRAFT     write against DIALOGUE_SCHEMA (the renderer's format)
  PASS 2  CRITIQUE  score the draft against a retention rubric
  PASS 3  REVISE    rewrite only if the rubric found real weaknesses

Beyond dialogue, the Director emits per-turn GESTURES (Tier 1 bone
engine triggers) and shot overrides (Tier 2 sequencer respects them),
then stamps the script with its Visual DNA so re-renders are identical.

Facts remain human-reviewed — the agent optimizes retention, never
invents constants (rubric enforces "omit if unsure").
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agents.llm import LLM
from config import prompts

RETENTION_RUBRIC = """You are a ruthless YouTube Shorts retention analyst.
Score this Gudiya & Chintu script 1-10 on EACH axis and list concrete fixes:

1. HOOK STRENGTH — does turn 1 create an information gap in <3 seconds?
   Specific numbers/stakes beat vague wonder.
2. ONE-IDEA-PER-TURN — any turn carrying two ideas or >15 spoken words?
3. QUESTION DENSITY — does curiosity get re-armed at least every 3 turns?
4. PAYOFF PLACEMENT — is the reveal ~60-70% in, with a reaction beat after?
5. CLIFFHANGER — does the final turn ask a question viewers will comment on?
6. FACT SAFETY — flag ANY number/formula that looks invented or off.

Return JSON only:
{"scores": {"hook": n, "one_idea": n, "questions": n, "payoff": n,
 "cliffhanger": n, "facts": n}, "verdict": "ship"|"revise",
 "fixes": ["specific fix", ...]}"""

REVISE_PROMPT = """Here is a draft script and a retention critique of it.
Rewrite the script applying EVERY fix while keeping what already works.
Keep the same JSON schema. Do not add facts you are not sure of.

DRAFT:
{draft}

CRITIQUE:
{critique}

Return ONLY the revised JSON script."""

GESTURE_PROMPT = """For each turn of this script, choose at most one gesture
for the speaker from: point, shrug, lean_in, nod, facepalm, jump, wave,
think, arms_open, none.  Trigger on meaning: "dekho/yeh/is" → point,
amazement → jump or arms_open, thinking → think, disbelief → facepalm.

Script turns:
{turns}

Return JSON only: {{"gestures": ["point", "none", ...]}} (one per turn,
same order)."""

VALID_GESTURES = {"point", "shrug", "lean_in", "nod", "facepalm", "jump",
                  "wave", "think", "arms_open", "none"}


class Director:
    """Multi-pass script generation with a retention self-critique loop."""

    def __init__(self, llm: Optional[LLM] = None):
        self.llm = llm or LLM()

    # ─── Public API ────────────────────────────────────────

    def write(self, topic: str, max_attempts: int = 3) -> Dict[str, Any]:
        """Full pipeline: draft → critique → (revise) → gestures → DNA."""
        from pipeline.scriptwriter import validate_dialogue

        draft = self._draft(topic, max_attempts)

        critique = self._critique(draft)
        if critique and critique.get("verdict") == "revise":
            fixes = critique.get("fixes", [])
            print(f"  [Director] Critique: revise ({len(fixes)} fixes)")
            revised = self._revise(draft, critique)
            if revised is not None:
                problems = validate_dialogue(revised)
                if not problems:
                    draft = revised
                else:
                    print(f"  [Director] Revision invalid ({problems}); "
                          "keeping draft")
        elif critique:
            print("  [Director] Critique: ship as-is "
                  f"(scores: {critique.get('scores')})")

        self._attach_gestures(draft)
        self._attach_dna(draft)
        draft["director"] = {"passes": 3 if critique else 1,
                             "critique": critique}
        return draft

    # ─── Pass 1: draft ─────────────────────────────────────

    def _draft(self, topic: str, max_attempts: int) -> Dict[str, Any]:
        from pipeline.scriptwriter import validate_dialogue
        last = "unknown"
        for attempt in range(1, max_attempts + 1):
            print(f"  [Director {attempt}/{max_attempts}] Drafting: {topic}")
            try:
                schema_hint = json.dumps(prompts.DIALOGUE_SCHEMA)
                dialogue = self.llm.generate_json(
                    prompts.GENERATION_PROMPT_TEMPLATE.format(topic=topic)
                    + f"\n\nJSON schema to match exactly:\n{schema_hint}",
                    system=prompts.SCRIPT_SYSTEM_PROMPT,
                    temperature=0.75)
                problems = validate_dialogue(dialogue)
                if problems:
                    raise ValueError("; ".join(problems))
                return dialogue
            except Exception as e:              # noqa: BLE001 — retry loop
                last = str(e)
                print(f"  [Director] draft attempt failed: {last}")
        raise RuntimeError(f"Director could not draft '{topic}': {last}")

    # ─── Pass 2: self-critique ─────────────────────────────

    def _critique(self, draft: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            result = self.llm.generate_json(
                "SCRIPT:\n" + json.dumps(draft, ensure_ascii=False),
                system=RETENTION_RUBRIC, temperature=0.2)
            if isinstance(result, dict) and "verdict" in result:
                return result
        except Exception as e:                  # noqa: BLE001 — optional pass
            print(f"  [Director] critique pass skipped: {e}")
        return None

    # ─── Pass 3: revision ──────────────────────────────────

    def _revise(self, draft: Dict[str, Any],
                critique: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            revised = self.llm.generate_json(
                REVISE_PROMPT.format(
                    draft=json.dumps(draft, ensure_ascii=False),
                    critique=json.dumps(critique, ensure_ascii=False)),
                system=prompts.SCRIPT_SYSTEM_PROMPT, temperature=0.6)
            if isinstance(revised, dict) and revised.get("turns"):
                return revised
        except Exception as e:                  # noqa: BLE001 — optional pass
            print(f"  [Director] revision pass skipped: {e}")
        return None

    # ─── Enrichment ────────────────────────────────────────

    def _attach_gestures(self, dialogue: Dict[str, Any]) -> None:
        """Tier 1 hook: one bone-engine gesture per turn."""
        turns = dialogue.get("turns", [])
        brief = [{"speaker": t.get("speaker"), "text": t.get("text", "")}
                 for t in turns]
        gestures: List[str] = []
        try:
            result = self.llm.generate_json(
                GESTURE_PROMPT.format(turns=json.dumps(brief,
                                                       ensure_ascii=False)),
                temperature=0.3)
            gestures = list(result.get("gestures", []))
        except Exception as e:                  # noqa: BLE001 — heuristic below
            print(f"  [Director] gesture pass offline ({e}); using heuristics")

        for i, turn in enumerate(turns):
            g = gestures[i] if i < len(gestures) else None
            if g not in VALID_GESTURES:
                g = _heuristic_gesture(turn)
            if g and g != "none":
                turn["gesture"] = g

    @staticmethod
    def _attach_dna(dialogue: Dict[str, Any]) -> None:
        """Stamp the genome so every future re-render is bit-identical."""
        from engine.visual_dna import VisualDNA
        dna = VisualDNA.from_dialogue(dialogue)
        dialogue["dna"] = dna.to_dict()


def _heuristic_gesture(turn: Dict[str, Any]) -> str:
    """Offline gesture picker — meaning-triggered, no LLM needed."""
    text = (turn.get("text") or "").lower()
    emotion = turn.get("emotion", "")
    if any(w in text for w in ("dekho", "yeh", "is ", "yahan")):
        return "point"
    if emotion == "amazed" or "?!" in text:
        return "jump"
    if emotion == "thinking":
        return "think"
    if emotion in ("confident", "enthusiastic") and "!" in text:
        return "arms_open"
    if "nahi" in text and "?" not in text:
        return "shrug"
    return "none"
