"""
JEEVidya V5 — Director Agent (Scriptwriter)
═══════════════════════════════════════════
Topic in → validated Gudiya & Chintu dialogue JSON out.

The V2 module referenced prompt constants that never existed and targeted a
dead storyboard schema — it crashed on import use. V5 targets DIALOGUE_SCHEMA
(the format the whole pipeline actually renders), uses Gemini's free-tier
flash model with structured output, and self-heals invalid responses instead
of rejecting them.
"""
import importlib
import json
import os
import re
from typing import Dict, Any, List, Optional, Tuple

from config import prompts


def _load_genai():
    """Lazy-load the Gemini SDK.

    The raw-script formatter path (parse_raw_script) is pure Python and
    must import cleanly on machines without google-genai installed —
    only the Director Agent (ScriptWriter) actually needs the SDK.
    """
    genai = importlib.import_module("google.genai")
    types = importlib.import_module("google.genai.types")
    return genai, types

VALID_SPEAKERS = {"girl", "boy", "explanation"}
VALID_EMOTIONS = {"curious", "enthusiastic", "confident", "amazed",
                  "thinking", "happy", "explaining", "dramatic"}
VALID_SHOTS = {"extreme_closeup", "two_shot", "medium",
               "fullscreen_explain", "reaction_cut", "reveal"}


def validate_dialogue(dialogue: Dict[str, Any]) -> List[str]:
    """
    Validate + self-heal a dialogue dict IN PLACE.
    Returns a list of fatal problems (empty list = usable).
    Non-fatal issues (bad emotion/shot enums, missing ids) are auto-repaired.
    """
    problems: List[str] = []

    if not dialogue.get("title"):
        problems.append("missing 'title'")
    turns = dialogue.get("turns")
    if not isinstance(turns, list) or not turns:
        problems.append("missing or empty 'turns'")
        return problems

    for i, turn in enumerate(turns):
        # Heal: sequential turn ids
        turn["turn_id"] = i + 1

        speaker = turn.get("speaker")
        if speaker not in VALID_SPEAKERS:
            problems.append(f"turn {i + 1}: invalid speaker '{speaker}'")
            continue

        if speaker == "explanation":
            turn.setdefault("duration_seconds", 4.0)
            turn.setdefault("pause_after", 1.0)
            turn.setdefault("visual_elements", [])
        else:
            if not str(turn.get("text", "")).strip():
                problems.append(f"turn {i + 1}: speaking turn with empty text")

        # Heal: clamp enums to valid values instead of failing
        if turn.get("emotion") not in VALID_EMOTIONS:
            turn["emotion"] = "neutral" if speaker == "explanation" else "explaining"
        if turn.get("shot_type") not in VALID_SHOTS:
            turn.pop("shot_type", None)  # let the shot sequencer decide

    return problems


class ScriptWriter:
    """Director Agent: generates validated dialogue scripts via Gemini."""

    def __init__(self, api_key: Optional[str] = None,
                 model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be set in environment variables")

        self.client = genai.Client(api_key=self.api_key)
        # Free-tier friendly, fast, supports structured output
        self.model_name = model_name

    def generate_dialogue(self, topic: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Generate a Gudiya & Chintu dialogue script for a topic.
        Structured output + self-healing validation loop.
        """
        last_error = "unknown"
        for attempt in range(1, max_retries + 1):
            try:
                print(f"[Director {attempt}/{max_retries}] Writing script for: '{topic}'...")

                config = types.GenerateContentConfig(
                    system_instruction=prompts.SCRIPT_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=prompts.DIALOGUE_SCHEMA,
                    temperature=0.7,  # creative dialogue, schema keeps it structured
                )
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompts.GENERATION_PROMPT_TEMPLATE.format(topic=topic),
                    config=config,
                )

                text = (response.text or "").strip()
                if text.startswith("```"):
                    text = text.strip("`")
                    if text.startswith("json"):
                        text = text[4:]

                dialogue = json.loads(text)
                problems = validate_dialogue(dialogue)
                if problems:
                    raise ValueError("; ".join(problems))

                n_turns = len(dialogue["turns"])
                print(f"✓ Script validated: '{dialogue['title']}' ({n_turns} turns)")
                return dialogue

            except json.JSONDecodeError as e:
                last_error = f"invalid JSON: {e}"
            except Exception as e:
                last_error = str(e)
            print(f"✗ Attempt {attempt} failed: {last_error}")

        raise RuntimeError(
            f"Director failed after {max_retries} attempts. Last error: {last_error}")

    # Legacy alias so old callers keep working
    def generate_storyboard(self, topic: str, max_retries: int = 3) -> Dict[str, Any]:
        return self.generate_dialogue(topic, max_retries)

    @staticmethod
    def save_storyboard(storyboard: Dict[str, Any], filepath: str) -> None:
        """Save the storyboard to a JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(storyboard, f, indent=4, ensure_ascii=False)
        print(f"✓ Storyboard saved to {filepath}")

    @staticmethod
    def load_storyboard(filepath: str) -> Dict[str, Any]:
        """Load a storyboard from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
