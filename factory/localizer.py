"""
JEEVidya V5 — Localizer (Tier 4)
════════════════════════════════
One script → N language variants → 3-4× content from one Director run.
Voices swap per language (edge-tts neural voices), spoken lines are
translated by the LLM with strict register instructions, and everything
downstream (Tier 0 DAG, DNA, thumbnails) works unchanged because the
output is just another dialogue JSON.
"""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional

from agents.llm import LLM

# Per-language voice pairs (girl, boy) — all free edge-tts neural voices
LANGUAGE_VOICES: Dict[str, Dict[str, str]] = {
    "hi": {"girl": "hi-IN-SwaraNeural", "boy": "hi-IN-MadhurNeural"},
    "en": {"girl": "en-IN-NeerjaNeural", "boy": "en-IN-PrabhatNeural"},
    "ta": {"girl": "ta-IN-PallaviNeural", "boy": "ta-IN-ValluvarNeural"},
    "te": {"girl": "te-IN-ShrutiNeural", "boy": "te-IN-MohanNeural"},
}

TRANSLATE_PROMPT = """Translate these educational dialogue lines from
Hinglish into {language_name} exactly as a friendly {language_name}-speaking
teacher would SPEAK them to a teenager. Rules:
- Keep physics/math terms in English (velocity, formula names, numbers).
- Keep the energy: exclamations stay exclamations, questions stay questions.
- Same number of lines, same order.

Lines (JSON array):
{lines}

Return JSON only: {{"lines": ["...", ...]}}"""

LANGUAGE_NAMES = {"en": "Indian English", "ta": "Tamil", "te": "Telugu",
                  "hi": "Hinglish"}


class Localizer:

    def __init__(self, llm: Optional[LLM] = None):
        self.llm = llm or LLM()

    def localize(self, dialogue: Dict[str, Any],
                 language: str) -> Dict[str, Any]:
        """One variant. 'hi' returns a tagged copy (source language)."""
        if language not in LANGUAGE_VOICES:
            raise ValueError(f"unsupported language '{language}' "
                             f"(have: {sorted(LANGUAGE_VOICES)})")
        variant = copy.deepcopy(dialogue)
        variant["language"] = language
        variant["voices"] = LANGUAGE_VOICES[language]

        if language != "hi":
            speaking = [t for t in variant.get("turns", [])
                        if t.get("speaker") != "explanation"]
            lines = [t.get("text", "") for t in speaking]
            result = self.llm.generate_json(
                TRANSLATE_PROMPT.format(
                    language_name=LANGUAGE_NAMES[language],
                    lines=json.dumps(lines, ensure_ascii=False)),
                temperature=0.4)
            translated = result.get("lines", [])
            if len(translated) != len(lines):
                raise RuntimeError(
                    f"translation returned {len(translated)} lines "
                    f"for {len(lines)} inputs")
            for turn, text in zip(speaking, translated, strict=True):
                turn["text"] = text
            variant["title"] = f"{dialogue.get('title', '')} [{language}]"
        return variant

    def localize_all(self, dialogue: Dict[str, Any],
                     languages: List[str] = ("hi", "en")
                     ) -> Dict[str, Dict[str, Any]]:
        out = {}
        for lang in languages:
            try:
                out[lang] = self.localize(dialogue, lang)
                print(f"  [Localize] {lang} ✓")
            except Exception as e:              # noqa: BLE001 — isolate langs
                print(f"  [Localize] {lang} failed: {e}")
        return out
