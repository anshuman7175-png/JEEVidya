"""
JEEVidya V5 — Batch Factory (Tier 4)
════════════════════════════════════
Drop 30 topics in topics.txt → overnight you get 30 self-QC'd,
thumbnailed, metadata'd shorts in output/batch/<slug>/.

Per topic:  Director script (flywheel-tuned DNA) → Tier 0 DAG render
            (a crash overnight costs one segment, not one video)
            → Critic preview review → auto-fix re-render if needed
            → thumbnail → Gemini titles/description/tags → bundle.

Resilient by design: every topic is isolated in try/except; a ledger
(batch_state.json) marks finished topics so re-running the batch skips
them — the whole factory is resumable, like everything in Tier 0.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from typing import Any, Dict, List, Optional

from config import settings

TOPICS_FILE = os.path.join(settings.PROJECT_ROOT, "topics.txt")
BATCH_DIR = os.path.join(settings.OUTPUT_DIR, "batch")
STATE_FILE = os.path.join(BATCH_DIR, "batch_state.json")

METADATA_PROMPT = """For this Hinglish educational Short, write upload
metadata. Return JSON only:
{{"youtube_title": "≤90 chars, curiosity-driven, may use the hook number",
 "description": "2-3 lines Hinglish + 3 hashtag line",
 "tags": ["12-18 search tags, mix Hindi and English"]}}

Script title: {title}
Script: {script}"""


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.strip())[:48]


def _load_state() -> Dict[str, Any]:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_state(state: Dict[str, Any]) -> None:
    os.makedirs(BATCH_DIR, exist_ok=True)
    tmp = STATE_FILE + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(tmp, STATE_FILE)


def read_topics(path: str = TOPICS_FILE) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f
                if ln.strip() and not ln.strip().startswith("#")]


class BatchFactory:
    """The overnight run."""

    def __init__(self, critic_enabled: bool = True,
                 max_fix_rounds: int = 1):
        self.critic_enabled = critic_enabled
        self.max_fix_rounds = max_fix_rounds

    # ─── One topic, end to end ─────────────────────────────

    def produce(self, topic: str) -> Dict[str, Any]:
        from agents.director import Director
        from engine.visual_dna import VisualDNA
        from factory.flywheel import Flywheel
        from factory.thumbnails import make_thumbnail
        from generate import run_dialogue_pipeline

        slug = _slug(topic)
        bundle = os.path.join(BATCH_DIR, slug)
        os.makedirs(bundle, exist_ok=True)
        result: Dict[str, Any] = {"topic": topic, "bundle": bundle}

        # 1. Script (Director agent, flywheel-tuned DNA)
        script_path = os.path.join(bundle, "script.json")
        if os.path.exists(script_path):
            with open(script_path, "r", encoding="utf-8") as f:
                dialogue = json.load(f)
            print(f"  [Batch] script cached: {slug}")
        else:
            tuning = Flywheel().recommend()
            dialogue = Director().write(topic)
            if tuning:
                dialogue["dna_tuning"] = tuning
                dialogue["dna"] = VisualDNA.from_title(
                    dialogue.get("title", topic), tuning=tuning).to_dict()
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(dialogue, f, indent=2, ensure_ascii=False)
        dna = VisualDNA.from_dialogue(dialogue)
        result["dna"] = dna.to_dict()

        # 2. Critic loop on a fast preview, then the full render
        if self.critic_enabled:
            dialogue = self._critic_loop(dialogue, script_path)
        video = run_dialogue_pipeline(dialogue)
        final_video = os.path.join(bundle, "video.mp4")
        shutil.copy2(video, final_video)
        result["video"] = final_video

        # 3. Thumbnail (best-frame + DNA title card)
        thumb = os.path.join(bundle, "thumbnail.jpg")
        make_thumbnail(final_video, dialogue.get("title", topic), thumb,
                       dna=dna)
        result["thumbnail"] = thumb

        # 4. Upload metadata
        meta = self._metadata(dialogue)
        meta["dna"] = dna.to_dict()
        with open(os.path.join(bundle, "meta.json"), "w",
                  encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        result["meta"] = meta
        return result

    def _critic_loop(self, dialogue: Dict[str, Any],
                     script_path: str) -> Dict[str, Any]:
        """Preview → vision review → apply fixes → re-preview (≤ N rounds)."""
        from agents.critic import Critic
        from generate import run_dialogue_pipeline
        try:
            critic = Critic()
            for round_no in range(self.max_fix_rounds + 1):
                preview = run_dialogue_pipeline(dialogue, preview=True)
                report = critic.review(preview)
                if not report.needs_fix or round_no == self.max_fix_rounds:
                    break
                print(f"  [Batch] Critic round {round_no + 1}: applying "
                      f"{len(report.fixes)} fixes")
                dialogue = critic.apply_fixes(report, dialogue)
                with open(script_path, "w", encoding="utf-8") as f:
                    json.dump(dialogue, f, indent=2, ensure_ascii=False)
        except Exception as e:                   # noqa: BLE001 — QC is optional
            print(f"  [Batch] Critic unavailable ({e}); rendering unreviewed")
        return dialogue

    def _metadata(self, dialogue: Dict[str, Any]) -> Dict[str, Any]:
        from agents.llm import LLM
        title = dialogue.get("title", "Untitled")
        fallback = {
            "youtube_title": title,
            "description": dialogue.get("description", title)
            + "\n#JEE #NEET #Physics",
            "tags": dialogue.get("tags", ["JEE", "NEET", "physics",
                                          "education", "shorts"]),
        }
        try:
            brief = json.dumps(
                [t.get("text", "") for t in dialogue.get("turns", [])],
                ensure_ascii=False)
            meta = LLM().generate_json(METADATA_PROMPT.format(
                title=title, script=brief), temperature=0.5)
            if isinstance(meta, dict) and meta.get("youtube_title"):
                return meta
        except Exception as e:                   # noqa: BLE001
            print(f"  [Batch] metadata agent offline ({e}); using fallback")
        return fallback

    # ─── The overnight loop ────────────────────────────────

    def run(self, topics: Optional[List[str]] = None) -> Dict[str, Any]:
        topics = topics if topics is not None else read_topics()
        if not topics:
            print(f"  [Batch] No topics. Add lines to {TOPICS_FILE}")
            return {"done": 0, "failed": 0}

        state = _load_state()
        done = failed = skipped = 0
        t0 = time.time()
        for i, topic in enumerate(topics, 1):
            slug = _slug(topic)
            if state.get(slug, {}).get("status") == "done":
                skipped += 1
                continue
            print(f"\n═══ Batch {i}/{len(topics)}: {topic} ═══")
            try:
                result = self.produce(topic)
                state[slug] = {"status": "done", "topic": topic,
                               "finished_at": time.time(),
                               "video": result["video"]}
                done += 1
            except Exception as e:               # noqa: BLE001 — isolate topics
                traceback.print_exc()
                state[slug] = {"status": "failed", "topic": topic,
                               "error": str(e)}
                failed += 1
            _save_state(state)

        hrs = (time.time() - t0) / 3600
        print(f"\n═══ Batch complete: {done} produced · {skipped} skipped "
              f"· {failed} failed · {hrs:.1f}h ═══")
        return {"done": done, "failed": failed, "skipped": skipped}
