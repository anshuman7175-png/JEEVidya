#!/usr/bin/env python3
"""
Parse the 20 Physics Season 2 Shorts from Gudiya_Chintu_Season2_Physics_Master_Plan.md
and save them as individual JSON scene files in scenes/season2/ and scenes/season2_index.json.
"""

import os
import re
import json

MASTER_PLAN_PATH = "/Users/anshuman/Downloads/Gudiya_Chintu_Season2_Physics_Master_Plan.md"
WORKSPACE_ROOT = "/Users/anshuman/an39/JEEVidya"
OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "scenes", "season2")
INDEX_PATH = os.path.join(WORKSPACE_ROOT, "scenes", "season2_index.json")

def parse_scripts():
    if not os.path.exists(MASTER_PLAN_PATH):
        raise FileNotFoundError(f"Master plan not found at {MASTER_PLAN_PATH}")

    with open(MASTER_PLAN_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Find section 3: ## 3. Season 2 — "Newton ki Duniya" — The 20 scripts
    s3_match = re.search(r"##\s*3\.\s*Season 2\s*—\s*[\"']Newton ki Duniya[\"'](.*?)(?=##\s*4\.|$)", content, re.DOTALL)
    if not s3_match:
        raise ValueError("Could not find Section 3 in Master Plan")

    section3_text = s3_match.group(1)

    # Split by `### Script `
    script_chunks = re.split(r"(?m)^###\s*Script\s+(\d+)\s*—\s*(.+)$", section3_text)
    # script_chunks: [preamble, script_num_1, title_1, body_1, script_num_2, title_2, body_2, ...]

    episodes = []
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    idx = 1
    while idx < len(script_chunks):
        script_num = int(script_chunks[idx].strip())
        raw_title = script_chunks[idx + 1].strip()
        body = script_chunks[idx + 2].strip()
        idx += 3

        # Extract metadata line: e.g. **Channel video #7 · NCERT XI Ch 7 ... · JEE/NEET: ...**
        meta_line_match = re.search(r"\*\*Channel video #(\d+)\s*·\s*([^\n*]+)\*\*", body)
        channel_video_num = None
        ncert_chapter = ""
        exam_focus = ""

        if meta_line_match:
            channel_video_num = int(meta_line_match.group(1))
            rest_of_meta = meta_line_match.group(2).strip()
            # Split rest_of_meta by ` · ` (middle dot with surrounding spaces)
            parts = re.split(r"\s+·\s+", rest_of_meta)
            ncert_chapter = parts[0].strip() if len(parts) > 0 else ""
            exam_focus = parts[1].strip() if len(parts) > 1 else ""

        # Hook text: **Hook text (first frame):** "..." or **Hook text:** "..."
        hook_match = re.search(r"\*\*Hook text(?:\s*\(first frame\))?:\*\*\s*[\"“]([^\"”\n]+)[\"”]", body)
        hook_text = hook_match.group(1).strip() if hook_match else ""

        # Concept in one line: **Concept in one line:** ...
        concept_match = re.search(r"\*\*Concept in one line:\*\*\s*([^\n]+)", body)
        concept = concept_match.group(1).strip() if concept_match else ""

        # On-screen cues: **On-screen cues:** ...
        cues_match = re.search(r"\*\*On-screen cues:\*\*\s*([^\n]+)", body)
        on_screen_cues = cues_match.group(1).strip() if cues_match else ""

        # Pinned comment: **Pinned comment:** "..."
        pinned_match = re.search(r"\*\*Pinned comment:\*\*\s*[\"“]?([^\n]+?)[\"”]?$", body, re.MULTILINE)
        pinned_comment = pinned_match.group(1).strip().strip('"') if pinned_match else ""

        # Titles: **Titles:** "..." / "..."
        titles_match = re.search(r"\*\*Titles:\*\*\s*([^\n]+)", body)
        titles_str = titles_match.group(1).strip() if titles_match else ""

        # Hashtags: **Hashtags:** ...
        hashtags_match = re.search(r"\*\*Hashtags:\*\*\s*([^\n]+)", body)
        hashtags = hashtags_match.group(1).strip() if hashtags_match else ""
        tags_list = [t.strip("#").strip() for t in hashtags.split() if t.strip()]

        # Extract Dialogue Turns
        # Lines matching: **Gudiya:** [emotion] Text... or **Chintu:** [emotion] Text...
        turns = []
        elevenlabs_dialogue_lines = []
        clean_dialogue_lines = []

        turn_counter = 1
        for line in body.splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            turn_match = re.match(r"^\*\*(Gudiya|Chintu):\*\*\s*(?:\[([^\]]+)\]\s*)?(.*)$", line_str, re.IGNORECASE)
            if turn_match:
                speaker_raw = turn_match.group(1).lower()
                speaker = "girl" if "gud" in speaker_raw else "boy"
                speaker_name = "Gudiya" if speaker == "girl" else "Chintu"
                emotion = (turn_match.group(2) or "").strip().lower()
                speech_text = turn_match.group(3).strip()

                elevenlabs_line = f"[{emotion}] {speech_text}" if emotion else speech_text
                elevenlabs_dialogue_lines.append(f"{speaker_name}: {elevenlabs_line}")
                clean_dialogue_lines.append(f"{speaker_name}: {speech_text}")

                turns.append({
                    "turn_id": turn_counter,
                    "speaker": speaker,
                    "speaker_name": speaker_name,
                    "emotion": emotion or ("curious" if speaker == "girl" else "enthusiastic"),
                    "shot_type": "two_shot",
                    "text": speech_text,
                    "elevenlabs_line": elevenlabs_line,
                    "custom_audio": None,
                    "audio_url": None
                })
                turn_counter += 1

        # Slug for filename
        clean_slug = re.sub(r"[^a-zA-Z0-9]+", "_", raw_title.lower()).strip("_")
        scene_filename = f"s02e{script_num:02d}_{clean_slug[:30]}.json"
        scene_id = f"s02e{script_num:02d}"

        episode_data = {
            "id": scene_id,
            "season": 2,
            "episode": script_num,
            "channel_video_num": channel_video_num,
            "title": raw_title,
            "description": f"Season 2 Episode {script_num}: {concept}",
            "ncert_chapter": ncert_chapter,
            "exam_focus": exam_focus,
            "hook_text": hook_text,
            "concept": concept,
            "on_screen_cues": on_screen_cues,
            "pinned_comment": pinned_comment,
            "suggested_titles": titles_str,
            "tags": tags_list,
            "elevenlabs_script": "\n".join(elevenlabs_dialogue_lines),
            "clean_script": "\n".join(clean_dialogue_lines),
            "turns": turns
        }

        # Save to individual scene json
        scene_filepath = os.path.join(OUTPUT_DIR, scene_filename)
        with open(scene_filepath, "w", encoding="utf-8") as out_f:
            json.dump(episode_data, out_f, indent=2, ensure_ascii=False)

        episodes.append({
            "id": scene_id,
            "season": 2,
            "episode": script_num,
            "channel_video_num": channel_video_num,
            "title": raw_title,
            "ncert_chapter": ncert_chapter,
            "exam_focus": exam_focus,
            "hook_text": hook_text,
            "concept": concept,
            "turn_count": len(turns),
            "filename": scene_filename,
            "relative_path": f"scenes/season2/{scene_filename}"
        })

    # Write master index
    with open(INDEX_PATH, "w", encoding="utf-8") as idx_f:
        json.dump({
            "season": 2,
            "title": "Newton ki Duniya",
            "curriculum": "Class 11 Physics (Mechanics) — NCERT Ch 2–7",
            "total_episodes": len(episodes),
            "episodes": episodes
        }, idx_f, indent=2, ensure_ascii=False)

    print(f"Successfully parsed and saved {len(episodes)} Season 2 scripts to {OUTPUT_DIR}")
    print(f"Index written to {INDEX_PATH}")

if __name__ == "__main__":
    parse_scripts()
