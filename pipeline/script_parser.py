"""
JEEVidya — Intelligent Script Parser & Full Audio Slicer
════════════════════════════════════════════════════════
Parses raw script text blocks into structured turns, identifies speaker continuity
(how many lines spoken consecutively by one character before the next), detects
emotion annotations, and slices full-length dialogue audio files into per-turn MP3s.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pydub
from pydub.silence import detect_nonsilent

# ─── Speaker Regex Patterns ───
# Maps various user aliases to standardized pipeline characters ("girl", "boy", "explanation")
SPEAKER_ALIASES = {
    # Gudiya / Girl aliases
    "gudiya": "girl",
    "gudya": "girl",
    "guddi": "girl",
    "girl": "girl",
    "didi": "girl",
    "sister": "girl",
    "g": "girl",
    "गुड़िया": "girl",
    "गुडिया": "girl",
    "दीदी": "girl",

    # Chintu / Boy aliases
    "chintu": "boy",
    "chintoo": "boy",
    "boy": "boy",
    "bhai": "boy",
    "brother": "boy",
    "c": "boy",
    "चिंटू": "boy",
    "चिन्टू": "boy",
    "भाई": "boy",

    # Blackboard / Presentation aliases
    "board": "explanation",
    "blackboard": "explanation",
    "slide": "explanation",
    "explain": "explanation",
    "explanation": "explanation",
    "visual": "explanation",
    "स्लाइड": "explanation",
    "बोर्ड": "explanation",

    # Speaker 1 / Speaker 2 / Generic aliases
    "speaker1": "girl",
    "speaker2": "boy",
    "speakera": "girl",
    "speakerb": "boy",
    "person1": "girl",
    "person2": "boy",
    "character1": "girl",
    "character2": "boy",
    "char1": "girl",
    "char2": "boy",
    "s1": "girl",
    "s2": "boy",
}

EMOTION_KEYWORDS = {
    "curious", "explaining", "amazed", "enthusiastic",
    "confident", "neutral", "dramatic", "scared", "celebrating",
    "happy", "surprised", "thinking", "serious"
}

# Mapping common synonyms to standard emotion set
EMOTION_SYNONYMS = {
    "happy": "enthusiastic",
    "surprised": "amazed",
    "thinking": "curious",
    "serious": "explaining",
    "question": "curious",
    "shocked": "dramatic",
}


@dataclass
class SpeakerBlock:
    """A consecutive sequence of lines spoken by a single speaker."""
    speaker: str
    lines: List[str] = field(default_factory=list)
    emotions: List[str] = field(default_factory=list)
    shot_types: List[str] = field(default_factory=list)


def normalize_speaker(raw_label: str) -> Optional[str]:
    """Normalize a candidate speaker label string to 'girl', 'boy', or 'explanation'."""
    clean = raw_label.strip().lower()
    clean = re.sub(r"[^a-z0-9\u0900-\u097F]", "", clean)
    return SPEAKER_ALIASES.get(clean, None)


def is_metadata_line(line: str) -> bool:
    """Check if a line contains metadata tags, title, scene notes, or hashtags rather than dialogue."""
    t = line.strip()
    if not t:
        return True
    if re.match(r"^(?:tags|hashtags|keywords|title|topic|description|subject|scene|setting|sfx|sound effect|bgm|music)\s*[:：\-–]", t, re.I):
        return True
    if re.match(r"^(?:#[\w\-]+\s*)+$", t):
        return True
    if re.match(r"^\[(?:scene|setting|sfx|sound effect|bgm|music)[:\s].*\]$", t, re.I):
        return True
    if re.match(r"^\((?:scene|setting|sfx|sound effect|bgm|music)[:\s].*\)$", t, re.I):
        return True
    return False


def extract_emotion_and_shot(text: str, default_speaker: str) -> Tuple[str, str, str]:
    """
    Extract bracketed emotion/shot annotations from dialogue line.
    Examples:
      '[curious] Kya tare timtimate hain?' -> emotion='curious', clean_text='Kya tare...'
      '(amazed, medium) Arre waah!' -> emotion='amazed', shot='medium', clean_text='Arre waah!'
    """
    emotion = "curious" if default_speaker == "girl" else "enthusiastic"
    shot_type = "two_shot"

    t = re.sub(r"^(?:turn\s*\d+[\.\:\-–\s]*|\d+[\.\)\-–\s]+)", "", text.strip(), flags=re.I)
    prefix_pattern = r"^(?:\*{1,2}|\[)?(?:speaker\s*\d+|gudiya|chintu|girl|boy|didi|bhai|character\s*\d+|person\s*\d+|s[12])(?:\*{1,2}|\])?\s*[:：\-–]\s*"

    # Match brackets [ ... ] or parentheses ( ... )
    match = re.search(r"^[\[\(]([^\]\)]+)[\]\)]\s*(.*)$", t)
    if match:
        meta_str = match.group(1).lower().strip()
        remaining_text = match.group(2).strip()

        # Check tokens inside bracket
        tokens = [tok.strip() for tok in re.split(r"[,;]+", meta_str) if tok.strip()]
        for tok in tokens:
            # Check shot types
            if tok in ("two_shot", "medium", "extreme_closeup", "reaction_cut", "reveal", "fullscreen_explain"):
                shot_type = tok
            elif tok in EMOTION_KEYWORDS:
                emotion = tok
            elif tok in EMOTION_SYNONYMS:
                emotion = EMOTION_SYNONYMS[tok]

        clean_text = re.sub(prefix_pattern, "", remaining_text, flags=re.I).strip('"\'*` ')
        return emotion, shot_type, clean_text

    clean_text = re.sub(prefix_pattern, "", t, flags=re.I).strip('"\'*` ')
    return emotion, shot_type, clean_text


def parse_dialogue_script(
    raw_script: str,
    group_consecutive: bool = False,
    default_first_speaker: str = "girl",
) -> List[Dict[str, Any]]:
    """
    Parse a whole dialogue script text into structured turns.

    Identifies:
    1. Speaker prefixes (e.g. 'Gudiya:', 'Chintu:', 'G:', 'C:').
    2. Speaker continuity: how many lines are spoken consecutively by each character.
    3. Emotion cues and shot directives.
    4. Blackboard slides ('[Slide: ...]' or 'Board: ...').

    If group_consecutive is True, consecutive lines by the same speaker are merged into
    one turn card. If False, each sentence/line is kept as its own turn card, while
    annotating speaker transitions and block counts.
    """
    lines = [line.strip() for line in raw_script.strip().splitlines() if line.strip()]
    if not lines:
        return []

    # Regex to catch speaker labels: e.g. "1. Gudiya: ...", "**Speaker 1:** ...", "Gudiya (curious): ...", "[Chintu]: ..."
    speaker_prefix_regex = re.compile(
        r"^(?:(?:\d+[\.\)\-]\s*)*)?(?:\*{1,2}|\[)?([A-Za-z0-9\s\u0900-\u097F]{1,25}?)(?:\*{1,2}|\])?\s*(?:\([^\)]+\)|\[[^\]]+\])?\s*[:：\-–]\s*(.*)$"
    )

    # Filter out pure metadata lines (Tags, Title, Description, Hashtags, Scene notes)
    clean_lines = [line for line in lines if not is_metadata_line(line)]
    if not clean_lines:
        return []

    # First pass: parse lines and assign speaker
    current_speaker = default_first_speaker
    parsed_entries: List[Tuple[str, str]] = []  # (speaker, raw_text)

    # Check if script has explicit labels
    has_labels = any(
        bool(speaker_prefix_regex.match(line) and normalize_speaker(speaker_prefix_regex.match(line).group(1)))
        for line in clean_lines
    )

    if has_labels:
        for line in clean_lines:
            # Check for slide directive like "[Slide: ...]" or "[Board: ...]" (supports formulas with parens)
            slide_match = (
                re.match(r"^\[(?:slide|board|explanation)[:\s]+(.*)\]$", line, re.IGNORECASE) or
                re.match(r"^\((?:slide|board|explanation)[:\s]+(.*)\)$", line, re.IGNORECASE)
            )
            if slide_match:
                parsed_entries.append(("explanation", slide_match.group(1).strip()))
                continue

            m = speaker_prefix_regex.match(line)
            if m:
                cand_speaker = normalize_speaker(m.group(1))
                if cand_speaker:
                    current_speaker = cand_speaker
                    text = m.group(2).strip().strip('"\'*` ')
                    if text:
                        parsed_entries.append((current_speaker, text))
                    continue

            # If no new prefix, this line continues the previous speaker's dialogue
            cleaned_line = line.strip('"\'*` ')
            if cleaned_line:
                parsed_entries.append((current_speaker, cleaned_line))
    else:
        # Alternating dialogue if no labels exist
        alt_speaker = default_first_speaker
        other_speaker = "boy" if default_first_speaker == "girl" else "girl"

        for line in clean_lines:
            slide_match = (
                re.match(r"^\[(?:slide|board|explanation)[:\s]+(.*)\]$", line, re.IGNORECASE) or
                re.match(r"^\((?:slide|board|explanation)[:\s]+(.*)\)$", line, re.IGNORECASE)
            )
            if slide_match:
                parsed_entries.append(("explanation", slide_match.group(1).strip()))
                continue

            cleaned_line = line.strip('"\'*` ')
            if cleaned_line:
                parsed_entries.append((alt_speaker, cleaned_line))
                alt_speaker, other_speaker = other_speaker, alt_speaker

    if not parsed_entries:
        return []

    # Second pass: group into consecutive speaker blocks
    blocks: List[SpeakerBlock] = []
    current_block = SpeakerBlock(speaker=parsed_entries[0][0])

    for spk, text in parsed_entries:
        if spk != current_block.speaker:
            blocks.append(current_block)
            current_block = SpeakerBlock(speaker=spk)

        em, shot, clean_text = extract_emotion_and_shot(text, spk)
        if spk == "explanation":
            shot = "fullscreen_explain"
            em = "neutral"

        current_block.lines.append(clean_text)
        current_block.emotions.append(em)
        current_block.shot_types.append(shot)

    if current_block.lines:
        blocks.append(current_block)

    # Third pass: build output turn list
    turns: List[Dict[str, Any]] = []
    turn_counter = 1

    if group_consecutive:
        # One turn per speaker block
        for block_idx, block in enumerate(blocks):
            combined_text = " ".join(block.lines)
            primary_emotion = block.emotions[0] if block.emotions else "neutral"
            primary_shot = block.shot_types[0] if block.shot_types else "two_shot"

            turn = {
                "turn_id": turn_counter,
                "speaker": block.speaker,
                "text": combined_text,
                "emotion": primary_emotion,
                "shot_type": primary_shot,
                "consecutive_lines_count": len(block.lines),
                "block_index": block_idx + 1,
            }
            if block.speaker == "explanation":
                turn["duration_seconds"] = 4.0
                turn["pause_after"] = 1.0
            turns.append(turn)
            turn_counter += 1
    else:
        # Separate turn for every line, with speaker continuity metadata
        for block_idx, block in enumerate(blocks):
            total_in_block = len(block.lines)
            for line_idx, (text, em, shot) in enumerate(
                zip(block.lines, block.emotions, block.shot_types)
            ):
                turn = {
                    "turn_id": turn_counter,
                    "speaker": block.speaker,
                    "text": text,
                    "emotion": em,
                    "shot_type": shot,
                    "speaker_block_index": block_idx + 1,
                    "line_in_block": line_idx + 1,
                    "total_in_block": total_in_block,
                    "is_block_start": line_idx == 0,
                    "is_block_end": line_idx == total_in_block - 1,
                }
                if block.speaker == "explanation":
                    turn["duration_seconds"] = 4.0
                    turn["pause_after"] = 1.0
                turns.append(turn)
                turn_counter += 1

    return turns


def split_audio_by_silence(
    input_audio_path: str,
    output_dir: str,
    expected_turns: Optional[int] = None,
    min_silence_len: int = 300,
    silence_thresh_db: float = -38.0,
    padding_ms: int = 60,
) -> List[Dict[str, Any]]:
    """
    Split a single unified full dialogue audio file into turn-specific MP3 files
    based on speech activity detection (pauses / silences between dialogue turns).

    Returns a list of dictionaries with turn audio metadata:
      [{ "turn_id": 1, "path": ".../turn_001.mp3", "filename": "turn_001.mp3",
         "duration_ms": 2500, "duration_s": 2.50 }]
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load audio via pydub
    sound = pydub.AudioSegment.from_file(input_audio_path)

    # Standardize to 44.1kHz, 2 channels for pipeline uniformity
    sound = sound.set_frame_rate(44100).set_channels(2)

    # Detect nonsilent chunks (start_ms, end_ms)
    chunks = detect_nonsilent(
        sound,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh_db,
        seek_step=10,
    )

    # If expected_turns is given and detected chunk count doesn't match,
    # try slightly adjusting the threshold to get closer to expected_turns
    if expected_turns and expected_turns > 1 and len(chunks) != expected_turns:
        # Try a slightly stricter or looser threshold
        for candidate_thresh in (silence_thresh_db - 4, silence_thresh_db + 4, silence_thresh_db - 8):
            cand_chunks = detect_nonsilent(
                sound,
                min_silence_len=min_silence_len,
                silence_thresh=candidate_thresh,
                seek_step=10,
            )
            if len(cand_chunks) == expected_turns:
                chunks = cand_chunks
                break

    # If still no speech detected, treat the entire file as one chunk
    if not chunks:
        chunks = [[0, len(sound)]]

    results: List[Dict[str, Any]] = []

    for i, (start_ms, end_ms) in enumerate(chunks):
        turn_id = i + 1
        # Add smooth padding around cuts, clamped to audio boundaries
        chunk_start = max(0, start_ms - padding_ms)
        chunk_end = min(len(sound), end_ms + padding_ms)

        chunk_audio = sound[chunk_start:chunk_end]

        out_filename = f"turn_{turn_id:03d}.mp3"
        out_path = os.path.join(output_dir, out_filename)

        # Export as 192k MP3
        chunk_audio.export(out_path, format="mp3", bitrate="192k")

        dur_ms = len(chunk_audio)
        dur_s = round(dur_ms / 1000.0, 2)

        results.append({
            "turn_id": turn_id,
            "path": out_path,
            "filename": out_filename,
            "duration_ms": dur_ms,
            "duration_s": dur_s,
            "start_ms": chunk_start,
            "end_ms": chunk_end,
        })

    return results
