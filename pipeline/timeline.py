"""
JEEVidya V5 — Global Millisecond Timeline
═════════════════════════════════════════
The single source of truth for time.

V2 truncated frame counts per-turn (int(ms/1000*fps)) while audio was
concatenated at exact length — accumulating up to 1 frame of A/V drift per
turn. V5 derives EVERY frame boundary from one cumulative millisecond axis,
so drift is structurally impossible.

It also promotes the edge-tts VTT word boundaries (previously dead data)
into first-class WordEvents and karaoke CaptionChunks, giving captions,
lipsync, shot cuts, and VFX a shared, speech-accurate clock.
"""
from __future__ import annotations

import bisect
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_VTT_TIMING = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})"
)

# Caption chunking targets (Shorts-style: short, punchy lines)
MAX_WORDS_PER_CHUNK = 4
MAX_CHARS_PER_CHUNK = 20
CHUNK_LINGER_MS = 600  # How long the last chunk stays after speech ends


@dataclass
class WordEvent:
    """A single spoken word with global millisecond timing."""
    text: str
    start_ms: int
    end_ms: int


@dataclass
class CaptionChunk:
    """A karaoke caption line: a few words + a display window."""
    words: List[WordEvent]
    start_ms: int   # display window start (global)
    end_ms: int     # display window end (global)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


@dataclass
class TurnSpan:
    """A dialogue turn placed on the global timeline."""
    turn: Dict[str, Any]
    start_ms: int
    end_ms: int
    start_frame: int
    end_frame: int
    words: List[WordEvent] = field(default_factory=list)
    chunks: List[CaptionChunk] = field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def parse_vtt_words(vtt_path: str) -> List[WordEvent]:
    """
    Parse an edge-tts VTT file into word-level events (local ms).
    edge-tts emits one cue per word/short group — exactly what karaoke needs.
    """
    if not vtt_path or not os.path.exists(vtt_path):
        return []

    events: List[WordEvent] = []
    try:
        with open(vtt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return []

    pending: Optional[Tuple[int, int]] = None
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("WEBVTT"):
            continue
        m = _VTT_TIMING.match(line)
        if m:
            h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(g) for g in m.groups())
            start = ((h1 * 3600 + m1 * 60 + s1) * 1000) + ms1
            end = ((h2 * 3600 + m2 * 60 + s2) * 1000) + ms2
            pending = (start, end)
        elif pending:
            start, end = pending
            # A cue may contain multiple words; distribute timing evenly.
            words = line.split()
            if words:
                step = max(1, (end - start) // len(words))
                for i, w in enumerate(words):
                    w_start = start + i * step
                    w_end = end if i == len(words) - 1 else w_start + step
                    events.append(WordEvent(text=w, start_ms=w_start, end_ms=w_end))
            pending = None

    events.sort(key=lambda w: w.start_ms)
    return events


def _build_chunks(words: List[WordEvent], turn_end_ms: int) -> List[CaptionChunk]:
    """Group word events into karaoke chunks with contiguous display windows."""
    if not words:
        return []

    groups: List[List[WordEvent]] = []
    current: List[WordEvent] = []
    for w in words:
        char_len = sum(len(x.text) + 1 for x in current) + len(w.text)
        if current and (len(current) >= MAX_WORDS_PER_CHUNK or char_len > MAX_CHARS_PER_CHUNK):
            groups.append(current)
            current = []
        current.append(w)
    if current:
        groups.append(current)

    chunks: List[CaptionChunk] = []
    for i, group in enumerate(groups):
        start = group[0].start_ms
        if i + 1 < len(groups):
            end = groups[i + 1][0].start_ms
        else:
            end = min(turn_end_ms, group[-1].end_ms + CHUNK_LINGER_MS)
        chunks.append(CaptionChunk(words=group, start_ms=start, end_ms=max(end, start + 1)))
    return chunks


class Timeline:
    """
    Builds the global timeline from VoiceEngine turn data.

    Every turn's duration_ms (actual audio + padding) is laid on one
    cumulative axis; frame indices are round(ms * fps / 1000) of that axis.
    """

    def __init__(self, turn_data: List[Dict[str, Any]], fps: int):
        self.fps = fps
        self.spans: List[TurnSpan] = []

        cursor_ms = 0
        for turn in turn_data:
            duration = int(turn.get("duration_ms", 0))
            start_ms = cursor_ms
            end_ms = cursor_ms + duration
            span = TurnSpan(
                turn=turn,
                start_ms=start_ms,
                end_ms=end_ms,
                start_frame=round(start_ms * fps / 1000),
                end_frame=round(end_ms * fps / 1000),
            )

            # Promote VTT word boundaries to global time
            local_words = parse_vtt_words(turn.get("vtt"))
            span.words = [
                WordEvent(w.text, w.start_ms + start_ms, w.end_ms + start_ms)
                for w in local_words
            ]
            span.chunks = _build_chunks(span.words, end_ms)

            self.spans.append(span)
            cursor_ms = end_ms

        self.total_ms = cursor_ms
        self.total_frames = round(self.total_ms * fps / 1000)
        self._starts = [s.start_ms for s in self.spans]

    # ─── Lookups ───────────────────────────────────────────

    def frame_ms(self, frame: int) -> float:
        """Exact millisecond position of a frame on the global axis."""
        return frame * 1000.0 / self.fps

    def span_at_frame(self, frame: int) -> Optional[TurnSpan]:
        """The turn active at a given frame."""
        if not self.spans:
            return None
        t = self.frame_ms(frame)
        idx = bisect.bisect_right(self._starts, t) - 1
        idx = max(0, min(idx, len(self.spans) - 1))
        return self.spans[idx]

    def active_caption(self, span: TurnSpan,
                       t_ms: float) -> Tuple[Optional[CaptionChunk], int]:
        """
        The caption chunk visible at time t within a span, plus the index of
        the currently spoken word inside it (-1 = none highlighted).
        """
        for chunk in span.chunks:
            if chunk.start_ms <= t_ms < chunk.end_ms:
                active = -1
                for i, w in enumerate(chunk.words):
                    if w.start_ms <= t_ms:
                        active = i
                    else:
                        break
                # Un-highlight once the word has clearly ended (gap between words)
                if 0 <= active < len(chunk.words) and t_ms >= chunk.words[active].end_ms + 120:
                    active = -1
                return chunk, active
        return None, -1

    def word_at(self, t_ms: float) -> Optional[WordEvent]:
        """The word being spoken anywhere on the timeline at t (for VFX triggers)."""
        idx = bisect.bisect_right(self._starts, t_ms) - 1
        if 0 <= idx < len(self.spans):
            for w in self.spans[idx].words:
                if w.start_ms <= t_ms < w.end_ms:
                    return w
        return None

    # ─── Diagnostics ───────────────────────────────────────

    def describe(self) -> str:
        n_words = sum(len(s.words) for s in self.spans)
        n_chunks = sum(len(s.chunks) for s in self.spans)
        return (f"Timeline: {len(self.spans)} turns, {self.total_ms / 1000:.2f}s, "
                f"{self.total_frames} frames @ {self.fps}fps, "
                f"{n_words} word events, {n_chunks} caption chunks, drift=0ms")
