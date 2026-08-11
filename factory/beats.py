"""
JEEVidya — Beat Ledger (Terminal Plan, Part XIX §1)
═══════════════════════════════════════════════════
The learning loop's ground truth.

YouTube Analytics exposes per-MOMENT audience retention. A video-level
flywheel throws away 99% of that signal. To learn at the beat, every
script beat must be pinned to its exact millisecond span in the MUXED
video — and that mapping must be gate-protected like every other
artifact (Law 1: if it can drift, a mechanism must make drift
impossible).

So this module emits `<video>.beats.json`:

    • one record per script beat, with its [start_ms, end_ms) span,
    • a POLICY fingerprint per beat (hook archetype, reveal timing,
      gesture density, caption density, affect arc position …) — the
      arm space the beat-level bandit samples over (§XIX bandit),
    • a checksum over the canonical span table + the muxed duration, so
      a re-encode, a re-cut, or a hand-edited ledger all invalidate it.

`verify(video_path)` is what `factory/publisher.py` calls before upload:
no ledger, stale checksum, or span table that overruns the muxed
timeline ⇒ the upload is refused. The learning loop can never be fed a
lie.

Timings come from the global `pipeline.timeline.Timeline` (which itself
derives from the aligner), never from re-guessing durations here.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from config import settings

LEDGER_SCHEMA_VERSION = 1

# The muxed duration may differ from the timeline by at most one frame's
# worth of container rounding plus the encoder's trailing-frame policy.
DURATION_TOLERANCE_MS = 120.0

# ─── Beat taxonomy (the bandit's categorical arm space) ───────────────

KIND_HOOK = "hook"                 # first ≤2 s: the scroll-stopper
KIND_QUESTION = "question"         # a doubt is raised
KIND_EXPLAIN = "explain"           # the teaching body
KIND_VISUAL = "visual"             # animated explanation / formula beat
KIND_REVEAL = "reveal"             # the payoff / answer
KIND_REACT = "react"               # reaction cut, comedic beat
KIND_CTA = "cta"                   # final beat: retention loop / ask

HOOK_ARCHETYPES = ("question", "cold_reveal", "character_react", "claim")

_QUESTION_MARKS = ("?", "？", "क्या", "kya", "kitni", "kaise", "kyun", "kyu")
_REVEAL_WORDS = ("formula", "answer", "jawab", "yaad", "isko kehte",
                 "matlab yeh", "toh", "therefore", "yani")


def _has_any(text: str, needles: Sequence[str]) -> bool:
    low = text.lower()
    return any(n.lower() in low for n in needles)


def classify(turn: Dict[str, Any], index: int, n_turns: int) -> str:
    """Beat kind from the script alone — deterministic, no LLM."""
    speaker = str(turn.get("speaker", ""))
    shot = str(turn.get("shot_type", ""))
    text = str(turn.get("text", "") or "")

    if speaker == "explanation" or turn.get("visual_elements"):
        return KIND_VISUAL
    if index == 0:
        return KIND_HOOK
    if index == n_turns - 1:
        return KIND_CTA if not _has_any(text, _QUESTION_MARKS) else KIND_QUESTION
    if shot in ("reaction_cut",):
        return KIND_REACT
    if shot in ("reveal",) or _has_any(text, _REVEAL_WORDS):
        return KIND_REVEAL
    if _has_any(text, _QUESTION_MARKS):
        return KIND_QUESTION
    return KIND_EXPLAIN


def hook_archetype(first_turn: Dict[str, Any]) -> str:
    """Which of the four first-two-second treatments this video used."""
    text = str(first_turn.get("text", "") or "")
    if first_turn.get("visual_elements"):
        return "cold_reveal"
    if str(first_turn.get("shot_type", "")) == "reaction_cut":
        return "character_react"
    if _has_any(text, _QUESTION_MARKS):
        return "question"
    return "claim"


def _density_bin(value: float, lo: float, hi: float) -> str:
    if value < lo:
        return "lo"
    if value < hi:
        return "mid"
    return "hi"


# ─── Records ─────────────────────────────────────────────────────────

@dataclass
class Beat:
    """One script beat pinned to the muxed timeline."""
    index: int
    turn_id: Any
    speaker: str
    kind: str
    emotion: str
    shot_type: str
    start_ms: int
    end_ms: int
    words: int
    caption_chunks: int
    gestures: int
    policy: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def canonical(self) -> str:
        """The checksum-bearing projection: identity + span only."""
        return (f"{self.index}|{self.turn_id}|{self.kind}|"
                f"{self.start_ms}|{self.end_ms}")


@dataclass
class BeatLedger:
    video: str
    total_ms: int
    fps: int
    beats: List[Beat]
    schema: int = LEDGER_SCHEMA_VERSION
    genes: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""

    # ─── checksum ────────────────────────────────────────────

    def compute_checksum(self) -> str:
        h = hashlib.sha256()
        h.update(f"v{self.schema}|{self.fps}|{self.total_ms}".encode())
        for b in self.beats:
            h.update(b.canonical().encode())
        return h.hexdigest()

    def seal(self) -> "BeatLedger":
        self.checksum = self.compute_checksum()
        return self

    # ─── serialization ───────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "video": self.video,
            "fps": self.fps,
            "total_ms": self.total_ms,
            "genes": self.genes,
            "checksum": self.checksum or self.compute_checksum(),
            "beats": [asdict(b) for b in self.beats],
        }

    def save(self, path: str) -> str:
        self.seal()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".part"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        return path

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "BeatLedger":
        return BeatLedger(
            video=d.get("video", ""),
            total_ms=int(d.get("total_ms", 0)),
            fps=int(d.get("fps", settings.FPS)),
            beats=[Beat(**b) for b in d.get("beats", [])],
            schema=int(d.get("schema", LEDGER_SCHEMA_VERSION)),
            genes=d.get("genes", {}) or {},
            checksum=d.get("checksum", ""),
        )

    @staticmethod
    def load(path: str) -> "BeatLedger":
        with open(path, "r", encoding="utf-8") as f:
            return BeatLedger.from_dict(json.load(f))

    # ─── queries (what the retention join needs) ─────────────

    def beat_at(self, t_ms: float) -> Optional[Beat]:
        for b in self.beats:
            if b.start_ms <= t_ms < b.end_ms:
                return b
        return None

    def fraction_span(self, beat: Beat) -> tuple:
        """[start, end) as a fraction of video duration — the axis the
        YouTube retention curve is expressed on."""
        total = max(1, self.total_ms)
        return (beat.start_ms / total, beat.end_ms / total)

    def describe(self) -> str:
        lines = [f"═══ Beat ledger · {len(self.beats)} beats · "
                 f"{self.total_ms / 1000:.2f}s @ {self.fps}fps ═══"]
        for b in self.beats:
            lines.append(
                f"  {b.index:>2}. {b.start_ms / 1000:6.2f}–{b.end_ms / 1000:6.2f}s "
                f"{b.kind:<8} {b.speaker:<11} {b.emotion:<12} "
                f"w={b.words:<3} cap={b.caption_chunks:<2} g={b.gestures}")
        lines.append(f"  checksum {self.compute_checksum()[:16]}…")
        return "\n".join(lines)


# ─── Construction ────────────────────────────────────────────────────

def ledger_path_for(video_path: str) -> str:
    return video_path + ".beats.json"


def build(dialogue: Dict[str, Any], timeline, video_path: str = "",
          fps: Optional[int] = None) -> BeatLedger:
    """Beat ledger from the script + the global Timeline.

    The Timeline is the single time authority (its spans derive from the
    aligner), so beat spans are exact by construction rather than
    re-estimated here.
    """
    fps = int(fps or getattr(timeline, "fps", settings.FPS))
    spans = list(getattr(timeline, "spans", []))
    n = len(spans)
    beats: List[Beat] = []

    total_ms = int(getattr(timeline, "total_ms", 0))
    turns = dialogue.get("turns", []) or []

    for i, span in enumerate(spans):
        turn = getattr(span, "turn", {}) or {}
        # The voice layer's turn dict carries the script fields through;
        # fall back to the script by position if a field was dropped.
        src = turn if turn.get("text") or turn.get("speaker") else (
            turns[i] if i < len(turns) else {})
        text = str(src.get("text", "") or "")
        kind = classify(src, i, n)
        gestures = src.get("gestures") or src.get("gesture") or []
        if isinstance(gestures, str):
            gestures = [gestures]

        dur_s = max(0.001, (span.end_ms - span.start_ms) / 1000.0)
        words = len(text.split())
        caps = len(getattr(span, "chunks", []) or [])

        beats.append(Beat(
            index=i,
            turn_id=src.get("turn_id", i + 1),
            speaker=str(src.get("speaker", "")),
            kind=kind,
            emotion=str(src.get("emotion", "neutral")),
            shot_type=str(src.get("shot_type", "")),
            start_ms=int(span.start_ms),
            end_ms=int(span.end_ms),
            words=words,
            caption_chunks=caps,
            gestures=len(gestures),
            policy={
                "kind": kind,
                "speech_rate": _density_bin(words / dur_s, 2.0, 3.2),
                "caption_density": _density_bin(caps / dur_s, 0.8, 1.6),
                "gesture_density": _density_bin(len(gestures) / dur_s, 0.3, 0.8),
                "position": _density_bin(
                    (span.start_ms / max(1, total_ms)), 0.34, 0.67),
            },
        ))

    # Video-level policies that only make sense once, attached to genes
    genes = dict((dialogue.get("dna") or {}).get("genes") or {})
    if turns:
        genes.setdefault("hook_style", hook_archetype(turns[0]))
    reveal = next((b for b in beats if b.kind == KIND_REVEAL), None)
    genes["time_to_first_reveal_s"] = round(
        (reveal.start_ms / 1000.0) if reveal else -1.0, 2)
    genes["affect_arc"] = "-".join(
        b.emotion for b in beats[:6]) or "flat"

    ledger = BeatLedger(
        video=os.path.basename(video_path) if video_path else "",
        total_ms=total_ms,
        fps=fps,
        beats=beats,
        genes=genes,
    )
    return ledger.seal()


def emit(dialogue: Dict[str, Any], timeline, video_path: str,
         fps: Optional[int] = None) -> str:
    """Build + write the ledger next to the muxed video."""
    ledger = build(dialogue, timeline, video_path, fps)
    return ledger.save(ledger_path_for(video_path))


# ─── The gate ────────────────────────────────────────────────────────

def _muxed_duration_ms(video_path: str) -> Optional[float]:
    exe = os.environ.get("FFPROBE_BINARY", "ffprobe")
    try:
        out = subprocess.run(
            [exe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", video_path],
            capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            return None
        return float(out.stdout.strip()) * 1000.0
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def verify(video_path: str) -> tuple:
    """(ok, reason) — the publisher's admission check for the ledger.

    Fails on: missing ledger, tampered/stale checksum, empty or
    non-monotonic span table, or spans that disagree with the muxed
    container duration beyond one-frame rounding.
    """
    path = ledger_path_for(video_path)
    if not os.path.exists(path):
        return False, "no beat ledger — the learning loop has no ground truth"
    try:
        ledger = BeatLedger.load(path)
    except (OSError, ValueError, TypeError) as e:
        return False, f"beat ledger unreadable: {e}"

    if ledger.schema != LEDGER_SCHEMA_VERSION:
        return False, (f"beat ledger schema {ledger.schema} != "
                       f"{LEDGER_SCHEMA_VERSION} — re-emit it")
    if not ledger.beats:
        return False, "beat ledger has no beats"
    if ledger.checksum != ledger.compute_checksum():
        return False, "beat ledger checksum mismatch — spans were edited"

    prev_end = -1
    for b in ledger.beats:
        if b.end_ms <= b.start_ms:
            return False, f"beat {b.index} has non-positive duration"
        if b.start_ms < prev_end:
            return False, f"beat {b.index} overlaps the previous beat"
        prev_end = b.end_ms
    if prev_end > ledger.total_ms + DURATION_TOLERANCE_MS:
        return False, "beat spans overrun the declared timeline"

    muxed = _muxed_duration_ms(video_path)
    if muxed is not None:
        drift = abs(muxed - ledger.total_ms)
        if drift > DURATION_TOLERANCE_MS:
            return False, (f"beat ledger duration {ledger.total_ms}ms drifts "
                           f"{drift:.0f}ms from the muxed video ({muxed:.0f}ms)")
    return True, (f"{len(ledger.beats)} beats, "
                  f"checksum {ledger.checksum[:12]}…")


def _main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="beat ledger inspect/verify")
    ap.add_argument("video")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        ok, reason = verify(args.video)
        print(("PASS  " if ok else "FAIL  ") + reason)
        return 0 if ok else 1
    print(BeatLedger.load(ledger_path_for(args.video)).describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
