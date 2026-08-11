"""
JEEVidya V5 — Retention Flywheel (Tier 4, the crown jewel)
══════════════════════════════════════════════════════════
Closes the loop from published videos back into future DNA:

  STORE   SQLite ledger: (video_id, DNA genes) + periodic stat snapshots
          (views, likes, comments) pulled from the free YouTube Data API.
  SCORE   engagement proxy per video: like-rate + comment-rate + view
          velocity, normalized against the channel's own median.
          (True retention curves need OAuth Analytics; the proxy uses
          only an API key and correlates well enough to steer genes.)
  LEARN   Thompson-sampling bandit (pure numpy Beta draws) over binned
          gene arms: energy{lo,mid,hi} × hook_style × cut_rate{lo,mid,hi}.
          Week 1 it guesses; week 6 it KNOWS what your audience rewards.
  ACT     recommend() → gene overrides the BatchFactory feeds into
          VisualDNA.from_title(tuning=...).
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import settings

DB_PATH = os.path.join(settings.PROJECT_ROOT, ".cache", "flywheel.db")
STATS_URL = ("https://www.googleapis.com/youtube/v3/videos"
             "?part=statistics&id={ids}&key={key}")

# Gene bins → bandit arms
ENERGY_BINS = [(0.35, 0.55, "lo"), (0.55, 0.78, "mid"), (0.78, 1.01, "hi")]
CUT_BINS = [(0.6, 0.95, "lo"), (0.95, 1.25, "mid"), (1.25, 1.61, "hi")]
# Voice-emotion reference intensity (Terminal Plan Part X) — learnable
# once the cloned-voice source is live; harmless before that.
VOICE_INTENSITY_BINS = [(0.0, 0.45, "subtle"), (0.45, 0.75, "natural"),
                        (0.75, 1.01, "strong")]
BIN_CENTERS = {
    "energy": {"lo": 0.45, "mid": 0.66, "hi": 0.88},
    "cut_rate": {"lo": 0.8, "mid": 1.1, "hi": 1.4},
    "voice_intensity": {"subtle": 0.35, "natural": 0.6, "strong": 0.85},
}

# The full arm space (Part XII §12.2). Continuous genes are binned;
# categorical genes are used verbatim. Adding a gene here is the ONLY
# thing needed to make it learnable — no other code changes.
CONTINUOUS_DIMS = {
    "energy": (ENERGY_BINS, 0.66),
    "cut_rate": (CUT_BINS, 1.0),
    "voice_intensity": (VOICE_INTENSITY_BINS, 0.6),
}
CATEGORICAL_DIMS = {
    "hook_style": "question",        # first-2-seconds treatment
    "thumb_style": "default",        # thumbnail frame × title treatment
    "affect_arc": "flat",            # emotional trajectory template
}
ARM_DIMS = tuple(CONTINUOUS_DIMS) + tuple(CATEGORICAL_DIMS)


def _bin_of(value: float, bins) -> str:
    for lo, hi, name in bins:
        if lo <= value < hi:
            return name
    return bins[-1][2]


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class Flywheel:

    def __init__(self, db_path: str = DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.db = sqlite3.connect(db_path)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                title TEXT,
                published_at REAL,
                genes TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS snapshots (
                video_id TEXT NOT NULL,
                ts REAL NOT NULL,
                views INTEGER, likes INTEGER, comments INTEGER,
                PRIMARY KEY (video_id, ts)
            );
        """)
        self.db.commit()

    # ─── STORE ─────────────────────────────────────────────

    def register_video(self, video_id: str, meta: Dict[str, Any]) -> None:
        genes = (meta.get("dna") or {}).get("genes") or {}
        self.db.execute(
            "INSERT OR REPLACE INTO videos VALUES (?,?,?,?)",
            (video_id, meta.get("youtube_title", ""), time.time(),
             json.dumps(genes)))
        self.db.commit()
        print(f"  [Flywheel] tracking {video_id} "
              f"({len(genes)} genes)")

    def pull_stats(self, api_key: Optional[str] = None) -> int:
        """Snapshot current stats for every tracked video (1 API unit/50)."""
        api_key = api_key or os.environ.get("YOUTUBE_API_KEY") \
            or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("  [Flywheel] set YOUTUBE_API_KEY to pull stats")
            return 0
        ids = [r[0] for r in
               self.db.execute("SELECT video_id FROM videos").fetchall()]
        pulled = 0
        for i in range(0, len(ids), 50):
            chunk = ids[i:i + 50]
            url = STATS_URL.format(ids=urllib.parse.quote(",".join(chunk)),
                                   key=api_key)
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            now = time.time()
            for item in data.get("items", []):
                s = item.get("statistics", {})
                self.db.execute(
                    "INSERT OR REPLACE INTO snapshots VALUES (?,?,?,?,?)",
                    (item["id"], now,
                     int(s.get("viewCount", 0)),
                     int(s.get("likeCount", 0)),
                     int(s.get("commentCount", 0))))
                pulled += 1
        self.db.commit()
        print(f"  [Flywheel] pulled stats for {pulled} videos")
        return pulled

    # ─── SCORE ─────────────────────────────────────────────

    def _performance(self) -> List[Tuple[Dict[str, Any], float]]:
        """(genes, engagement score) per video with ≥1 snapshot."""
        rows = self.db.execute("""
            SELECT v.video_id, v.genes, v.published_at,
                   s.views, s.likes, s.comments, MAX(s.ts)
            FROM videos v JOIN snapshots s ON s.video_id = v.video_id
            GROUP BY v.video_id
        """).fetchall()
        out = []
        for _vid, genes_json, pub, views, likes, comments, ts in rows:
            views = max(1, views or 0)
            age_days = max(0.25, (ts - (pub or ts)) / 86400)
            like_rate = (likes or 0) / views
            comment_rate = (comments or 0) / views
            velocity = views / age_days
            score = (like_rate * 8 + comment_rate * 25
                     + np.log1p(velocity) / 12)
            out.append((json.loads(genes_json), float(score)))
        return out

    # ─── LEARN: Thompson sampling over gene arms ───────────

    def _arm_stats(self) -> Dict[str, Dict[str, List[int]]]:
        """Per gene-dimension: arm → [successes, failures].
        Success = video scored above the channel median."""
        perf = self._performance()
        if len(perf) < 4:
            return {}
        median = float(np.median([s for _, s in perf]))
        arms: Dict[str, Dict[str, List[int]]] = {d: {} for d in ARM_DIMS}
        for genes, score in perf:
            win = score > median
            keys: Dict[str, str] = {}
            for dim, (bins, default) in CONTINUOUS_DIMS.items():
                if dim not in genes:
                    continue
                keys[dim] = _bin_of(_as_float(genes.get(dim), default), bins)
            for dim, default in CATEGORICAL_DIMS.items():
                if dim not in genes:
                    continue
                keys[dim] = str(genes.get(dim, default))
            for dim, arm in keys.items():
                sf = arms[dim].setdefault(arm, [0, 0])
                sf[0 if win else 1] += 1
        return {d: a for d, a in arms.items() if a}

    def recommend(self, rng: Optional[np.random.Generator] = None
                  ) -> Dict[str, Any]:
        """Thompson-sample the best gene overrides. Empty dict = not
        enough data yet (the DNA keeps exploring on its own seed)."""
        arms = self._arm_stats()
        if not arms:
            return {}
        rng = rng or np.random.default_rng()
        tuning: Dict[str, Any] = {}
        for dim, arm_map in arms.items():
            if not arm_map:
                continue
            best_arm, best_draw = None, -1.0
            for arm, (wins, losses) in arm_map.items():
                draw = float(rng.beta(1 + wins, 1 + losses))
                if draw > best_draw:
                    best_arm, best_draw = arm, draw
            if best_arm is None:
                continue
            centers = BIN_CENTERS.get(dim)
            tuning[dim] = centers[best_arm] if centers else best_arm
        return tuning

    def recommend_beats(self, rng: Optional[np.random.Generator] = None
                        ) -> Dict[str, Any]:
        """Beat-level policy from the retention learner (Part XIX). The
        video-level bandit steers genes; this steers *seconds*."""
        try:
            from factory.retention import RetentionEngine
            return RetentionEngine(self.db_path).recommend(rng)
        except Exception as e:                   # noqa: BLE001 — optional layer
            print(f"  [Flywheel] beat policy unavailable: {e}")
            return {}

    # ─── Reporting (the dashboard's data) ──────────────────

    def report(self) -> Dict[str, Any]:
        arms = self._arm_stats()
        n_videos = self.db.execute(
            "SELECT COUNT(*) FROM videos").fetchone()[0]
        n_snaps = self.db.execute(
            "SELECT COUNT(*) FROM snapshots").fetchone()[0]
        posterior = {
            dim: {arm: {"wins": wl[0], "losses": wl[1],
                        "mean": round((1 + wl[0]) / (2 + wl[0] + wl[1]), 3)}
                  for arm, wl in arm_map.items()}
            for dim, arm_map in arms.items()}
        return {"videos": n_videos, "snapshots": n_snaps,
                "posterior": posterior,
                "recommendation": self.recommend(
                    np.random.default_rng(0)) or "collecting data",
                "beat_policy": self.recommend_beats(
                    np.random.default_rng(0)) or "collecting data"}

    def describe(self) -> str:
        r = self.report()
        lines = [f"═══ Flywheel · {r['videos']} videos · "
                 f"{r['snapshots']} snapshots ═══"]
        for dim, arm_map in r["posterior"].items():
            lines.append(f"  {dim}:")
            for arm, st in sorted(arm_map.items(),
                                  key=lambda kv: -kv[1]["mean"]):
                lines.append(f"    {arm:<14} {st['mean']:.3f} "
                             f"({st['wins']}W/{st['losses']}L)")
        lines.append(f"  → next-batch tuning: {r['recommendation']}")
        lines.append(f"  → next-batch beat policy: {r['beat_policy']}")
        return "\n".join(lines)
