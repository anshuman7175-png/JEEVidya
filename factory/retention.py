"""
JEEVidya — Retention Intelligence (Terminal Plan, Part XIX)
═══════════════════════════════════════════════════════════
The flywheel learns per VIDEO. YouTube tells you per SECOND. This module
closes that gap: it joins the per-moment audience-retention curve onto
the beat ledger (`factory/beats.py`) so every *type* of beat accumulates
its own evidence, and extends Thompson sampling from video-level genes
to beat-level policies.

    PULL    YouTube Analytics API v2 (free, OAuth) →
            elapsedVideoTimeRatio × audienceWatchRatio, 100 samples.
            No OAuth? `ingest_curve()` takes a CSV/JSON export instead;
            everything downstream is identical.
    JOIN    resample the curve onto each beat's [start,end) fraction
            span → per-beat retention level AND slope (the honest
            signal: absolute level decays monotonically for everyone,
            slope is what a beat actually *causes*).
    LEARN   Beta-Bernoulli Thompson sampling per policy dimension
            (kind × speech_rate × caption_density × gesture_density ×
            position, plus video-level hook archetype and reveal
            timing). Success = beat slope above the video's own median
            slope, so video-level popularity can't contaminate
            beat-level credit.
    ACT     `recommend()` → beat-policy targets for the scriptwriter,
            always emitted as an A/B ARM (§XIX counterfactual
            discipline): a learned policy never flips globally, and the
            ledger records which policy version produced every video so
            regressions stay attributable.

Pure stdlib + numpy. Same SQLite file as the video-level flywheel so one
`.cache/flywheel.db` is the whole channel's memory.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import settings
from factory.beats import BeatLedger, ledger_path_for

DB_PATH = os.path.join(settings.PROJECT_ROOT, ".cache", "flywheel.db")

ANALYTICS_URL = "https://youtubeanalytics.googleapis.com/v2/reports"
ANALYTICS_SCOPE = "https://www.googleapis.com/auth/yt-analytics.readonly"

# The policy dimensions the beat-level bandit samples over.
BEAT_DIMS = ("kind", "speech_rate", "caption_density",
             "gesture_density", "position")
VIDEO_DIMS = ("hook_style", "reveal_timing", "affect_arc_head")

CURVE_SAMPLES = 100          # YouTube's native retention resolution
MIN_BEATS_TO_LEARN = 24      # ≈ 3 videos before any posterior is served
POLICY_VERSION = 1


# ═══════════════════════════════════════════
# Curve algebra
# ═══════════════════════════════════════════

def resample_curve(points: Sequence[Tuple[float, float]],
                   n: int = CURVE_SAMPLES) -> np.ndarray:
    """(fraction, watch_ratio) samples → a dense curve on [0,1]."""
    if not points:
        return np.zeros(n, dtype=np.float64)
    arr = np.asarray(sorted(points), dtype=np.float64)
    xs = np.clip(arr[:, 0], 0.0, 1.0)
    ys = np.clip(arr[:, 1], 0.0, 2.0)
    grid = np.linspace(0.0, 1.0, n)
    return np.interp(grid, xs, ys)


def span_metrics(curve: np.ndarray, frac_start: float,
                 frac_end: float) -> Tuple[float, float]:
    """(mean level, slope) of the retention curve across a beat span.

    Slope is per unit of video fraction; a beat that holds people has a
    slope shallower than the video's own median — that is the credit
    signal, not the level (which only ever falls).
    """
    n = len(curve)
    if n < 2:
        return 0.0, 0.0
    i0 = int(np.floor(np.clip(frac_start, 0.0, 1.0) * (n - 1)))
    i1 = int(np.ceil(np.clip(frac_end, 0.0, 1.0) * (n - 1)))
    i1 = max(i1, i0 + 1)
    seg = curve[i0:i1 + 1]
    if len(seg) < 2:
        return float(seg.mean()), 0.0
    x = np.linspace(frac_start, frac_end, len(seg))
    slope = float(np.polyfit(x, seg, 1)[0])
    return float(seg.mean()), slope


# ═══════════════════════════════════════════
# Store + learner
# ═══════════════════════════════════════════

class RetentionEngine:
    """Beat ledger × retention curve → beat-level posteriors."""

    def __init__(self, db_path: str = DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS beat_spans (
                video_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                kind TEXT, policy TEXT NOT NULL,
                frac_start REAL NOT NULL, frac_end REAL NOT NULL,
                PRIMARY KEY (video_id, idx)
            );
            CREATE TABLE IF NOT EXISTS retention_curves (
                video_id TEXT PRIMARY KEY,
                pulled_at REAL NOT NULL,
                curve TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS video_policy (
                video_id TEXT PRIMARY KEY,
                policy_version INTEGER NOT NULL,
                arm TEXT,
                genes TEXT NOT NULL,
                registered_at REAL NOT NULL
            );
        """)
        self.db.commit()

    # ─── STORE ─────────────────────────────────────────────

    def register_ledger(self, video_id: str, ledger: BeatLedger,
                        arm: Optional[str] = None) -> int:
        """Pin a published video's beats. Called by the publisher right
        after upload, from the gate-verified ledger."""
        rows = []
        for b in ledger.beats:
            f0, f1 = ledger.fraction_span(b)
            rows.append((video_id, b.index, b.kind,
                         json.dumps(b.policy, sort_keys=True), f0, f1))
        self.db.executemany(
            "INSERT OR REPLACE INTO beat_spans VALUES (?,?,?,?,?,?)", rows)
        genes = dict(ledger.genes)
        genes["reveal_timing"] = _reveal_bin(
            float(genes.get("time_to_first_reveal_s", -1.0)))
        genes["affect_arc_head"] = "-".join(
            str(genes.get("affect_arc", "flat")).split("-")[:2])
        self.db.execute(
            "INSERT OR REPLACE INTO video_policy VALUES (?,?,?,?,?)",
            (video_id, POLICY_VERSION, arm, json.dumps(genes), time.time()))
        self.db.commit()
        print(f"  [Retention] pinned {len(rows)} beats for {video_id}"
              + (f" (arm {arm})" if arm else ""))
        return len(rows)

    def register_from_video(self, video_id: str, video_path: str,
                            arm: Optional[str] = None) -> int:
        return self.register_ledger(
            video_id, BeatLedger.load(ledger_path_for(video_path)), arm)

    def ingest_curve(self, video_id: str,
                     points: Sequence[Tuple[float, float]]) -> None:
        """Manual/export path: (elapsed_fraction, watch_ratio) pairs."""
        curve = resample_curve(points)
        self.db.execute(
            "INSERT OR REPLACE INTO retention_curves VALUES (?,?,?)",
            (video_id, time.time(), json.dumps([round(v, 5) for v in curve])))
        self.db.commit()

    # ─── PULL (free YouTube Analytics API, OAuth) ──────────

    def pull_curves(self, video_ids: Optional[Sequence[str]] = None) -> int:
        """Fetch retention curves for tracked videos. Requires the
        Analytics scope on the publisher's OAuth token; silently a no-op
        without it, so the DAG never blocks on network or consent."""
        creds = self._analytics_credentials()
        if creds is None:
            print("  [Retention] no Analytics OAuth token; use ingest_curve()")
            return 0
        ids = list(video_ids or [
            r[0] for r in self.db.execute(
                "SELECT DISTINCT video_id FROM beat_spans").fetchall()])
        pulled = 0
        for vid in ids:
            points = self._fetch_curve(creds, vid)
            if points:
                self.ingest_curve(vid, points)
                pulled += 1
        print(f"  [Retention] pulled {pulled}/{len(ids)} retention curves")
        return pulled

    def _analytics_credentials(self):
        try:
            from google.oauth2.credentials import Credentials
        except ImportError:
            return None
        from factory.publisher import TOKEN_PATH
        if not os.path.exists(TOKEN_PATH):
            return None
        try:
            creds = Credentials.from_authorized_user_file(
                TOKEN_PATH, [ANALYTICS_SCOPE])
        except (OSError, ValueError):
            return None
        if not creds or not creds.valid:
            try:
                from google.auth.transport.requests import Request
                creds.refresh(Request())
            except Exception:                    # noqa: BLE001 — offline is fine
                return None
        return creds

    def _fetch_curve(self, creds, video_id: str
                     ) -> List[Tuple[float, float]]:
        try:
            from googleapiclient.discovery import build
        except ImportError:
            return []
        try:
            api = build("youtubeAnalytics", "v2", credentials=creds)
            resp = api.reports().query(
                ids="channel==MINE",
                startDate="2005-01-01",
                endDate=time.strftime("%Y-%m-%d"),
                metrics="audienceWatchRatio",
                dimensions="elapsedVideoTimeRatio",
                filters=f"video=={video_id}",
                sort="elapsedVideoTimeRatio").execute()
        except Exception as e:                   # noqa: BLE001 — isolate videos
            print(f"  [Retention] {video_id}: {e}")
            return []
        return [(float(r[0]), float(r[1]))
                for r in resp.get("rows", []) if len(r) >= 2]

    # ─── JOIN ──────────────────────────────────────────────

    def join(self) -> List[Dict[str, Any]]:
        """Every beat that has a curve → its policy + retention deltas.

        Both metrics are expressed RELATIVE to the video's own median so
        a viral video cannot make its mediocre beats look good.
        """
        curves = {vid: np.asarray(json.loads(c), dtype=np.float64)
                  for vid, _ts, c in self.db.execute(
                      "SELECT video_id, pulled_at, curve "
                      "FROM retention_curves").fetchall()}
        if not curves:
            return []
        out: List[Dict[str, Any]] = []
        for vid, curve in curves.items():
            rows = self.db.execute(
                "SELECT idx, kind, policy, frac_start, frac_end "
                "FROM beat_spans WHERE video_id=? ORDER BY idx",
                (vid,)).fetchall()
            if not rows:
                continue
            genes_row = self.db.execute(
                "SELECT genes, arm FROM video_policy WHERE video_id=?",
                (vid,)).fetchone()
            genes = json.loads(genes_row[0]) if genes_row else {}
            arm = genes_row[1] if genes_row else None

            metrics = [span_metrics(curve, f0, f1) for _i, _k, _p, f0, f1 in rows]
            levels = np.array([m[0] for m in metrics])
            slopes = np.array([m[1] for m in metrics])
            med_level = float(np.median(levels))
            med_slope = float(np.median(slopes))
            for (idx, kind, policy, f0, f1), (lvl, slp) in zip(rows, metrics, strict=True):
                out.append({
                    "video_id": vid, "index": idx, "kind": kind,
                    "policy": json.loads(policy),
                    "genes": genes, "arm": arm,
                    "frac_start": f0, "frac_end": f1,
                    "level": lvl, "slope": slp,
                    "level_delta": lvl - med_level,
                    # slope is negative-going; ABOVE median slope = holds
                    "slope_delta": slp - med_slope,
                })
        return out

    # ─── LEARN ─────────────────────────────────────────────

    def _arm_stats(self) -> Dict[str, Dict[str, List[int]]]:
        joined = self.join()
        if len(joined) < MIN_BEATS_TO_LEARN:
            return {}
        arms: Dict[str, Dict[str, List[int]]] = {
            d: {} for d in BEAT_DIMS + VIDEO_DIMS}
        for rec in joined:
            win = rec["slope_delta"] > 0.0
            for dim in BEAT_DIMS:
                val = rec["policy"].get(dim)
                if val is None:
                    continue
                sf = arms[dim].setdefault(str(val), [0, 0])
                sf[0 if win else 1] += 1
            for dim in VIDEO_DIMS:
                val = rec["genes"].get(dim)
                if val is None:
                    continue
                sf = arms[dim].setdefault(str(val), [0, 0])
                sf[0 if win else 1] += 1
        return {d: a for d, a in arms.items() if a}

    def recommend(self, rng: Optional[np.random.Generator] = None
                  ) -> Dict[str, Any]:
        """Thompson-sample the best beat policy per dimension. Empty dict
        until MIN_BEATS_TO_LEARN beats of evidence exist."""
        arms = self._arm_stats()
        if not arms:
            return {}
        rng = rng or np.random.default_rng()
        best: Dict[str, Any] = {}
        for dim, arm_map in arms.items():
            top, top_draw = None, -1.0
            for arm, (w, l) in arm_map.items():
                draw = float(rng.beta(1 + w, 1 + l))
                if draw > top_draw:
                    top, top_draw = arm, draw
            if top is not None:
                best[dim] = top
        return best

    def assign_ab(self, rng: Optional[np.random.Generator] = None
                  ) -> Tuple[str, Dict[str, Any]]:
        """Counterfactual discipline: half the batch renders the learned
        policy ("treatment"), half keeps the incumbent ("control"). The
        arm label is stored with the video so credit is attributable."""
        rng = rng or np.random.default_rng()
        policy = self.recommend(rng)
        if not policy or rng.random() < 0.5:
            return "control", {}
        return f"treatment.v{POLICY_VERSION}", policy

    # ─── Reporting ─────────────────────────────────────────

    def weakest_moments(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Which SECOND loses viewers — the operator-facing payoff."""
        joined = self.join()
        joined.sort(key=lambda r: r["slope_delta"])
        return joined[:limit]

    def report(self) -> Dict[str, Any]:
        arms = self._arm_stats()
        n_beats = self.db.execute(
            "SELECT COUNT(*) FROM beat_spans").fetchone()[0]
        n_curves = self.db.execute(
            "SELECT COUNT(*) FROM retention_curves").fetchone()[0]
        posterior = {
            dim: {arm: {"wins": wl[0], "losses": wl[1],
                        "mean": round((1 + wl[0]) / (2 + wl[0] + wl[1]), 3)}
                  for arm, wl in arm_map.items()}
            for dim, arm_map in arms.items()}
        return {
            "beats": n_beats, "curves": n_curves,
            "policy_version": POLICY_VERSION,
            "posterior": posterior,
            "weakest": [
                {"video": r["video_id"], "beat": r["index"],
                 "kind": r["kind"], "slope_delta": round(r["slope_delta"], 4)}
                for r in self.weakest_moments()],
            "recommendation": self.recommend(
                np.random.default_rng(0)) or "collecting data",
        }

    def describe(self) -> str:
        r = self.report()
        lines = [f"═══ Retention · {r['beats']} beats · "
                 f"{r['curves']} curves · policy v{r['policy_version']} ═══"]
        for dim, arm_map in r["posterior"].items():
            lines.append(f"  {dim}:")
            for arm, st in sorted(arm_map.items(),
                                  key=lambda kv: -kv[1]["mean"]):
                lines.append(f"    {arm:<16} {st['mean']:.3f} "
                             f"({st['wins']}W/{st['losses']}L)")
        if r["weakest"]:
            lines.append("  weakest beats (where viewers leave):")
            for w in r["weakest"]:
                lines.append(f"    {w['video']} #{w['beat']:<3} "
                             f"{w['kind']:<8} Δslope {w['slope_delta']:+.4f}")
        lines.append(f"  → next-batch beat policy: {r['recommendation']}")
        return "\n".join(lines)


def _reveal_bin(seconds: float) -> str:
    if seconds < 0:
        return "none"
    if seconds < 4.0:
        return "early"
    if seconds < 10.0:
        return "mid"
    return "late"


def _main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="beat-level retention learner")
    ap.add_argument("--pull", action="store_true",
                    help="fetch retention curves via the Analytics API")
    ap.add_argument("--register", nargs=2, metavar=("VIDEO_ID", "VIDEO_PATH"),
                    help="pin a published video's beat ledger")
    args = ap.parse_args()
    eng = RetentionEngine()
    if args.register:
        eng.register_from_video(args.register[0], args.register[1])
    if args.pull:
        eng.pull_curves()
    print(eng.describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
