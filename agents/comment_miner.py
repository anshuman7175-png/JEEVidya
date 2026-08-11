"""
JEEVidya — Comment Miner (Terminal Plan, Part XIX)
══════════════════════════════════════════════════
"What your actual audience is confused about" — extracted, clustered and
ranked, so topic selection stops being guesswork.

Pipeline (free YouTube Data API key only, no OAuth):

    FETCH     commentThreads.list over tracked videos (1 unit/page).
    FILTER    keep questions and misconception markers (Hinglish +
              Devanagari + English), drop praise/spam/emoji-only.
    VECTORIZE character 3-gram hashing → L2-normalized vectors. Character
              n-grams are the right call for Hinglish: "velocity",
              "vilocity" and "वेलोसिटी" all share structure that word
              tokens would miss, and it needs no vocabulary file.
    CLUSTER   deterministic greedy agglomeration by cosine similarity
              (seeded, order-normalized by sorting on text hash → the
              same comments always produce the same clusters, Law 4).
    RANK      demand = cluster size × recency × like weight × question
              purity, then intersected with the syllabus graph
              (`factory/syllabus.py`) so a ranked TOPIC QUEUE comes out,
              not just a word cloud.

Offline-first: with no API key it mines a local JSONL/JSON dump, so the
whole ranking path is testable and deterministic without network.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from config import settings

DB_PATH = os.path.join(settings.PROJECT_ROOT, ".cache", "flywheel.db")
QUEUE_PATH = os.path.join(settings.PROJECT_ROOT, ".cache", "topic_queue.json")
THREADS_URL = ("https://www.googleapis.com/youtube/v3/commentThreads"
               "?part=snippet&videoId={vid}&maxResults=100"
               "&order=relevance&textFormat=plainText&key={key}")

N_HASH_DIMS = 512
NGRAM = 3
SIM_THRESHOLD = 0.42          # cosine floor for joining a cluster
MIN_CHARS = 12
MAX_CHARS = 400

# Doubt signal: interrogatives + misconception markers, three scripts.
_QUESTION_CUES = (
    "?", "kya", "kyu", "kyun", "kaise", "kitna", "kitni", "kab", "kaun",
    "why", "how", "what", "when", "which", "doubt", "confus", "samajh nahi",
    "samjha nahi", "nahi aaya", "clear nahi", "explain", "sikhao", "batao",
    "क्या", "क्यों", "कैसे", "कितना", "समझ", "बताओ", "डाउट",
)
_MISCONCEPTION_CUES = (
    "but ", "lekin", "magar", "galat", "wrong", "isn't it", "shouldn't",
    "nahi hona chahiye", "answer to", "sir yeh", "actually",
)
_NOISE = re.compile(r"(https?://\S+)|([\U0001F300-\U0001FAFF])")
_WS = re.compile(r"\s+")


# ═══════════════════════════════════════════
# Text → vector
# ═══════════════════════════════════════════

def normalize(text: str) -> str:
    text = _NOISE.sub(" ", str(text or "")).lower()
    return _WS.sub(" ", text).strip()


def is_doubt(text: str) -> Tuple[bool, float]:
    """(keep, question purity). Purity separates a real doubt from a
    compliment that happens to contain 'kaise'."""
    low = normalize(text)
    if not (MIN_CHARS <= len(low) <= MAX_CHARS):
        return False, 0.0
    q_hits = sum(1 for c in _QUESTION_CUES if c in low)
    m_hits = sum(1 for c in _MISCONCEPTION_CUES if c in low)
    if q_hits == 0 and m_hits == 0:
        return False, 0.0
    purity = min(1.0, 0.34 * q_hits + 0.22 * m_hits
                 + (0.25 if low.endswith("?") else 0.0))
    return True, purity


def hash_vector(text: str, dims: int = N_HASH_DIMS) -> np.ndarray:
    """Character-n-gram hashing vector. Deterministic across processes
    (blake2b, not Python's salted hash())."""
    v = np.zeros(dims, dtype=np.float64)
    low = normalize(text)
    if not low:
        return v
    padded = f" {low} "
    for i in range(len(padded) - NGRAM + 1):
        gram = padded[i:i + NGRAM]
        h = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "big") % dims
        sign = 1.0 if h[4] & 1 else -1.0
        v[idx] += sign
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


# ═══════════════════════════════════════════
# Clustering
# ═══════════════════════════════════════════

@dataclass
class Comment:
    text: str
    likes: int = 0
    published: float = 0.0
    video_id: str = ""
    purity: float = 0.0

    @property
    def key(self) -> str:
        return hashlib.blake2b(normalize(self.text).encode(),
                               digest_size=8).hexdigest()


@dataclass
class Cluster:
    centroid: np.ndarray
    members: List[Comment] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.members)

    def add(self, c: Comment, vec: np.ndarray) -> None:
        n = len(self.members)
        self.centroid = (self.centroid * n + vec) / (n + 1)
        norm = float(np.linalg.norm(self.centroid))
        if norm > 0:
            self.centroid = self.centroid / norm
        self.members.append(c)

    def exemplar(self) -> Comment:
        """The most-liked, highest-purity member: what the operator reads."""
        return max(self.members,
                   key=lambda m: (m.purity * 2 + np.log1p(m.likes)))

    def demand(self, now: Optional[float] = None) -> float:
        now = now or time.time()
        recency = 0.0
        for m in self.members:
            age_days = max(0.5, (now - (m.published or now)) / 86400.0)
            recency += 1.0 / (1.0 + age_days / 30.0)
        likes = float(np.log1p(sum(m.likes for m in self.members)))
        purity = float(np.mean([m.purity for m in self.members]))
        return round(self.size * 0.6 + recency * 0.8 + likes * 0.5
                     + purity * 1.2, 4)


def cluster_comments(comments: Sequence[Comment],
                     threshold: float = SIM_THRESHOLD) -> List[Cluster]:
    """Deterministic greedy agglomeration.

    Insertion order is normalized by sorting on a content hash, so the
    same comment set clusters identically no matter what order the API
    returned it in (Law 4 applies to the learning loop too).
    """
    ordered = sorted(comments, key=lambda c: c.key)
    clusters: List[Cluster] = []
    for c in ordered:
        vec = hash_vector(c.text)
        if not np.any(vec):
            continue
        best, best_sim = None, threshold
        for cl in clusters:
            sim = float(np.dot(cl.centroid, vec))
            if sim > best_sim:
                best, best_sim = cl, sim
        if best is None:
            clusters.append(Cluster(centroid=vec, members=[c]))
        else:
            best.add(c, vec)
    clusters.sort(key=lambda cl: (-cl.demand(), cl.exemplar().key))
    return clusters


# ═══════════════════════════════════════════
# Miner
# ═══════════════════════════════════════════

class CommentMiner:

    def __init__(self, db_path: str = DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS comments (
                comment_key TEXT PRIMARY KEY,
                video_id TEXT, text TEXT NOT NULL,
                likes INTEGER, published REAL, purity REAL
            );
        """)
        self.db.commit()

    # ─── FETCH ─────────────────────────────────────────────

    def fetch(self, video_ids: Optional[Sequence[str]] = None,
              api_key: Optional[str] = None) -> int:
        api_key = api_key or os.environ.get("YOUTUBE_API_KEY")
        if not api_key:
            print("  [Miner] set YOUTUBE_API_KEY (or use ingest_dump)")
            return 0
        ids = list(video_ids or [
            r[0] for r in self.db.execute(
                "SELECT video_id FROM videos").fetchall()]) \
            if _has_table(self.db, "videos") else list(video_ids or [])
        kept = 0
        for vid in ids:
            url = THREADS_URL.format(vid=urllib.parse.quote(vid), key=api_key)
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except Exception as e:               # noqa: BLE001 — isolate videos
                print(f"  [Miner] {vid}: {e}")
                continue
            for item in data.get("items", []):
                sn = (item.get("snippet", {})
                      .get("topLevelComment", {}).get("snippet", {}))
                kept += self._store(Comment(
                    text=sn.get("textOriginal", ""),
                    likes=int(sn.get("likeCount", 0) or 0),
                    published=_iso_to_epoch(sn.get("publishedAt", "")),
                    video_id=vid))
        self.db.commit()
        print(f"  [Miner] stored {kept} doubt comments")
        return kept

    def ingest_dump(self, path: str) -> int:
        """Offline path: JSON list or JSONL of {text, likes, publishedAt}."""
        kept = 0
        for rec in _read_records(path):
            kept += self._store(Comment(
                text=rec.get("text", "") or rec.get("textOriginal", ""),
                likes=int(rec.get("likes", 0) or 0),
                published=_iso_to_epoch(rec.get("publishedAt", ""))
                or float(rec.get("published", 0) or 0),
                video_id=str(rec.get("video_id", "") or "")))
        self.db.commit()
        return kept

    def _store(self, c: Comment) -> int:
        keep, purity = is_doubt(c.text)
        if not keep:
            return 0
        c.purity = purity
        self.db.execute(
            "INSERT OR REPLACE INTO comments VALUES (?,?,?,?,?,?)",
            (c.key, c.video_id, c.text, c.likes, c.published, purity))
        return 1

    # ─── MINE ──────────────────────────────────────────────

    def comments(self) -> List[Comment]:
        return [Comment(text=t, likes=lk or 0, published=pb or 0.0,
                        video_id=vid or "", purity=pu or 0.0)
                for _k, vid, t, lk, pb, pu in self.db.execute(
                    "SELECT comment_key, video_id, text, likes, published, "
                    "purity FROM comments").fetchall()]

    def clusters(self) -> List[Cluster]:
        return cluster_comments(self.comments())

    def topic_queue(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Ranked queue for the scriptwriter: audience demand joined to
        the syllabus graph so every entry is a teachable topic with its
        prerequisites attached."""
        from factory.syllabus import Syllabus
        syl = Syllabus()
        out: List[Dict[str, Any]] = []
        for cl in self.clusters():
            ex = cl.exemplar()
            match = syl.match(" ".join(normalize(m.text)
                                       for m in cl.members[:12]))
            out.append({
                "question": ex.text.strip()[:220],
                "cluster_size": cl.size,
                "demand": cl.demand(),
                "topic": match.get("topic"),
                "chapter": match.get("chapter"),
                "prerequisites": match.get("prerequisites", []),
                "confidence": match.get("confidence", 0.0),
            })
        out.sort(key=lambda r: (-(r["demand"] * (0.5 + r["confidence"])),
                               r["question"]))
        return out[:limit]

    def save_queue(self, path: str = QUEUE_PATH, limit: int = 20) -> str:
        queue = self.topic_queue(limit)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"generated_at": time.time(), "queue": queue},
                      f, indent=2, ensure_ascii=False)
        return path

    def describe(self, limit: int = 10) -> str:
        queue = self.topic_queue(limit)
        n = self.db.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
        lines = [f"═══ Comment miner · {n} doubts · "
                 f"{len(queue)} ranked topics ═══"]
        for i, r in enumerate(queue, 1):
            lines.append(f"  {i:>2}. [{r['demand']:.2f}] "
                         f"{r['topic'] or '(unmapped)'} "
                         f"×{r['cluster_size']}")
            lines.append(f"      “{r['question'][:90]}”")
        return "\n".join(lines)


# ─── helpers ─────────────────────────────────────────────────────────

def _has_table(db: sqlite3.Connection, name: str) -> bool:
    return bool(db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,)).fetchone())


def _iso_to_epoch(iso: str) -> float:
    if not iso:
        return 0.0
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return time.mktime(time.strptime(iso, fmt))
        except ValueError:
            continue
    return 0.0


def _read_records(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            yield from json.load(f)
            return
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="mine audience doubts")
    ap.add_argument("--dump", help="offline JSON/JSONL comment export")
    ap.add_argument("--fetch", action="store_true", help="pull via Data API")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()
    miner = CommentMiner()
    if args.dump:
        print(f"  [Miner] ingested {miner.ingest_dump(args.dump)} doubts")
    if args.fetch:
        miner.fetch()
    print(miner.describe(args.limit))
    miner.save_queue(limit=max(args.limit, 20))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
