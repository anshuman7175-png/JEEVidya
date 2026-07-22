"""Tier 4 — Flywheel: the bandit must actually learn (offline, no API)."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factory.flywheel import Flywheel


def _seeded_flywheel(tmp_path):
    fw = Flywheel(db_path=str(tmp_path / "fly.db"))
    # Simulate 24 published videos: high-energy question-hooks WIN
    # (2x engagement), everything else limps.
    rng = np.random.default_rng(3)
    for i in range(24):
        energy = float(rng.uniform(0.36, 0.99))
        hook = ["question", "shock_number", "challenge",
                "myth_bust"][i % 4]
        genes = {"energy": energy, "cut_rate": 1.0, "hook_style": hook}
        vid = f"vid{i:02d}"
        fw.register_video(vid, {"youtube_title": f"t{i}",
                                "dna": {"genes": genes}})
        winner = energy > 0.78 and hook == "question"
        views = 12000 if winner else 3000
        likes = int(views * (0.09 if winner else 0.03))
        fw.db.execute("INSERT OR REPLACE INTO snapshots VALUES (?,?,?,?,?)",
                      (vid, time.time(), views, likes, likes // 4))
    fw.db.commit()
    return fw


def test_bandit_learns_the_winning_genes(tmp_path):
    fw = _seeded_flywheel(tmp_path)
    # Thompson sampling is stochastic — check the MODE of many draws
    votes = {"energy": {}, "hook_style": {}}
    rng = np.random.default_rng(0)
    for _ in range(200):
        rec = fw.recommend(rng)
        for dim in votes:
            v = rec.get(dim)
            if v is not None:
                votes[dim][v] = votes[dim].get(v, 0) + 1
    top_energy = max(votes["energy"], key=votes["energy"].get)
    top_hook = max(votes["hook_style"], key=votes["hook_style"].get)
    assert top_energy == 0.88, f"bandit picked energy={top_energy}"
    assert top_hook == "question", f"bandit picked hook={top_hook}"


def test_cold_start_recommends_nothing(tmp_path):
    fw = Flywheel(db_path=str(tmp_path / "cold.db"))
    assert fw.recommend() == {}          # keep exploring on raw DNA


def test_report_is_json_safe(tmp_path):
    fw = _seeded_flywheel(tmp_path)
    json.dumps(fw.report())              # must not raise
    assert "Flywheel" in fw.describe()
