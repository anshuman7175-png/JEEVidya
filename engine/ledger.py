"""
JEEVidya — Frame Hash Ledger (Terminal Plan, Part II §2.3)
══════════════════════════════════════════════════════════
Determinism made checkable: every render emits `frames.sha256` — one
hash per frame plus a rollup. `jvmake render --verify-determinism`
renders twice and diffs the ledgers; the golden corpus byte-compares
against a blessed ledger on every engine/pipeline commit.

The ledger is order-independent-mergeable: scene-parallel renders
(Singularity Part XX) hash frames independently and merge, so
multiprocessing cannot perturb the rollup.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


def hash_frame(rgba_bytes: bytes) -> str:
    return hashlib.sha256(rgba_bytes).hexdigest()


@dataclass
class FrameLedger:
    """Accumulates per-frame hashes; persists to frames.sha256 (JSON)."""
    fps: int
    size: Tuple[int, int]
    entries: Dict[int, str] = field(default_factory=dict)  # frame_idx → sha256

    def record(self, frame_idx: int, rgba_bytes: bytes) -> str:
        h = hash_frame(rgba_bytes)
        self.entries[frame_idx] = h
        return h

    def record_image(self, frame_idx: int, img) -> str:
        """PIL image convenience — hashes raw RGBA pixels, not the PNG
        encoding (PNG encoders may differ across library versions)."""
        return self.record(frame_idx, img.convert("RGBA").tobytes())

    @property
    def rollup(self) -> str:
        """Order-independent rollup: hash of sorted (idx, hash) pairs."""
        h = hashlib.sha256()
        for idx in sorted(self.entries):
            h.update(f"{idx}:{self.entries[idx]}".encode())
            h.update(b"\x1f")
        return h.hexdigest()

    def merge(self, other: "FrameLedger") -> None:
        """Merge a scene-parallel worker's ledger. Overlapping frame
        indices with different hashes are a hard error — two workers
        claiming the same frame differently IS the bug."""
        for idx, h in other.entries.items():
            if idx in self.entries and self.entries[idx] != h:
                raise ValueError(
                    f"Ledger merge conflict at frame {idx}: "
                    f"{self.entries[idx][:12]} != {h[:12]}")
            self.entries[idx] = h

    # ─── persistence ──────────────────────────────────────

    def save(self, path: str) -> str:
        payload = {
            "version": 1,
            "fps": self.fps,
            "size": list(self.size),
            "rollup": self.rollup,
            "frames": {str(k): v for k, v in sorted(self.entries.items())},
        }
        tmp = path + ".part"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)
        os.replace(tmp, path)
        return path

    @staticmethod
    def load(path: str) -> "FrameLedger":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        led = FrameLedger(fps=d["fps"], size=tuple(d["size"]))
        led.entries = {int(k): v for k, v in d["frames"].items()}
        return led

    # ─── verification ─────────────────────────────────────

    def diff(self, other: "FrameLedger") -> List[int]:
        """Frame indices whose hashes differ (or exist on one side only)."""
        keys = set(self.entries) | set(other.entries)
        return sorted(k for k in keys
                      if self.entries.get(k) != other.entries.get(k))

    def verify_against(self, blessed_path: str) -> Optional[List[int]]:
        """None = identical. Otherwise the list of divergent frames."""
        if not os.path.exists(blessed_path):
            return None if not self.entries else list(self.entries)
        blessed = FrameLedger.load(blessed_path)
        d = self.diff(blessed)
        return d or None
