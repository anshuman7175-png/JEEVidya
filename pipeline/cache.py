"""
JEEVidya V5 — Content-Addressed Build Cache
═══════════════════════════════════════════
The memory of the factory. Every expensive artifact (TTS audio, VTT,
rendered assets) is stored under sha256(inputs). Re-rendering a script
where one line changed re-does exactly one turn's work.

Lives at PROJECT_ROOT/.cache — deliberately OUTSIDE .tmp, which the
pipeline wipes on every run.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from typing import Optional

from config import settings

CACHE_ROOT = os.path.join(settings.PROJECT_ROOT, ".cache")


def key_of(*parts: object) -> str:
    """Stable content key from any number of parts."""
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x1f")  # unit separator: ("ab","c") != ("a","bc")
    return h.hexdigest()


class BuildCache:
    """Flat content-addressed store: <root>/<key[:2]>/<key>.<ext>"""

    def __init__(self, root: str = CACHE_ROOT):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, key: str, ext: str) -> str:
        shard = os.path.join(self.root, key[:2])
        return os.path.join(shard, f"{key}.{ext.lstrip('.')}")

    def get(self, key: str, ext: str) -> Optional[str]:
        """Path to a cached artifact, or None on miss."""
        p = self._path(key, ext)
        return p if os.path.exists(p) and os.path.getsize(p) > 0 else None

    def put(self, key: str, ext: str, src_path: str) -> Optional[str]:
        """Store a file under this key. Returns the cached path."""
        if not src_path or not os.path.exists(src_path):
            return None
        dst = self._path(key, ext)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".part"
        shutil.copy2(src_path, tmp)
        os.replace(tmp, dst)  # atomic — a crash never leaves half a file
        return dst

    def fetch(self, key: str, ext: str, dst_path: str) -> bool:
        """Copy a cached artifact to dst_path. Returns True on hit."""
        src = self.get(key, ext)
        if not src:
            return False
        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
        shutil.copy2(src, dst_path)
        return True

    # ─── Tiny text sidecars (metadata: durations, recipes) ──

    def put_text(self, key: str, ext: str, text: str) -> str:
        """Store a small text artifact (e.g. a duration) under this key."""
        dst = self._path(key, ext)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".part"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, dst)
        return dst

    def get_text(self, key: str, ext: str) -> Optional[str]:
        """Read a small text artifact, or None on miss."""
        p = self.get(key, ext)
        if not p:
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
        except OSError:
            return None

    def evict(self, key: str, ext: str) -> bool:
        """Remove one artifact (lets --force stay honest around the
        parallel prepass)."""
        p = self._path(key, ext)
        try:
            os.remove(p)
            return True
        except OSError:
            return False

    # ─── Maintenance ───────────────────────────────────────

    def stats(self) -> dict:
        files, size = 0, 0
        for dirpath, _, names in os.walk(self.root):
            for n in names:
                files += 1
                try:
                    size += os.path.getsize(os.path.join(dirpath, n))
                except OSError:
                    pass
        return {"files": files, "bytes": size, "root": self.root}

    def clear(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
        os.makedirs(self.root, exist_ok=True)
