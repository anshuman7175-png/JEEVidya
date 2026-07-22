"""
JEEVidya V5 — jvmake DAG Core
═════════════════════════════
A tiny content-addressed build system (a mini-Bazel for video).

Rules of the game:
  • Every node's `key` is sha256 of ALL of its inputs — content, params,
    code fingerprint. Same key ⇒ same bytes ⇒ never built twice.
  • Artifacts live in the BuildCache (survives runs); .tmp is scratch.
  • A crash mid-build loses only the node that was executing — every
    finished node is already in the cache, so re-running the same graph
    resumes exactly where it died.
  • `force=True` rebuilds everything but still repopulates the cache.

The graph itself is dumb on purpose: correctness comes entirely from
honest keys. If a node's pixels can change, something in its key must
change too (see buildgraph.code_fingerprint / assets_fingerprint).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from pipeline.cache import BuildCache


@dataclass
class Node:
    """One build step: inputs are captured in `key`, output is one file."""
    name: str                                   # human label, e.g. "seg:03"
    key: str                                    # sha256 of every input
    ext: str                                    # artifact extension ("mp4"…)
    build: Optional[Callable[["Node"], str]]    # produces a file, returns path
    deps: List["Node"] = field(default_factory=list)

    # Runtime state (set by Graph.build)
    path: Optional[str] = None      # resolved artifact path (in cache)
    cached: Optional[bool] = None   # True=hit, False=built, None=untouched
    seconds: float = 0.0

    @property
    def short_key(self) -> str:
        return self.key[:8]


class Graph:
    """A content-addressed DAG with cache-backed incremental builds."""

    def __init__(self, cache: Optional[BuildCache] = None):
        self.cache = cache or BuildCache()
        self._nodes: dict = {}

    def node(self, name: str, key: str, ext: str,
             build: Callable[[Node], str],
             deps: Sequence[Node] = ()) -> Node:
        if name in self._nodes:
            raise ValueError(f"duplicate node name: {name}")
        n = Node(name=name, key=key, ext=ext, build=build, deps=list(deps))
        self._nodes[name] = n
        return n

    # ─── Traversal ─────────────────────────────────────────

    def _topo(self, target: Node) -> List[Node]:
        """Dependency-first order, with cycle detection."""
        order: List[Node] = []
        seen, visiting = set(), set()

        def visit(n: Node) -> None:
            if id(n) in seen:
                return
            if id(n) in visiting:
                raise ValueError(f"dependency cycle at node '{n.name}'")
            visiting.add(id(n))
            for d in n.deps:
                visit(d)
            visiting.discard(id(n))
            seen.add(id(n))
            order.append(n)

        visit(target)
        return order

    def status(self, target: Node) -> List[Tuple[Node, bool]]:
        """Dry-run: (node, is_cached) for every node, WITHOUT building."""
        return [(n, self.cache.get(n.key, n.ext) is not None)
                for n in self._topo(target)]

    # ─── Execution ─────────────────────────────────────────

    def build(self, target: Node, force: bool = False,
              on_progress: Optional[Callable[[int, int, Node], None]] = None
              ) -> str:
        """
        Bring `target` up to date. Returns the cached artifact path.
        Each node: cache hit → skip; miss → build, then store atomically.
        """
        order = self._topo(target)
        total = len(order)
        for i, n in enumerate(order):
            t0 = time.time()
            hit = None if force else self.cache.get(n.key, n.ext)
            if hit:
                n.path, n.cached = hit, True
            else:
                produced = n.build(n)
                stored = self.cache.put(n.key, n.ext, produced)
                if stored is None:
                    raise RuntimeError(
                        f"node '{n.name}' produced no artifact ({produced!r})")
                n.path, n.cached = stored, False
            n.seconds = time.time() - t0
            if on_progress:
                on_progress(i + 1, total, n)
        return target.path

    def summary(self, target: Node) -> str:
        """'12 cached · 3 built · 41.2s' — the jvmake scoreboard."""
        order = [n for n in self._topo(target) if n.cached is not None]
        hits = sum(1 for n in order if n.cached)
        built = len(order) - hits
        secs = sum(n.seconds for n in order)
        return f"{hits} cached · {built} built · {secs:.1f}s"
