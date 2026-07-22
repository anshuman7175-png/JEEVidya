"""
JEEVidya V5 — Parallel Segment Rendering
════════════════════════════════════════
The payoff of making every segment a PURE FUNCTION of its content key:
they can render on every core simultaneously with bit-identical output.
A 12-segment video on an M-series laptop renders ~4× faster; overnight
batches scale with the machine.

Design:
  • ProcessPoolExecutor (spawn-safe: top-level worker fn, picklable
    payloads only — turn_data, DNA dict, seed keys).
  • Each worker process lazily builds ONE StreamingCompositor per
    (res_scale, dna, overrides) and reuses it across its segments —
    the heavy asset load is amortized, exactly like the serial path.
  • Workers write to distinct temp files; the parent puts results into
    the BuildCache. A worker crash costs one segment (Tier 0 resumes).

Set JV_RENDER_WORKERS=1 to force the serial path (debugging), or any N
to override the default (cpu_count − 2, capped at 4 for memory sanity).
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

# Per-process compositor cache: key → StreamingCompositor
_COMPOSITORS: Dict[str, Any] = {}


def default_workers(n_jobs: int) -> int:
    env = os.environ.get("JV_RENDER_WORKERS")
    if env and env.isdigit():
        return max(1, int(env))
    cpus = os.cpu_count() or 2
    return max(1, min(4, cpus - 2, n_jobs))


def _worker_render(payload: Dict[str, Any]) -> Tuple[str, str]:
    """Runs inside a worker process. Returns (seg_key, produced_path)."""
    import sys
    sys.path.insert(0, payload["project_root"])

    from config import settings
    from engine.visual_dna import VisualDNA
    from pipeline.compositor_v5 import StreamingCompositor
    from pipeline.encoder import StreamEncoder
    from pipeline.timeline import Timeline

    comp_key = json.dumps([payload["res_scale"], payload["dna"],
                           payload["overrides"]], sort_keys=True)
    comp = _COMPOSITORS.get(comp_key)
    if comp is None:
        dna = VisualDNA.from_dict(payload["dna"]) if payload["dna"] else None
        comp = StreamingCompositor(res_scale=payload["res_scale"], dna=dna,
                                   overrides=payload["overrides"])
        _COMPOSITORS.clear()            # one compositor per process
        _COMPOSITORS[comp_key] = comp

    timeline = Timeline(payload["turn_data"], fps=settings.FPS)
    out = os.path.join(settings.TEMP_DIR,
                       f"seg_{payload['index']:03d}_w{os.getpid()}.mp4")
    with StreamEncoder(out, comp.width, comp.height, settings.FPS) as enc:
        comp.render_segment(timeline, payload["index"], enc,
                            payload["seed_key"],
                            prev_shot=payload["prev_shot"],
                            max_frames=payload["cap"])
    return payload["seg_key"], out


def render_segments_parallel(
        jobs: List[Dict[str, Any]], cache,
        on_done: Optional[Callable[[int, int, str], None]] = None,
        workers: Optional[int] = None) -> int:
    """Render all `jobs` (payload dicts incl. 'seg_key') across processes
    and store results in the BuildCache. Returns segments rendered.

    Falls back to in-process rendering when workers==1 or the pool
    cannot start (some sandboxes) — same outputs either way."""
    if not jobs:
        return 0
    workers = workers or default_workers(len(jobs))

    if workers <= 1 or len(jobs) == 1:
        for i, payload in enumerate(jobs):
            seg_key, path = _worker_render(payload)
            cache.put(seg_key, "mp4", path)
            if on_done:
                on_done(i + 1, len(jobs), payload["name"])
        return len(jobs)

    done = 0
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker_render, p): p for p in jobs}
            for fut in as_completed(futures):
                seg_key, path = fut.result()      # crash → propagate (Tier 0 resumes)
                cache.put(seg_key, "mp4", path)
                done += 1
                if on_done:
                    on_done(done, len(jobs), futures[fut]["name"])
    except (OSError, RuntimeError) as e:
        # Pool unavailable (sandbox/fork limits): finish serially
        print(f"  [Parallel] pool unavailable ({e}); continuing serial")
        remaining = [p for p in jobs
                     if cache.get(p["seg_key"], "mp4") is None]
        for p in remaining:
            seg_key, path = _worker_render(p)
            cache.put(seg_key, "mp4", path)
            done += 1
            if on_done:
                on_done(done, len(jobs), p["name"])
    return done
