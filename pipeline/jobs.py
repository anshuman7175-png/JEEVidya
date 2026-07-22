"""
JEEVidya V5 — Render Job Queue (Tier 5)
═══════════════════════════════════════
Replaces app.py's single global _progress (which broke with 2 users):
a thread-backed queue with per-job ids, progress snapshots, results,
and cancellation flags. SSE-friendly: poll `job.snapshot()`.
"""
from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Job:
    id: str
    kind: str
    params: Dict[str, Any]
    status: str = "queued"          # queued | running | done | failed | cancelled
    stage: str = ""
    percent: float = 0.0
    message: str = ""
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    cancel_requested: bool = False

    def snapshot(self) -> Dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "status": self.status,
                "stage": self.stage, "percent": round(self.percent, 1),
                "message": self.message, "error": self.error,
                "result": self.result if isinstance(self.result, (str, int,
                                                                  float, dict,
                                                                  list)) else None,
                "queued_for": round(time.time() - self.created_at, 1)}


class JobQueue:
    """One render at a time (rendering saturates the CPU anyway), any
    number of queued jobs, all individually observable."""

    def __init__(self, workers: int = 1):
        self._jobs: Dict[str, Job] = {}
        self._q: "queue.Queue[Job]" = queue.Queue()
        self._lock = threading.Lock()
        self._handlers: Dict[str, Callable[[Job, Callable], Any]] = {}
        for _ in range(max(1, workers)):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()

    def register(self, kind: str,
                 handler: Callable[[Job, Callable], Any]) -> None:
        """handler(job, report) → result; report(stage, pct, msg)."""
        self._handlers[kind] = handler

    def submit(self, kind: str, **params) -> Job:
        if kind not in self._handlers:
            raise ValueError(f"no handler for job kind '{kind}'")
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, params=params)
        with self._lock:
            self._jobs[job.id] = job
        self._q.put(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job and job.status == "queued":
            job.cancel_requested = True
            job.status = "cancelled"
            return True
        if job and job.status == "running":
            job.cancel_requested = True     # honored at next progress tick
            return True
        return False

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(),
                          key=lambda j: j.created_at, reverse=True)
        return [j.snapshot() for j in jobs[:limit]]

    # ─── Worker loop ───────────────────────────────────────

    def _worker(self) -> None:
        while True:
            job = self._q.get()
            if job.status == "cancelled":
                continue
            job.status, job.started_at = "running", time.time()

            def report(stage: str, pct: float, msg: str,
                       _job: Job = job) -> None:
                _job.stage, _job.percent, _job.message = stage, pct, msg
                if _job.cancel_requested:
                    raise JobCancelled()

            try:
                job.result = self._handlers[job.kind](job, report)
                job.status = "done"
                job.percent = 100.0
            except JobCancelled:
                job.status = "cancelled"
            except Exception as e:              # noqa: BLE001 — jobs must not kill workers
                job.status, job.error = "failed", str(e)
            job.finished_at = time.time()


class JobCancelled(Exception):
    pass


# ─── Default queue with the render handler wired in ────────

_default: Optional[JobQueue] = None


def default_queue() -> JobQueue:
    global _default
    if _default is None:
        q = JobQueue(workers=1)

        def render_handler(job: Job, report):
            from generate import run_dialogue_pipeline
            return run_dialogue_pipeline(
                job.params["dialogue"],
                progress_callback=report,
                preview=bool(job.params.get("preview", False)))

        q.register("render", render_handler)
        _default = q
    return _default
