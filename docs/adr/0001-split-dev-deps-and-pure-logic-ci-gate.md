# ADR-0001: Split dev dependencies; gate every push on a pure-logic suite

- **Status:** accepted
- **Date:** 2026-08-12
- **Plan reference:** Phase −1 (code-health floor) and Phase 0.2 (the Harness)

## Context

The runtime pipeline needs a heavy render stack (torch, mediapipe,
opencv, pydub, …) that takes minutes to install and is irrelevant to
the phonology, timing, cache-key, and coalescing laws — the code where
regressions are cheapest to catch and most expensive to miss. A CI
gate that installs the render stack would either be skipped ("too
slow") or trusted less each time it flaked. Doctrine Law 1 says the
bug class to kill is *silently unexercised tests*, not any one missing
dependency.

## Decision

1. `requirements-dev.txt` contains only `pytest`, `hypothesis`, and
   `ruff`; it never grows a runtime dependency.
2. `tests/conftest.py` derives each test module's import closure and
   skips — loudly, in the terminal summary — modules whose render
   stack is absent. A skip can never be mistaken for a pass.
3. `.github/workflows/ci.yml` runs on every push/PR: `ruff check .`
   plus the pure-logic suite on Python 3.10 (repo floor) and 3.13,
   installing only `requirements-dev.txt`. Target: green in under a
   minute.

## Consequences

- Every push gets a sub-minute verdict on the pure-logic laws; there
  is no incentive to skip CI.
- Render-stack tests (golden frames, Tier-2 media) stay local-first
  until an artifact-versioned runner exists (Phase −1 DVC item);
  their absence from CI is *reported*, never hidden.
- Adding a runtime import to a pure-logic module demotes it out of
  the gate automatically — visible in the "not collected" summary,
  which is the intended alarm.

## Evidence

- `python -m pytest tests/ -q` with only dev deps: 64 passed in ~4 s,
  10 modules reported (not hidden) as needing the render stack.
- `ruff check .`: 0 findings at the floor configured in `ruff.toml`.
