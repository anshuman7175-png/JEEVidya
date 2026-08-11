"""
JEEVidya — Device & Determinism Resolver (Terminal Plan, Part II §2.1/2.3)
══════════════════════════════════════════════════════════════════════════
THE single authority for compute device selection and determinism.

Law 4 (Doctrine): "Determinism is a feature. Same inputs → bit-identical
frames and samples." This module is the mechanism that makes Law 4
mechanical instead of aspirational:

  • resolve_device()   — mps → cpu resolver. `.cuda()` NEVER appears in
                         this codebase; a CI grep-lint enforces it.
  • seed_everything()  — pins python, numpy, and (if present) torch RNGs
                         and enables deterministic algorithms.
  • rng_for(name)      — named, isolated np.random.Generator streams so
                         one subsystem consuming randomness can never
                         perturb another's sequence (thread-order and
                         call-order independence by construction).
  • determinism_selftest() — renders a probe twice, byte-compares.

Every torch call site imports THIS module. Nothing else touches devices.
"""
from __future__ import annotations

import hashlib
import os
import random
from typing import Callable, Dict, Optional

import numpy as np

GLOBAL_SEED = 60_1987  # pinned forever; changing it is a blessed event

# Named RNG streams. Each subsystem gets its own child generator derived
# from (GLOBAL_SEED, name) so streams are mutually independent AND
# individually reproducible regardless of creation order.
_streams: Dict[str, np.random.Generator] = {}

_TORCH: Optional[object] = None
_DEVICE: Optional[str] = None


def _try_torch():
    """Import torch lazily — the face pipeline must never break if the
    voice stack (torch) fails to build (Part II §2.1)."""
    global _TORCH
    if _TORCH is None:
        try:
            import torch  # noqa: PLC0415
            _TORCH = torch
        except Exception:
            _TORCH = False
    return _TORCH or None


def resolve_device() -> str:
    """mps → cpu. Never cuda. Sets the MPS fallback env var exactly once."""
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    torch = _try_torch()
    if torch is not None:
        try:
            if torch.backends.mps.is_available():
                _DEVICE = "mps"
                return _DEVICE
        except Exception:
            pass
    _DEVICE = "cpu"
    return _DEVICE


def seed_everything(seed: int = GLOBAL_SEED) -> None:
    """Pin every RNG the pipeline can reach. Idempotent."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch = _try_torch()
    if torch is not None:
        torch.manual_seed(seed)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    # PIL text rendering: force single-threaded rasterization where the
    # build supports it (thread pools are a nondeterminism source).
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def rng_for(name: str, seed: int = GLOBAL_SEED) -> np.random.Generator:
    """A named, isolated, reproducible random stream.

    Derivation: SeedSequence(seed, sha256(name)) — two different names can
    never collide, and the stream for a name is identical across runs and
    across process/thread scheduling.
    """
    key = f"{seed}:{name}"
    if key not in _streams:
        digest = hashlib.sha256(name.encode("utf-8")).digest()
        entropy = [seed] + list(digest[:8])
        _streams[key] = np.random.default_rng(np.random.SeedSequence(entropy))
    return _streams[key]


def reset_streams() -> None:
    """Drop all named streams (used by the determinism self-test to
    simulate a fresh process)."""
    _streams.clear()


def determinism_selftest(render_probe: Callable[[], bytes]) -> bool:
    """Run a probe render twice from identical state; byte-compare.

    `render_probe` must internally call reset_streams() + seed_everything()
    so both invocations start from the same RNG state. Returns True when
    the two byte streams are identical.
    """
    a = render_probe()
    b = render_probe()
    return hashlib.sha256(a).digest() == hashlib.sha256(b).digest()
