"""
tools/gauntlet.py — the adversarial gauntlet (Terminal Plan Part XXI).

`pipeline/delivery_qc.py` audits a *sample* of frames for container and
per-frame correctness: colour metadata, loudness, A/V start offset, a
face being present, phone-scale legibility. It answers "is this frame
shippable?".

The gauntlet answers a strictly harder question: **is this frame
shippable given the frame before it?** Every artifact that survives
per-frame QC and still reads as fake is temporal:

  flicker        global luma/chroma stepping between adjacent frames —
                 the single most common "AI video" tell
  freeze         byte-identical consecutive frames: a dead compositor,
                 a stalled cache, or a dropped puppet update
  teleport       sub-pixel global motion measured by phase correlation;
                 a spike means the subject jumped rather than moved
  jitter         high-frequency oscillation of that same motion signal —
                 motion whose 3rd derivative eyes read as buzzing
  letterbox      an unintended uniform border (wrong scale/pad math)
  chroma_drift   slow hue migration across the video (grade instability)

All gates run on CONTIGUOUS decoded frames at the real frame rate,
sampled as several short bursts spread over the runtime, because a
temporal defect is invisible in the evenly-spaced single frames that
delivery QC decodes.

Output is a `QCReport` (tools/face_qc.py) — the one report format the
ship DAG and the publisher already consume.

CLI:
    python -m tools.gauntlet out/video.mp4 [--strict] [--report r.json]
    python -m tools.gauntlet out/video.mp4 --bursts 6 --burst-frames 40
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pipeline.delivery_qc import (_duration_s, _ffmpeg_exe, _source_size,
                                  probe_streams)
from tools.face_qc import GateResult, QCReport

# ── Thresholds. Derived from perceptual limits, never magic numbers. ──

# Weber contrast: below ~1% global luma step is invisible on a phone.
FLICKER_STEP_LIMIT = 0.012          # max |Δ mean luma| / mean luma
FLICKER_P99_LIMIT = 0.008           # 99th pct of the same signal
FREEZE_MAX_RUN = 2                  # identical frames in a row (>2 = stall)
FREEZE_EPS = 0.4                    # mean |Δ| in 8-bit codes → "identical"
TELEPORT_LIMIT_FRAC = 0.06          # global shift ≤ 6% of frame width / frame
JITTER_LIMIT_FRAC = 0.004           # per-frame motion 2nd difference
LETTERBOX_MAX_FRAC = 0.005          # uniform border ≤ 0.5% of a dimension
CHROMA_DRIFT_LIMIT = 6.0            # max drift of mean a*/b*-ish channels
BURSTS = 5
BURST_FRAMES = 30


# ═══════════════════════════════════════════
# CONTIGUOUS DECODE
# ═══════════════════════════════════════════

def decode_burst(path: str, start_s: float, n_frames: int,
                 width: Optional[int] = None) -> List[np.ndarray]:
    """Decode `n_frames` CONSECUTIVE frames starting at `start_s`.

    delivery_qc.decode_frames spreads its samples across the whole file
    (one ffmpeg invocation per frame) which destroys adjacency. Temporal
    gates need neighbours, so this is a single seek + sequential read.
    """
    size = _source_size(path)
    if size is None:
        return []
    w0, h0 = size
    if width and width < w0:
        w = int(width) // 2 * 2
        h = int(round(h0 * w / w0 / 2)) * 2
        vf = f"scale={w}:{h}:flags=bilinear"
    else:
        w, h = w0, h0
        vf = "null"
    proc = subprocess.run(
        [_ffmpeg_exe(), "-hide_banner", "-v", "error",
         "-ss", f"{max(0.0, start_s):.3f}", "-i", path,
         "-frames:v", str(int(n_frames)), "-vf", vf,
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        capture_output=True)
    stride = w * h * 3
    if proc.returncode != 0 or len(proc.stdout) < stride:
        return []
    buf = np.frombuffer(proc.stdout, dtype=np.uint8)
    count = len(buf) // stride
    return [buf[i * stride:(i + 1) * stride].reshape(h, w, 3)
            for i in range(count)]


def sample_bursts(path: str, bursts: int = BURSTS,
                  burst_frames: int = BURST_FRAMES,
                  width: int = 480) -> List[List[np.ndarray]]:
    """Several short contiguous bursts spread over the runtime. Cheap
    (≈150 small frames) yet adjacency-preserving."""
    dur = _duration_s(path)
    if not dur or dur <= 0:
        return []
    fps = probe_fps(path) or 30.0
    span = burst_frames / fps
    out: List[List[np.ndarray]] = []
    for i in range(max(1, bursts)):
        # Keep every burst fully inside the file, including the last one.
        t = (dur - span) * (i / max(1, bursts - 1)) if bursts > 1 else 0.0
        frames = decode_burst(path, max(0.0, t), burst_frames, width)
        if len(frames) >= 3:
            out.append(frames)
    return out


def probe_fps(path: str) -> Optional[float]:
    for s in probe_streams(path):
        if s.get("codec_type") == "video" or "r_frame_rate" in s:
            rate = s.get("avg_frame_rate") or s.get("r_frame_rate") or ""
            try:
                num, _, den = str(rate).partition("/")
                f = float(num) / float(den or 1)
                if 1.0 < f < 480.0:
                    return f
            except (TypeError, ValueError, ZeroDivisionError):
                continue
    return None


# ═══════════════════════════════════════════
# SIGNALS
# ═══════════════════════════════════════════

def _luma(frame: np.ndarray) -> np.ndarray:
    f = frame.astype(np.float32)
    return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]


def phase_shift(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Global translation a→b by phase correlation, in pixels.

    Robust to lighting change (magnitude is normalised away), which is
    exactly what separates "the subject moved" from "the grade shifted".
    Sub-pixel accuracy comes from a 3-point parabolic peak fit.
    """
    ha = np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None, :]
    fa = np.fft.rfft2((a - a.mean()) * ha)
    fb = np.fft.rfft2((b - b.mean()) * ha)
    cross = fa * np.conj(fb)
    mag = np.abs(cross)
    mag[mag < 1e-8] = 1e-8
    corr = np.fft.irfft2(cross / mag, s=a.shape)
    peak = np.unravel_index(int(np.argmax(corr)), corr.shape)

    def refine(axis: int) -> float:
        n = corr.shape[axis]
        i = peak[axis]
        idx = [peak[0], peak[1]]
        vals = []
        for d in (-1, 0, 1):
            idx[axis] = (i + d) % n
            vals.append(float(corr[idx[0], idx[1]]))
        denom = vals[0] - 2 * vals[1] + vals[2]
        frac = 0.0 if abs(denom) < 1e-12 else 0.5 * (vals[0] - vals[2]) / denom
        shift = i + max(-0.5, min(0.5, frac))
        return shift - n if shift > n / 2 else shift

    return refine(1), refine(0)          # (dx, dy) in pixels


def motion_track(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Per-frame global motion magnitude, in fractions of frame width."""
    lumas = [_luma(f) for f in frames]
    w = float(frames[0].shape[1])
    out = []
    for i in range(1, len(lumas)):
        dx, dy = phase_shift(lumas[i - 1], lumas[i])
        out.append(float(np.hypot(dx, dy)) / w)
    return np.asarray(out, dtype=np.float64)


def luma_steps(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Relative global luma change per frame (Weber-style)."""
    means = np.asarray([float(_luma(f).mean()) for f in frames])
    base = np.maximum(means[:-1], 1.0)
    return np.abs(np.diff(means)) / base


def freeze_runs(frames: Sequence[np.ndarray]) -> int:
    """Longest run of consecutive frames that are effectively identical."""
    longest = run = 1
    for i in range(1, len(frames)):
        d = float(np.mean(np.abs(frames[i].astype(np.int16)
                                 - frames[i - 1].astype(np.int16))))
        if d <= FREEZE_EPS:
            run += 1
            longest = max(longest, run)
        else:
            run = 1
    return longest


def uniform_border_frac(frame: np.ndarray) -> float:
    """Largest uniform (letterbox/pillarbox) border as a fraction of the
    corresponding dimension."""
    y = _luma(frame)
    h, w = y.shape

    def scan(lines) -> int:
        n = 0
        for line in lines:
            if float(line.std()) < 1.5:
                n += 1
            else:
                break
        return n

    top = scan(y[i] for i in range(h // 4))
    bot = scan(y[h - 1 - i] for i in range(h // 4))
    left = scan(y[:, i] for i in range(w // 4))
    right = scan(y[:, w - 1 - i] for i in range(w // 4))
    return max((top + bot) / h, (left + right) / w)


def chroma_drift(bursts: Sequence[Sequence[np.ndarray]]) -> float:
    """Spread of the mean opponent-colour channels across the whole
    video. A slow hue migration means the grade is not deterministic."""
    rg, by = [], []
    for burst in bursts:
        for f in burst:
            x = f.astype(np.float32)
            rg.append(float((x[..., 0] - x[..., 1]).mean()))
            by.append(float((x[..., 2] - 0.5 * (x[..., 0] + x[..., 1])).mean()))
    if len(rg) < 2:
        return 0.0
    return max(float(np.ptp(rg)), float(np.ptp(by)))


# ═══════════════════════════════════════════
# GATES
# ═══════════════════════════════════════════

def run(path: str, bursts: int = BURSTS, burst_frames: int = BURST_FRAMES
        ) -> QCReport:
    """The full temporal gauntlet over one muxed deliverable."""
    report = QCReport()
    if not os.path.exists(path):
        report.add(GateResult("gauntlet_input", False, 0.0, 1.0,
                              f"missing file: {path}"))
        return report

    windows = sample_bursts(path, bursts, burst_frames)
    if not windows:
        report.add(GateResult("gauntlet_decode", False, 0.0, 1.0,
                              "could not decode contiguous frames"))
        return report
    n = sum(len(w) for w in windows)

    steps = np.concatenate([luma_steps(w) for w in windows])
    worst_step = float(steps.max())
    p99 = float(np.percentile(steps, 99))
    report.add(GateResult(
        "flicker_max", worst_step <= FLICKER_STEP_LIMIT, worst_step,
        FLICKER_STEP_LIMIT,
        f"largest single-frame luma step over {n} frames"))
    report.add(GateResult(
        "flicker_p99", p99 <= FLICKER_P99_LIMIT, p99, FLICKER_P99_LIMIT,
        "sustained shimmer (99th percentile luma step)"))

    longest = max(freeze_runs(w) for w in windows)
    report.add(GateResult(
        "freeze_run", longest <= FREEZE_MAX_RUN, float(longest),
        float(FREEZE_MAX_RUN), "longest identical-frame run"))

    motion = np.concatenate([motion_track(w) for w in windows
                             if len(w) >= 2])
    worst_motion = float(motion.max()) if motion.size else 0.0
    report.add(GateResult(
        "teleport", worst_motion <= TELEPORT_LIMIT_FRAC, worst_motion,
        TELEPORT_LIMIT_FRAC, "max global shift per frame (frame widths)"))

    jitters = [float(np.abs(np.diff(motion_track(w), n=2)).max())
               for w in windows if len(w) >= 4]
    worst_jitter = max(jitters) if jitters else 0.0
    report.add(GateResult(
        "motion_jitter", worst_jitter <= JITTER_LIMIT_FRAC, worst_jitter,
        JITTER_LIMIT_FRAC, "2nd difference of global motion"))

    border = max(uniform_border_frac(w[len(w) // 2]) for w in windows)
    report.add(GateResult(
        "letterbox", border <= LETTERBOX_MAX_FRAC, border,
        LETTERBOX_MAX_FRAC, "largest uniform border"))

    drift = chroma_drift(windows)
    report.add(GateResult(
        "chroma_drift", drift <= CHROMA_DRIFT_LIMIT, drift,
        CHROMA_DRIFT_LIMIT, "opponent-colour spread across runtime"))
    return report


def _main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Adversarial temporal gauntlet (Part XXI)")
    ap.add_argument("video")
    ap.add_argument("--bursts", type=int, default=BURSTS)
    ap.add_argument("--burst-frames", type=int, default=BURST_FRAMES)
    ap.add_argument("--report")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero when any gate fails")
    args = ap.parse_args(argv)

    rep = run(args.video, args.bursts, args.burst_frames)
    print(rep.summary())
    if args.report:
        rep.save(args.report)
        print(f"  → {args.report}")
    return 1 if (args.strict and not rep.passed) else 0


if __name__ == "__main__":
    sys.exit(_main())


__all__ = ["run", "decode_burst", "sample_bursts", "phase_shift",
           "motion_track", "luma_steps", "freeze_runs",
           "uniform_border_frac", "chroma_drift", "probe_fps"]
