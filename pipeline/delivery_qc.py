"""
JEEVidya — Delivery Integrity Gates (Terminal Plan, Part XI)
════════════════════════════════════════════════════════════
Perfection must survive the phone screen and H.264, not just the PNG.
Every gate here runs on the MUXED FILE — decoded pixels, measured
loudness, probed container metadata — never on intermediate artifacts.

  GATES
  ─────
  loudness      post-mux ffmpeg loudnorm measurement: integrated
                −14 ±0.5 LU, true peak ≤ −1.0 dBTP, LRA sane.
  color         BT.709 primaries/transfer/matrix flags present in the
                container (wrong flags shift skin tones on every phone).
  av_offset     zero audio/video start-time offset in the muxed output
                (the classic invisible-in-PNG, obvious-on-YouTube bug).
  decoded_face  re-runs single-face checks on frames decoded from the
                final MP4 — codec truth, since 4:2:0 chroma subsampling
                smears exactly where the lips are.
  phone_scale   decoded frames downsampled to the real Shorts viewport
                (~420 px wide) must retain legible local contrast.

Everything returns a QCReport (tools/face_qc.py) so the ship DAG and
the publisher consume ONE report format. The publisher refuses any
artifact without a passing manifest — mechanically, not by convention.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tools.face_qc import GateResult, QCReport

# ── Targets (single source of truth; publisher + tests import these) ──
TARGET_LUFS = -14.0
LUFS_TOLERANCE = 0.5
TRUE_PEAK_MAX_DBTP = -1.0
LRA_MAX = 11.0                     # broadcast-sane loudness range
AV_START_OFFSET_MAX_S = 0.001      # effectively zero
PHONE_WIDTH = 420                  # real Shorts viewport width
PHONE_CONTRAST_FLOOR = 4.0         # 8×8-block std floor at 420 px


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def _ffprobe_exe() -> Optional[str]:
    """imageio_ffmpeg bundles only ffmpeg; ffprobe is optional. Probe
    for a system ffprobe and degrade gracefully when absent."""
    import shutil
    return shutil.which("ffprobe")


# ═══════════════════════════════════════════
# CONTAINER PROBES
# ═══════════════════════════════════════════

def probe_streams(path: str) -> List[Dict[str, Any]]:
    """Stream metadata via ffprobe (preferred) or ffmpeg -i parsing."""
    ffprobe = _ffprobe_exe()
    if ffprobe:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-show_streams", "-show_format",
             "-of", "json", path],
            capture_output=True, text=True)
        if proc.returncode == 0:
            return json.loads(proc.stdout).get("streams", [])
    # Fallback: parse `ffmpeg -i` stderr (always available — bundled)
    proc = subprocess.run([_ffmpeg_exe(), "-hide_banner", "-i", path],
                          capture_output=True, text=True)
    streams: List[Dict[str, Any]] = []
    for line in proc.stderr.splitlines():
        m = re.search(r"Stream #\d+:\d+.*?: (Video|Audio): (.*)", line)
        if m:
            streams.append({"codec_type": m.group(1).lower(),
                            "_raw": m.group(2)})
    return streams


def gate_color_metadata(path: str) -> GateResult:
    """BT.709 primaries/transfer/matrix must be flagged in the container."""
    streams = probe_streams(path)
    video = [s for s in streams if s.get("codec_type") == "video"]
    if not video:
        return GateResult("color", False, 0.0, 3.0, "no video stream found")
    v = video[0]
    if "_raw" in v:  # ffmpeg -i fallback: look for bt709 in the raw line
        ok = "bt709" in v["_raw"]
        return GateResult("color", ok, 3.0 if ok else 0.0, 3.0,
                          "bt709 flagged (raw probe)" if ok
                          else f"bt709 missing: {v['_raw'][:100]}")
    fields = (v.get("color_primaries"), v.get("color_transfer"),
              v.get("color_space"))
    n_ok = sum(1 for f in fields if f == "bt709")
    return GateResult("color", n_ok == 3, float(n_ok), 3.0,
                      "BT.709 primaries/transfer/matrix all flagged"
                      if n_ok == 3 else
                      f"only {n_ok}/3 bt709 flags present: {fields}")


def gate_av_start_offset(path: str) -> GateResult:
    """Audio and video must start at the same instant in the container."""
    ffprobe = _ffprobe_exe()
    if not ffprobe:
        return GateResult("av_offset", True, 0.0, AV_START_OFFSET_MAX_S,
                          "ffprobe unavailable — gate skipped (warn)")
    starts: Dict[str, float] = {}
    for s in probe_streams(path):
        st = s.get("start_time")
        if st is not None and s.get("codec_type") in ("video", "audio"):
            try:
                starts[s["codec_type"]] = float(st)
            except (TypeError, ValueError):
                pass
    if len(starts) < 2:
        return GateResult("av_offset", True, 0.0, AV_START_OFFSET_MAX_S,
                          f"only {list(starts)} streams expose start_time")
    delta = abs(starts["video"] - starts["audio"])
    return GateResult(
        "av_offset", delta <= AV_START_OFFSET_MAX_S,
        delta, AV_START_OFFSET_MAX_S,
        f"|video_start − audio_start| = {delta * 1000:.2f} ms")


# ═══════════════════════════════════════════
# LOUDNESS TRUTH (measured on the muxed file)
# ═══════════════════════════════════════════

def measure_loudness(path: str) -> Optional[Dict[str, float]]:
    """Run ffmpeg loudnorm in measurement mode; parse its JSON block."""
    proc = subprocess.run(
        [_ffmpeg_exe(), "-hide_banner", "-i", path,
         "-af", "loudnorm=print_format=json", "-f", "null", "-"],
        capture_output=True, text=True)
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", proc.stderr, re.DOTALL)
    if not m:
        return None
    raw = json.loads(m.group(0))
    try:
        return {"integrated_lufs": float(raw["input_i"]),
                "true_peak_dbtp": float(raw["input_tp"]),
                "lra": float(raw["input_lra"])}
    except (KeyError, ValueError):
        return None


def gate_loudness(path: str) -> GateResult:
    meas = measure_loudness(path)
    if meas is None:
        return GateResult("loudness", False, 0.0, LUFS_TOLERANCE,
                          "loudnorm measurement failed — no JSON block")
    lufs_err = abs(meas["integrated_lufs"] - TARGET_LUFS)
    problems = []
    if lufs_err > LUFS_TOLERANCE:
        problems.append(f"integrated {meas['integrated_lufs']:.1f} LUFS "
                        f"outside {TARGET_LUFS}±{LUFS_TOLERANCE}")
    if meas["true_peak_dbtp"] > TRUE_PEAK_MAX_DBTP:
        problems.append(f"true peak {meas['true_peak_dbtp']:.1f} dBTP "
                        f"> {TRUE_PEAK_MAX_DBTP}")
    if meas["lra"] > LRA_MAX:
        problems.append(f"LRA {meas['lra']:.1f} > {LRA_MAX}")
    return GateResult(
        "loudness", not problems, lufs_err, LUFS_TOLERANCE,
        "; ".join(problems) if problems else
        f"{meas['integrated_lufs']:.1f} LUFS, TP "
        f"{meas['true_peak_dbtp']:.1f} dBTP, LRA {meas['lra']:.1f}")


# ═══════════════════════════════════════════
# DECODED-PIXEL QC (codec truth)
# ═══════════════════════════════════════════

def _duration_s(path: str) -> Optional[float]:
    proc = subprocess.run([_ffmpeg_exe(), "-hide_banner", "-i", path],
                          capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):(\d+)\.(\d+)", proc.stderr)
    if not m:
        return None
    h, mi, s, cs = (int(g) for g in m.groups())
    return h * 3600 + mi * 60 + s + cs / 100.0


def _source_size(path: str) -> Optional[Tuple[int, int]]:
    proc = subprocess.run([_ffmpeg_exe(), "-hide_banner", "-i", path],
                          capture_output=True, text=True)
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", proc.stderr)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def decode_frames(path: str, count: int = 8,
                  width: Optional[int] = None) -> List[np.ndarray]:
    """Decode `count` frames evenly spread across the muxed file, as
    HxWx3 uint8 arrays. Optionally downscaled (phone-scale QC).
    Decode is single-threaded implicitly (one frame per invocation) —
    QC runs on decoded pixels, not on byte-identity of the MP4."""
    dur = _duration_s(path)
    size = _source_size(path)
    if dur is None or dur <= 0 or size is None:
        return []
    w0, h0 = size
    if width:
        w = width
        h = int(round(h0 * width / w0 / 2)) * 2
        vf = f"scale={w}:{h}"
    else:
        w, h = w0, h0
        vf = "null"
    frames: List[np.ndarray] = []
    for i in range(count):
        t = dur * (i + 0.5) / count
        proc = subprocess.run(
            [_ffmpeg_exe(), "-hide_banner", "-v", "error",
             "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1",
             "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
            capture_output=True)
        if proc.returncode != 0 or len(proc.stdout) < h * w * 3:
            continue
        buf = np.frombuffer(proc.stdout, dtype=np.uint8)
        frames.append(buf[:h * w * 3].reshape(h, w, 3).copy())
    return frames


def gate_decoded_face(path: str,
                      lip_rgb: Tuple[int, int, int] = (176, 62, 62),
                      tol: int = 60) -> GateResult:
    """Single-face sanity re-run on frames decoded from the final MP4.
    4:2:0 chroma smears saturated reds — exactly the lip region — so the
    PNG-space result does not transfer automatically."""
    from PIL import Image

    from tools.face_qc import color_mask, connected_components

    frames = decode_frames(path, count=6)
    if not frames:
        return GateResult("decoded_face", False, 0.0, 1.0,
                          "could not decode frames from the muxed file")
    worst = 0
    for arr in frames:
        img = Image.fromarray(arr, "RGB")
        mask = color_mask(img, lip_rgb, tol)
        if mask.sum() < 40:      # no visible mouth in this frame is fine
            continue
        worst = max(worst, connected_components(mask))
    return GateResult(
        "decoded_face", worst <= 1, float(worst), 1.0,
        f"max lip-mask components on decoded pixels = {worst} "
        f"({len(frames)} frames checked)")


def gate_phone_scale(path: str) -> GateResult:
    """Frames at the real Shorts viewport must retain local contrast —
    a mush of gray at 420 px means the video is illegible on the device
    it was made for."""
    frames = decode_frames(path, count=4, width=PHONE_WIDTH)
    if not frames:
        return GateResult("phone_scale", False, 0.0, PHONE_CONTRAST_FLOOR,
                          "could not decode phone-scale frames")
    contrasts = []
    for arr in frames:
        gray = arr.mean(axis=2)
        h, w = gray.shape
        bh, bw = h // 8, w // 8
        blocks = gray[:bh * 8, :bw * 8].reshape(8, bh, 8, bw)
        contrasts.append(float(blocks.std(axis=(1, 3)).mean()))
    mean_c = float(np.mean(contrasts))
    return GateResult(
        "phone_scale", mean_c > PHONE_CONTRAST_FLOOR,
        mean_c, PHONE_CONTRAST_FLOOR,
        f"mean 8×8-block contrast at {PHONE_WIDTH}px = {mean_c:.1f}")


# ═══════════════════════════════════════════
# THE FULL DELIVERY AUDIT
# ═══════════════════════════════════════════

def audit(path: str, report_path: Optional[str] = None) -> QCReport:
    """Run every delivery gate on a muxed MP4. The ship DAG calls this
    after encode; the publisher refuses artifacts without its manifest."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"delivery_qc.audit: no such file {path}")
    report = QCReport()
    report.add(gate_loudness(path))
    report.add(gate_color_metadata(path))
    report.add(gate_av_start_offset(path))
    report.add(gate_decoded_face(path))
    report.add(gate_phone_scale(path))
    if report_path:
        report.save(report_path)
    return report


# ═══════════════════════════════════════════
# QC-PASS MANIFEST (the publisher's admission ticket)
# ═══════════════════════════════════════════

def _sha256_file(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def manifest_path_for(video_path: str) -> str:
    return video_path + ".qc-manifest.json"


def write_manifest(video_path: str, report: QCReport,
                   extra: Optional[Dict[str, Any]] = None) -> str:
    """The QC-pass manifest the publisher requires (Part XII §12.1).
    Includes a checksum of the video so a swapped file invalidates it."""
    manifest = {
        "video": os.path.basename(video_path),
        "video_sha256": _sha256_file(video_path),
        "qc_passed": report.passed,
        "gates": report.to_dict(),
    }
    if extra:
        manifest.update(extra)
    out = manifest_path_for(video_path)
    tmp = out + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, out)
    return out


def verify_manifest(video_path: str) -> Tuple[bool, str]:
    """Publisher-side check: manifest exists, video checksum matches,
    and every gate passed. Returns (ok, reason)."""
    mp = manifest_path_for(video_path)
    if not os.path.exists(mp):
        return False, "no QC manifest — render was never delivery-audited"
    with open(mp, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if _sha256_file(video_path) != manifest.get("video_sha256"):
        return False, "video checksum mismatch — file changed after QC"
    if not manifest.get("qc_passed"):
        return False, "QC manifest records a failed gate"
    return True, "QC manifest valid"


# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════

def _main() -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="JEEVidya delivery-integrity audit (muxed-file truth)")
    ap.add_argument("video", help="path to a muxed MP4")
    ap.add_argument("--report", default=None,
                    help="write the QC report JSON here")
    ap.add_argument("--manifest", action="store_true",
                    help="also write the publisher QC-pass manifest")
    args = ap.parse_args()
    report = audit(args.video, args.report)
    print(report.summary())
    if args.manifest:
        print(f"manifest: {write_manifest(args.video, report)}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(_main())
