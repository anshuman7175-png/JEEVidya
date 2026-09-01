"""
JEEVidya V5 — Streaming FFmpeg Encoder
══════════════════════════════════════
Pipes raw RGB frames straight into ffmpeg's stdin and muxes audio in the
same pass. Replaces the V2 path (PNG → disk → moviepy → second encode),
eliminating thousands of file writes and an entire re-encode.

Usage:
    with StreamEncoder(out_path, w, h, fps, audio_path=mix) as enc:
        for frame in frames:
            enc.write_frame(frame)   # PIL.Image or HxWx3 uint8 ndarray
"""
from __future__ import annotations

import os
import subprocess
from typing import Optional, Union

import numpy as np
from PIL import Image


def _ffmpeg_exe() -> str:
    """Resolve the bundled imageio-ffmpeg binary (no system install needed)."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


class StreamEncoder:
    """Single-pass rawvideo → H.264 encoder with audio mux."""

    def __init__(self, output_path: str, width: int, height: int, fps: int,
                 audio_path: Optional[str] = None,
                 crf: int = 19, preset: str = "faster"):
        self.output_path = output_path
        self.width = int(width)
        self.height = int(height)
        self.fps = fps
        self.frames_written = 0

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        self._stderr_log = output_path + ".ffmpeg.log"

        cmd = [
            _ffmpeg_exe(), "-y",
            # Video input: raw frames on stdin
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
        ]
        if audio_path and os.path.exists(audio_path):
            cmd += ["-i", audio_path]

        cmd += [
            # Subtle film grain: kills AI sterility + banding in flat areas
            "-vf", "noise=alls=4:allf=t+u",
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-profile:v", "high",
            # BT.709 color metadata (Terminal Plan §XI): wrong/missing flags
            # shift skin tones on every phone. Set AND asserted post-mux by
            # pipeline/delivery_qc.py — verified, never assumed.
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
            "-movflags", "+faststart",
        ]
        if audio_path and os.path.exists(audio_path):
            cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]

        cmd += [output_path]

        self._log_handle = open(self._stderr_log, "wb")
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._log_handle,
        )

    # ─── Frame ingestion ───────────────────────────────────

    def write_frame(self, frame: Union[Image.Image, np.ndarray]) -> None:
        """Write one frame. Accepts PIL Image (any mode) or HxWx3 uint8 array."""
        if isinstance(frame, Image.Image):
            if frame.mode != "RGB":
                frame = frame.convert("RGB")
            if frame.size != (self.width, self.height):
                frame = frame.resize((self.width, self.height), Image.Resampling.BILINEAR)
            data = frame.tobytes()
        else:
            arr = np.ascontiguousarray(frame, dtype=np.uint8)
            if arr.shape[0] != self.height or arr.shape[1] != self.width:
                raise ValueError(f"Frame shape {arr.shape} != ({self.height},{self.width},3)")
            data = arr.tobytes()

        try:
            self.proc.stdin.write(data)
        except BrokenPipeError as e:
            raise RuntimeError(
                f"ffmpeg died mid-encode. See log: {self._stderr_log}"
            ) from e
        self.frames_written += 1

    # ─── Lifecycle ─────────────────────────────────────────

    def close(self) -> None:
        if self.proc.stdin:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        code = self.proc.wait()
        self._log_handle.close()
        if code != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {code}. See log: {self._stderr_log}"
            )
        # Success: remove the noise log
        try:
            os.remove(self._stderr_log)
        except OSError:
            pass

    def __enter__(self) -> "StreamEncoder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            # Abort: kill ffmpeg, keep the log for forensics
            try:
                self.proc.kill()
            finally:
                self._log_handle.close()
            return
        self.close()


# ═══════════════════════════════════════════
# SEGMENT CONCAT + AUDIO MUX (jvmake final node)
# ═══════════════════════════════════════════

def concat_and_mux(segment_paths: list, audio_path: Optional[str],
                   output_path: str) -> str:
    """
    Losslessly concatenate per-turn video segments (ffmpeg concat demuxer,
    -c:v copy — zero re-encode, ~instant) and mux the mixed audio track in
    the same pass. All segments come from StreamEncoder with identical
    codec parameters, which is exactly what the concat demuxer requires.
    """
    if not segment_paths:
        raise ValueError("concat_and_mux: no segments to concatenate")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    list_path = output_path + ".concat.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for p in segment_paths:
            escaped = os.path.abspath(p).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    # HARD GUARANTEE (§XVII): a caller that passes an audio path expects
    # voice in the final video. If the mix file vanished (temp cleanup,
    # failed upstream node, haunted cache), fail LOUDLY here instead of
    # silently shipping a voiceless mp4 — that is the exact class of bug
    # that produced "the video has no voice".
    has_audio = audio_path is not None
    if has_audio and not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"concat_and_mux: audio mix expected at '{audio_path}' but the "
            "file does not exist — refusing to render a silent video. "
            "Re-run the voice/mix stage (the upstream node likely failed "
            "or its temp output was cleaned up).")
    if has_audio and os.path.getsize(audio_path) == 0:
        raise ValueError(
            f"concat_and_mux: audio mix at '{audio_path}' is empty (0 bytes) "
            "— refusing to render a silent video.")

    cmd = [_ffmpeg_exe(), "-y", "-f", "concat", "-safe", "0", "-i", list_path]
    if has_audio:
        cmd += ["-i", audio_path]
    cmd += ["-c:v", "copy"]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += [output_path]

    log_path = output_path + ".ffmpeg.log"
    with open(log_path, "wb") as log:
        code = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                              stderr=log).returncode
    if code != 0:
        raise RuntimeError(
            f"ffmpeg concat failed with code {code}. See log: {log_path}")

    for p in (list_path, log_path):
        try:
            os.remove(p)
        except OSError:
            pass
    return output_path
