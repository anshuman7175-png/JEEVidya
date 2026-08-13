"""
JEEVidya V5 — Pose Stager (Tier 1, one-time asset staging)
══════════════════════════════════════════════════════════
Turns the COMMITTED mouth art in assets/mouth_art/<character>/<letter>.png
(the single authoritative source, recorded in config/mouth_art_manifest.json)
into the exact directory layout the rest of Tier 1 consumes:

  assets/characters/<name>/
    body.png                 ← the matted neutral frame (rig source)
    poses/neutral.png        ← the ONLY pose (all 10 frames share one body)
    visemes_src/<VISEME>.png ← real-art mouth shapes for the Rig Builder

WHY EXACTLY ONE POSE: every frame in a character's set shares one
unchanged body — only the mouth articulation differs. Staging b–j as
gesture poses would make body swaps flap the mouth mid-sentence (a body
"transition" between two identical bodies with different mouths IS a
mouth flap). Letters b–j are therefore routed ONLY to visemes_src/,
per the manifest's viseme_mapping. Body variety needs new gesture
renders later.

MATTING (the root cause of the "flying mouth"): the committed art has
OPAQUE flat backgrounds (grey for chintu, white for gudiya). rig_v3
computes the character bbox from alpha > 40 — with an opaque frame
every pixel passes, the rig centres on the canvas, and every derived
coordinate (including the mouth plate) lands in the wrong place. This
module mattes the border-connected flat background to true alpha with a
flood fill seeded ONLY from the image border (so a white shirt or white
shoe soles that match the backdrop colour survive — they are not
border-connected), then feathers the cut edge by ~1 px.

HARD GATE: after matting, the alpha bounding box must span < 92% of the
canvas in both axes. A bbox that still fills the canvas means the matte
failed (background not flat, or character touches every edge) and the
rig would still mis-centre — the stage FAILS naming the exact file
rather than producing a rig that puts the mouth in the wrong place.

A labelled contact sheet of every matted frame is written to
output/mouth_art_contact_sheet_<character>.png so any letter can be
reassigned by eye (edit config/mouth_art_manifest.json → viseme_mapping).

Run:  python3 jvmake.py stage          (or: python3 tools/pose_stager.py)
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from engine.rig import VISEME_NAMES

MANIFEST_PATH = os.path.join(settings.PROJECT_ROOT, "config",
                             "mouth_art_manifest.json")

# Background-colour tolerance (Chebyshev distance to the border median).
# Flat studio backdrops are uniform to within a few counts; 14 absorbs
# JPEG-ish ringing without eating anti-aliased character edges.
BG_TOL = 14

# The alpha bbox must span less than this fraction of the canvas in BOTH
# axes, or the matte is declared failed for that file.
BBOX_MAX_FRAC = 0.92

# Feather width at the matte edge, px (~1 px: one 3×3 box-blur pass).
FEATHER_PX = 1


class StageError(RuntimeError):
    """A stage that cannot complete correctly fails loudly, naming the
    exact offending file (Law 1: no silent wrong output)."""


# ═══════════════════════════════════════════
# Manifest
# ═══════════════════════════════════════════

def load_manifest(path: str = MANIFEST_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════
# Border-seeded background matting (pure numpy)
# ═══════════════════════════════════════════

def _border_median_rgb(rgb: np.ndarray) -> np.ndarray:
    """Median colour of the 1-px border ring — the flat backdrop colour."""
    ring = np.concatenate([
        rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]], axis=0)
    return np.median(ring, axis=0)


def _fill_runs_rows(mask: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Extend `mask` to every full horizontal run of `close` pixels that
    already contains a mask pixel. Vectorized per row via run labels."""
    h, w = close.shape
    # run id: constant within maximal horizontal runs of `close`
    run_id = np.cumsum(~close, axis=1)
    out = mask.copy()
    for y in range(h):
        row_close = close[y]
        if not row_close.any():
            continue
        row_mask = mask[y] & row_close
        if not row_mask.any():
            continue
        ids = run_id[y]
        seeded = np.zeros(int(ids.max()) + 1, dtype=bool)
        seeded[ids[row_mask]] = True
        out[y] |= row_close & seeded[ids]
    return out


def _flood_from_border(close: np.ndarray, max_iter: int = 64) -> np.ndarray:
    """Boolean flood fill over `close`, seeded ONLY at the image border.
    Alternates full horizontal and vertical run-fills until stable —
    converges in a handful of passes on flat studio backdrops.
    Deterministic and dependency-free."""
    mask = np.zeros_like(close)
    mask[0, :] = close[0, :]
    mask[-1, :] = close[-1, :]
    mask[:, 0] |= close[:, 0]
    mask[:, -1] |= close[:, -1]
    for _ in range(max_iter):
        before = int(mask.sum())
        mask = _fill_runs_rows(mask, close)
        mask = _fill_runs_rows(mask.T, close.T).T
        if int(mask.sum()) == before:
            break
    return mask


def _box_blur_2d(a: np.ndarray) -> np.ndarray:
    """One 3×3 box-blur pass with edge replication (the ~1 px feather)."""
    p = np.pad(a, 1, mode="edge")
    return (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
            p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
            p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]) / 9.0


def matte_flat_background(img: Image.Image, src_name: str = "?",
                          tol: int = BG_TOL) -> Image.Image:
    """Convert a flat, border-connected opaque background to true alpha.

    Flood fill is seeded exclusively from the border, so interior
    regions that happen to match the backdrop colour (a white shirt on a
    white backdrop) are NEVER cut out — they are not border-connected.
    The resulting edge is feathered ~1 px so the silhouette composites
    without a hard jaggy fringe.
    """
    rgba = np.asarray(img.convert("RGBA")).copy()
    rgb = rgba[..., :3].astype(np.int16)

    bg = _border_median_rgb(rgb)
    close = (np.abs(rgb - bg).max(axis=-1) <= tol)
    background = _flood_from_border(close)

    if not background.any():
        raise StageError(
            f"matte failed for {src_name}: no border-connected flat "
            f"background found (border median {bg.astype(int).tolist()}, "
            f"tol {tol}). Is this frame already matted or non-flat?")

    alpha = np.where(background, 0.0, 1.0).astype(np.float32)
    for _ in range(FEATHER_PX):
        alpha = _box_blur_2d(alpha)
    # Keep the character core fully opaque; only the boundary feathers.
    alpha = np.where(background, alpha, np.maximum(alpha, 0.999))
    rgba[..., 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(rgba)


def alpha_bbox_fraction(img: Image.Image,
                        thresh: int = 8) -> Tuple[float, float]:
    """(bbox_w / w, bbox_h / h) of the alpha>thresh bounding box."""
    a = np.asarray(img.convert("RGBA"))[..., 3]
    ys, xs = np.nonzero(a > thresh)
    if len(xs) == 0:
        return 0.0, 0.0
    h, w = a.shape
    return ((int(xs.max()) - int(xs.min()) + 1) / w,
            (int(ys.max()) - int(ys.min()) + 1) / h)


def gate_matte(img: Image.Image, src_name: str,
               max_frac: float = BBOX_MAX_FRAC) -> None:
    """HARD GATE: the matted alpha bbox must be < max_frac of the canvas
    in both axes, or the file is named and the stage fails. An opaque or
    near-opaque frame is exactly what put the mouth in the wrong place."""
    fx, fy = alpha_bbox_fraction(img)
    if fx >= max_frac or fy >= max_frac:
        raise StageError(
            f"matte gate FAILED for {src_name}: alpha bbox spans "
            f"{fx * 100:.1f}% × {fy * 100:.1f}% of the canvas "
            f"(limit {max_frac * 100:.0f}%). The background was not "
            f"removed cleanly — the rig would mis-centre and the mouth "
            f"would land in the wrong place. Fix the source art or the "
            f"matting tolerance; do NOT rig this frame as-is.")
    if fx <= 0.0 or fy <= 0.0:
        raise StageError(
            f"matte gate FAILED for {src_name}: matting removed "
            f"EVERYTHING (empty alpha). The frame background is not "
            f"distinguishable from the character at tol {BG_TOL}.")


# ═══════════════════════════════════════════
# Contact sheet (reassign any letter by eye)
# ═══════════════════════════════════════════

def contact_sheet(frames: Dict[str, Image.Image],
                  labels: Dict[str, str],
                  thumb_w: int = 220) -> Image.Image:
    """Labelled grid of matted frames: '<letter> → <viseme>' under each."""
    letters = sorted(frames)
    cols = 5
    rows = (len(letters) + cols - 1) // cols
    label_h = 26
    thumb_h = max(
        int(round(frames[k].height * thumb_w / frames[k].width))
        for k in letters)
    cell_h = thumb_h + label_h
    sheet = Image.new("RGBA", (cols * thumb_w, rows * cell_h),
                      (34, 34, 40, 255))
    draw = ImageDraw.Draw(sheet)
    for i, letter in enumerate(letters):
        x = (i % cols) * thumb_w
        y = (i // cols) * cell_h
        img = frames[letter]
        th = int(round(img.height * thumb_w / img.width))
        thumb = img.resize((thumb_w, th), Image.LANCZOS)
        sheet.alpha_composite(thumb, (x, y))
        draw.rectangle([x, y + thumb_h, x + thumb_w, y + cell_h],
                       fill=(20, 20, 24, 255))
        draw.text((x + 6, y + thumb_h + 6),
                  f"{letter} → {labels.get(letter, '?')}",
                  fill=(240, 240, 240, 255))
    return sheet


# ═══════════════════════════════════════════
# Staging
# ═══════════════════════════════════════════

def _clean_pngs(d: str) -> None:
    """Remove previously staged sprites so mapping changes never leave
    stale files behind."""
    if not os.path.isdir(d):
        return
    for f in os.listdir(d):
        if f.lower().endswith(".png"):
            os.remove(os.path.join(d, f))


def stage_character(name: str, manifest: dict) -> bool:
    """Stage one character from the committed mouth art.

    • pose_source_letter ('a') → matted body.png AND poses/neutral.png —
      the ONLY pose (see module docstring: b–j are mouths, not gestures).
    • every letter with a real viseme class → visemes_src/<VISEME>.png.
    • labelled contact sheet → output/.
    """
    art_dir = os.path.join(settings.PROJECT_ROOT,
                           manifest.get("art_dir", "assets/mouth_art"), name)
    vmap: Dict[str, str] = manifest["viseme_mapping"][name]
    pose_letter: str = manifest.get("pose_source_letter", "a")

    char_dir = os.path.join(settings.CHARACTERS_DIR, name)
    poses_out = os.path.join(char_dir, "poses")
    visemes_out = os.path.join(char_dir, "visemes_src")
    os.makedirs(poses_out, exist_ok=True)
    os.makedirs(visemes_out, exist_ok=True)
    _clean_pngs(poses_out)
    _clean_pngs(visemes_out)

    missing = [letter for letter in sorted(vmap)
               if not os.path.exists(os.path.join(art_dir, f"{letter}.png"))]
    if missing:
        print(f"  [Stage] {name}: MISSING source art "
              f"{', '.join(os.path.join(art_dir, f'{m}.png') for m in missing)}"
              f" — run `python3 jvmake.py art` to verify/re-fetch")
        return False

    matted: Dict[str, Image.Image] = {}
    canvas_size: Optional[Tuple[int, int]] = None
    for letter in sorted(vmap):
        src = os.path.join(art_dir, f"{letter}.png")
        img = Image.open(src).convert("RGBA")
        out = matte_flat_background(img, src_name=src)
        gate_matte(out, src_name=src)
        if canvas_size is None:
            canvas_size = out.size
        elif out.size != canvas_size:
            # Same character set exported at a different resolution:
            # normalize so every viseme frame shares one canvas.
            out = out.resize(canvas_size, Image.LANCZOS)
        matted[letter] = out

    # 1 · body.png + poses/neutral.png ← the single shared body pose
    body = matted[pose_letter]
    body.save(os.path.join(char_dir, "body.png"))
    body.save(os.path.join(poses_out, "neutral.png"))
    print(f"  [Stage] {name}: body.png + poses/neutral.png ← "
          f"{pose_letter}.png (matted, bbox "
          f"{alpha_bbox_fraction(body)[0] * 100:.0f}%×"
          f"{alpha_bbox_fraction(body)[1] * 100:.0f}%)")

    # 2 · Viseme sources — NEVER staged as poses (they are mouths on one
    #     unchanged body; staging them as poses would flap the mouth on
    #     every body swap).
    staged_vis: List[str] = []
    for letter, viseme in sorted(vmap.items()):
        if viseme not in VISEME_NAMES:
            if viseme != "SPARE":
                print(f"  [Stage] {name}: letter '{letter}' maps to "
                      f"unknown viseme '{viseme}' — skipped "
                      f"(edit config/mouth_art_manifest.json)")
            continue
        matted[letter].save(os.path.join(visemes_out, f"{viseme}.png"))
        staged_vis.append(viseme)
    print(f"  [Stage] {name}: {len(staged_vis)} viseme sources "
          f"({', '.join(staged_vis)})")

    # 3 · Labelled contact sheet for by-eye reassignment
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    sheet_path = os.path.join(settings.OUTPUT_DIR,
                              f"mouth_art_contact_sheet_{name}.png")
    contact_sheet(matted, vmap).save(sheet_path)
    print(f"  [Stage] {name}: contact sheet → {sheet_path}")

    return bool(staged_vis)


def stage_all() -> bool:
    if not os.path.exists(MANIFEST_PATH):
        print(f"  [Stage] No mouth art manifest at {MANIFEST_PATH}")
        return False
    manifest = load_manifest()
    ok = True
    for name in sorted(manifest["viseme_mapping"]):
        try:
            ok = stage_character(name, manifest) and ok
        except StageError as e:
            print(f"  [Stage] {name}: FAILED — {e}")
            ok = False
    return ok


if __name__ == "__main__":
    sys.exit(0 if stage_all() else 1)
