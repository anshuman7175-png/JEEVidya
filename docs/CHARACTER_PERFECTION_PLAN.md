# JEEVidya Character Perfection Plan

Status: APPROVED — execution order is Phase 1 → 2 → 3 → 4 → 5.
Every phase ends with its machine-checkable QC gate passing for BOTH characters
(chintu, gudiya) before moving on. No eyeballing, no vacuous passes.

---

## Ground truth (what's measurably broken)

Evidence sources: baked asset inspection (`assets/characters/*/rig`), rendered
frame tests (rest / blink / gaze / visemes / head motion), and the repo's own
`jvmake.py verify-face` QC numbers.

| # | Symptom | Root cause (file) | QC evidence |
|---|---------|------------------|-------------|
| 1 | Head cut / hair severed, face wedge rotates alone | `head_mask()` in `tools/rig_v3.py` uses a convex hull of face landmarks — can never contain ponytails/hair spikes | Orphan hair on headless body; visible plate seams in yaw/nod/tilt renders |
| 2 | No real blink; eyes turn into flat skin discs | `lid_sprite()` in `tools/art_eyes.py` row-walk collapses on glasses frames / hairlines | `blink_closure`: 0.58 (chintu), 0.25 (gudiya) vs ≤ 0.02 required |
| 3 | Mouth in wrong place | Origin mismatch between `_bake_viseme_plates` fitting and `engine/head_assembly.py` compositing | `registration:mouth` fails all 10 visemes (2–12px vs 1.7px) |
| 4 | Mouth slides across face during motion | Mouth anchored to body blend, not head transform | `pose_mouth_lock`: 325px (chintu) vs 3.5px allowed |
| 5 | Lip-sync shows no shape change | Viseme plates composited at wrong scale — BILABIAL renders identical to REST | `discriminability` = 0.0000 |
| 6 | Characters frozen (no gestures) | Only `neutral` pose staged; poses k–p never mapped | `config/mouth_art_manifest.json`, `pipeline/pose_stager.py` |
| 7 | All of the above shipped silently | QC never enforced — committed `face_qc_report.json` "passed" with 0 gates run | `jvmake.py` never blocks on verify-face |

---

## Phase 1 — Head integrity (`tools/rig_v3.py`)

Goal: the head plate contains ALL head pixels (hair, ears, accessories); the
headless body contains none.

1. Replace convex-hull mask with a silhouette-connected mask:
   - Seed: current face-landmark hull (correct as a *seed*, wrong as a *boundary*).
   - Compute the neck seam line from jaw landmarks + shoulder detection
     (partially derived already for `neck` in `rig.json`).
   - Flood-fill on the character's alpha channel from the seed, constrained
     below by the neck seam. Every opaque pixel connected to the face above the
     seam joins the head mask — captures ponytails, spikes, buns, ears,
     glasses arms.
   - Feather only the neck seam edge (6–10px); keep silhouette edges hard.
2. Rebuild `head_plate.png`, `headless/*.png`, and `head_canonical.png` from
   the new mask; headless body gets content-aware fill only at the neck seam
   (not across former hair regions).
3. Enforce existing but unused gates: `border_opaque_counts` = 0 on the plate;
   add a new gate — zero opaque pixels above the neck seam on every headless pose.

Acceptance: motion sheet (yaw/nod/tilt/sway) shows zero orphan hair, no seams,
hair moves with head. Gates pass for both characters.

## Phase 2 — True blink (`tools/art_eyes.py`)

Goal: blink=1 fully covers the eye aperture, glasses untouched.

1. Aperture-driven lid construction: measure the eye aperture bbox from the
   iris/sclera segmentation (already computed for `eyeball_*.png`). Build the
   lid as: extracted skin strip above the lash line, vertically stretched to
   aperture height + 2px overshoot, back-filled with the socket gradient where
   the strip is thin.
2. Glasses-aware masking: detect frame ink (dark, thin, high-contrast ring
   around lens) and exclude it from the lid sprite so the lid slides *behind*
   the lens, never smearing the frame.
3. Lid motion curve in `engine/eye_model.py`: ease-in-out closure, upper lid
   does 80% of travel, 1-frame hold at closed.

Acceptance: `blink_closure` ≤ 0.02 for both characters; rendered half/full
blink shows a natural lash-edged lid; glasses frames pixel-identical to the
rest frame.

## Phase 3 — Mouth anchoring + viseme fidelity

Files: `tools/rig_v3.py` (`_bake_viseme_plates`), `engine/head_assembly.py`.

Goal: one coordinate system; visibly distinct visemes.

1. Single anchor source of truth: store the mouth anchor once in `rig.json`
   in *head-plate-local* coordinates. Bake viseme plates in that same local
   frame; delete the second (mismatched) origin computation in the compositor.
2. Lock mouth to head transform: in `head_assembly.py`, apply the *identical*
   affine used for the head plate to the viseme plate — the mouth becomes a
   child of the head, immune to body-pose blends.
3. True-scale visemes: fit each art viseme so its lip-corner span matches the
   measured rest-mouth span × the viseme's canonical width ratio
   (BILABIAL ≈ 0.85, OPEN_A ≈ 1.05, WIDE_E ≈ 1.25) instead of the current
   miniature uniform fit. Re-blend `mouth_shading.png` accordingly.
4. Jaw coupling: OPEN_A/MID_E drop the jaw region of the plate by
   `jaw × jaw_travel` px so the chin actually moves.

Acceptance: all `registration:mouth[*]` < 1.7px; `pose_mouth_lock` < 3.5px;
`discriminability` > 0.04; REST vs BILABIAL vs OPEN_A visibly distinct
side-by-side.

## Phase 4 — Living characters (poses + lip-sync timing)

1. Map gesture poses k–p in `config/mouth_art_manifest.json` and stage them in
   `pipeline/pose_stager.py` (same neck-seam bake as Phase 1 so all poses share
   one head anchor).
2. Enable the pose rotation already written in `pipeline/puppet.py` — verify
   weight-based selection during speech, neutral return at silence.
3. Lip-sync timing polish in `pipeline/lipsync.py`: coarticulation (visemes
   onset 40–60ms before their audio), collapse sub-50ms visemes into
   neighbors, guarantee BILABIAL closure on /p b m/ before the vowel opens.

Acceptance: a 15s rendered clip shows pose changes on emphasis, and the mouth
closes visibly on every "b/p/m" in the script.

## Phase 5 — Enforcement + previously confirmed fixes

1. Make QC blocking: `jvmake.py render`/`preview` runs `verify-face` first; a
   report with 0 gates executed is a hard failure (fixes the vacuous-pass bug).
   Rig bake refuses to write assets that fail Phase 1–3 gates.
2. Foley NaN clamp in the audio mixer; preview segment sampling spread across
   the full timeline; resolution-scaled caption font.
3. Final validation: full rebake of both characters → full `verify-face`
   gauntlet green → render the Escape Velocity preview → frame-by-frame visual
   pass on blinks, mouth shapes, and head motion.

---

## Order and dependencies

Phases must run 1 → 2 → 3 → 4 → 5: the head mask defines the coordinate frame
the lids and mouth anchor into, and poses (Phase 4) reuse the Phase 1 neck-seam
bake.

## QC gate summary (definition of done)

| Gate | Current (chintu / gudiya) | Required |
|------|---------------------------|----------|
| `blink_closure` | 0.58 / 0.25 | ≤ 0.02 |
| `registration:mouth[*]` | 2–12px, all 10 visemes fail | < 1.7px |
| `pose_mouth_lock` | 325px / drift | < 3.5px |
| `discriminability` | 0.0000 | > 0.04 |
| `border_opaque_counts` (plate) | unenforced | 0 |
| Orphan head pixels above neck seam (headless) | present | 0 |
| verify-face gates executed | 0 (vacuous pass) | all, blocking |
