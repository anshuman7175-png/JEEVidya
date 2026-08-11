# JEEVidya — The Definitive Production Plan (Singularity Edition)

**Target:** a professional, million-subscriber-grade automated Shorts channel running on a MacBook M2 / 16 GB, 100% free & libre tooling, zero cloud dependency on the critical path.
**Scope:** this supersedes and absorbs the prior 9-phase face/voice plan. Everything in that plan survives; this document hardens it, upgrades its weakest links, and extends it from "the face is perfect" to "the *channel* is perfect" — because at millions of subs the pipeline IS the product.

**Structure:** Parts 0–XIV are the Terminal core — correctness by construction. Parts XV–XXII are the Singularity layer — the ceiling-breakers: a 2.5D depth head from flat art, a physics-based motion grammar, a unified affect nervous system, multimodal foley coherence, beat-level retention intelligence, a real-time performance engine, and adversarial QC that attacks the renderer instead of merely checking it. The Terminal core makes the face *correct*. The Singularity layer makes it *impossible to tell it was made by a machine* — and makes the machine learn faster than any human editor could.

This is the terminal plan. There should never be a v-next, because every decision below is either (a) correct by construction, (b) gated on rendered-pixel evidence, or (c) behind a swappable contract.

---

## Part 0 — Doctrine (the five laws)

Every decision during execution resolves against these, in order:

1. **Make bug classes unrepresentable.** No duplicated constants, no "remember to." If a mistake *can* recur, a mechanism (derivation, schema version, CI gate) must make it impossible. The mouth was on the eyes because two head-compose implementations coexisted — that class of bug dies by having exactly one implementation.
2. **Measure before opinion.** Nothing is "done" on vibes. Every fix has a number, a diff strip, or a test that moves. QC gates are the constitution; features are legislation.
3. **Free means free *and* sovereign.** Local, offline, open-license (Apache-2.0 / MIT / BSD) over free-tier cloud. A free tier is a paid tool with a delay. edge-tts is permitted only as a bridge with a named local successor (IndexTTS-2).
4. **Determinism is a feature.** Same inputs → bit-identical frames and samples. Every seed pinned, every thread-order dependency removed. Without this, the cache, the golden tests, and your own eyes are all lying.
5. **Evidence-gated investment.** Expensive work happens only after a gate whose pass/fail criteria were written *before* the evidence was gathered.

---

## Part I — Ground truth (confirmed by reading the code, not assumed)

### Face defects D1–D10 (from the prior plan, all re-verified)

| # | Defect | Root location |
|---|--------|---------------|
| D1 | Face boxes detected once on `body.png`, pasted onto every other pose | `tools/rig_builder.py:183` + `engine/pose_library.py:120` |
| D2 | Cross-fade composites two full bodies → two heads + one synthetic mouth = three mouths | `pose_library.blended_body` |
| D3 | Painted mouth never removed; hidden by feathered-ellipse backing + `_tone_match` | `bone_engine.py` |
| D4 | `render()` ignores `head_yaw/tilt/nod`; `_compose_head`/`_staged_head` are dead code | `bone_engine.py` (`face_dx = 0.0`) |
| D5 | Mouth blend = sprite-over-opaque-sprite, top-2 of 10 weights, quantized | `bone_engine.py:686-695` |
| D6 | Even-split grapheme timing, wrong G2P (`c→DENTAL`, `h→RETROFLEX`) | `engine/visemes.py:157` |
| D7 | `max(35ms)` events at 30 fps — sub-frame visemes never sampled | `engine/visemes.py` |
| D8 | No global A/V offset calibration | pipeline-wide |
| D9 | Blink = slid rectangle; iris coverage never guaranteed | `bone_engine.py` `_crop_lid_feathered` |
| D10 | Lids ignore gaze/brow; no crease, lash, lower lid | `pipeline/puppet.py` |

### 30 fps couplings (found in `config/settings.py`, must be de-hardcoded)

`AMPLITUDE_FRAME_MS = 33`, `SCENE_TRANSITION_FRAMES = 8`, `BODY_BREATHE_SPEED`, `BODY_SPEAK_SPEED`, `PARTICLE_SPEED_MIN/MAX`, `AMPLITUDE_SMOOTHING_FRAMES` — all per-frame quantities that run 2× fast at 60 fps. Every one becomes a per-second quantity scaled by `FPS` at use time; a lint (Part VI) bans new per-frame literals forever.

### Assets already in the codebase that this plan must exploit, not duplicate

- `engine/prosody.py` — F0 + RMS extraction, cached. **Currently underused for acting.**
- `engine/gestures.py` — 10 keyframed gestures fired on VTT word timestamps. **Timing source must migrate to the aligner.**
- `pipeline/mixdown.py` — broadcast voice chain, sound design, generative music bed, −14 LUFS target. **Output is asserted, never verified on the muxed file.**
- `agents/critic.py` — vision critic sampling frames. **Must consume the new QC metrics instead of heuristics.**
- `factory/flywheel.py` — Thompson-sampling bandit over DNA genes with YouTube stats. **The learning loop the channel grows on; must gain new arms.**
- `factory/publisher.py`, `factory/thumbnails.py`, `factory/localizer.py` — upload/bundle, best-frame thumbnails, localization. **All join the render DAG behind QC gates.**
- `tests/test_golden.py`, `tests/test_naturalism.py` — the seed of the regression harness.

---

## Part II — Phase 0: Environment, determinism, and the 60 fps foundation

**Target: MacBook M2, 16 GB, Apple Silicon. No CUDA anywhere.** Verified: `torchaudio.pipelines.MMS_FA` runs CPU-only on arm64; IndexTTS-2 and Chatterbox support `device="mps"` (with `PYTORCH_ENABLE_MPS_FALLBACK=1`).

### 2.1 Dependency spine

- `requirements.txt`: add `scipy>=1.11`, `librosa>=0.10`, `soundfile>=0.12`, `torch>=2.2`, `torchaudio>=2.2`, `uroman`. Voice-cloning deps isolated in `requirements-voice.txt` so the face pipeline never breaks if the TTS stack fails to build.
- `engine/device.py` (new): single resolver `mps → cpu`, sets the MPS fallback env var, pins `torch.manual_seed`, `np.random.default_rng` seeds, and forces deterministic algorithms where available. Every torch call site imports it. `.cuda()` never appears in this codebase.
- `jvmake doctor` hard checks: mediapipe, headless cv2, scipy, torchaudio, MMS_FA bundle reachable, MPS availability (warn-only), ffmpeg with libx264 + loudnorm filter, `rig.version == 3` for every character, and **a determinism self-test**: render 30 frames of the golden scene twice, byte-compare.

### 2.2 The 60 fps decision (confirmed necessary and cheap)

30 fps cannot represent a 15–25 ms plosive closure (one frame = 33 ms). `settings.FPS = 60` (16.7 ms frames) removes the last structural source of unnaturalness and is natively supported by YouTube Shorts.

- `AMPLITUDE_FRAME_MS` becomes `1000 / FPS`, derived, never literal.
- All per-frame constants (§Part I) converted to per-second units scaled by FPS at use time.
- `--fps` CLI flag (default 60): draft at 30, publish at 60. Every QC threshold expressed in frames is derived from FPS.
- Test asserts frame-count/duration relationships regenerated from FPS.

### 2.3 Determinism contract (Law 4, made mechanical)

- **Frame hash ledger:** every render emits `frames.sha256` (hash per frame + rollup). `jvmake render --verify-determinism` renders twice and diffs ledgers. Runs weekly in CI and before any release tag.
- **Sources of nondeterminism to hunt and pin:** PIL text rendering thread pools, numpy reduction order, `dict` iteration in scene assembly, wall-clock-seeded particles, ffmpeg threaded encode (`-threads` pinned for the QC decode pass; the *encode* may be threaded because QC runs on decoded pixels, not on byte-identity of the MP4).
- **Content-addressed cache** (`pipeline/cache.py` extended): every cache key is a hash of *all* inputs including code version of the producing module (module source hash), so a code change can never serve a stale artifact. Eviction: LRU with a configurable disk budget (default 40 GB), pinned entries for the golden corpus.

---

## Part III — Phase 1: Rig v3 — multi-pose registration & bakes

`RIG_VERSION = 3` in `engine/rig.py`. Loader keeps v1/v2 read support (so `/studio` and old bundles don't break) but `render()` requires v3; a v2 rig raises "run `jvmake rig --force`" instead of silently misplacing the face.

### 3.1 Landmark every pose

For `body.png` and every `assets/characters/<c>/poses/*.png` (17 per character), run `_detect_landmarks` at full resolution. Store all **478 landmarks**, not 5 boxes. Detection failure on any pose = hard error. No heuristic fallback in v3 — the heuristic is what put the mouth on the eyes historically.

### 3.2 Canonical → pose similarity transform

Umeyama least-squares similarity (scale s, rotation θ, translation t) on a **rigid landmark subset only** — nose bridge (6, 168, 197, 195, 5, 4), outer eye corners (33, 263), temples (127, 356), chin (152). Deliberately excludes lips/lids/brows, which differ by expression and would bias the fit.

- **Robustness:** IRLS with Huber weights, 3 reweight passes. Reject any pose with post-fit RMS > 0.8 px, fail loudly with the pose name. Store `{s, theta, tx, ty, rms}` per pose.
- θ is that pose's head roll — recovering it is why the head *sits* correctly instead of merely being translated.

### 3.3 Headless body bakes

Per pose: head mask = convex hull of face landmarks, extended to the alpha silhouette top (hair), dilated 0.02·face_h, feathered with a signed-distance ramp across the neck seam. Write `rig/headless/<pose>.png` and `rig/headmask/<pose>.png`. The feather band is **complementary** between head plate and headless body (α_head + α_body = 1 across the seam) — energy-preserving, no seam line.

### 3.4 Occluder bakes

For poses where a hand/arm crosses the head mask (reuse + fix the diff logic in `set_alt_torsos`, `bone_engine.py:369`): bake `rig/occluder/<pose>.png` — pixels inside the head mask that differ from canonical. Composited **after** the head, so a hand still passes in front of the face. Flag `occluded: true`.

### 3.5 Canonical head plate

From `body.png`: crop head by mask → `rig/head_canonical.png`. Inpaint mouth + both eyes out with `cv2.inpaint(INPAINT_NS)` seeded from dilated lip/lid contours → `rig/head_plate.png` — clean face with correct painted shading where the features were. **This kills D3 permanently.** Delete `_feathered_backing`, `_ring_clutter`, `_tone_match` from the bake.

Store in head-plate space: outer/inner lip rings, upper/lower lid polylines, iris center+radius per eye, brow polylines. Sample palette (`lip`, `lip_shadow`, `oral_cavity`, `teeth`, `tongue`, `sclera`, `iris`, `lash`, `skin`) plus a per-pixel mouth-region shading gradient map so the parametric mouth inherits the artwork's lighting.

### 3.6 Viseme targets fitted from the art

For each `visemes_src/<VISEME>.png` (`rig_builder.py:332`): landmark, register to canonical, least-squares fit the 5 mouth parameters (§Part IV) that best reproduce its lip contour. Art-derived targets, not hand-guessed. Missing shapes fall back to a built-in articulatory table. Store in `rig.mouth_targets`.

### 3.7 Rig schema v3

```json
{ "version": 3,
  "canonical_pose": "neutral",
  "head": { "landmarks": [[x,y]…478], "plate": "head_plate.png",
            "lip_outer": [...], "lip_inner": [...],
            "lid_upper_l": [...], "lid_lower_l": [...], "iris_l": [cx,cy,r],
            "brow_l": [...], "palette": {…}, "shading": "mouth_shading.png" },
  "mouth_targets": { "OPEN_A": {"jaw":…,"width":…,"round":…,"press":…,"pull":…}, … },
  "poses": { "<name>": { "landmarks": [...],
                         "xform": {"s":…,"theta":…,"tx":…,"ty":…,"rms":…},
                         "headless": "headless/<name>.png",
                         "headmask": "headmask/<name>.png",
                         "occluder": "occluder/<name>.png"|null,
                         "seam_y": …, "occluded": false } } }
```

---

## Part IV — Phase 2: Parametric mouth with true coarticulation

`engine/mouth_model.py` (new). This is the section that goes beyond the prior plan: naive weight-blending of viseme targets produces *average* mouths, not *coarticulated* mouths. Real speech anticipates: lips round for "u" *during* the preceding "s" ("suno"), and a bilabial closure resists being averaged away by its vowel neighbors.

### 4.1 Parameters (continuous, 5-D)

| Param | Meaning |
|-------|---------|
| `jaw` | inner aperture height, 0 = sealed |
| `width` | commissure separation |
| `round` | pucker / protrusion, aperture → circular |
| `press` | lip compression (bilabial thickening) |
| `pull` | commissure vertical pull (smile/frown) |

### 4.2 Cohen–Massaro dominance blending (upgrade over linear mixing)

Replace `p = Σ wᵢ · target(vᵢ)` with the dominance model:

- Each viseme class gets a **dominance function** per parameter: a raised-cosine envelope centered on the segment with class-specific width and peak. Bilabials (P/B/M) have near-total dominance on `jaw`/`press` (a closure cannot be diluted); vowels have wide, soft dominance on `round`/`width` (they spread into neighbors — that's anticipatory rounding).
- Per frame: `p_k = Σᵢ dᵢₖ(t)·targetᵢₖ / Σᵢ dᵢₖ(t)` per parameter k. Dominance envelopes come from a small articulatory table (10 classes × 5 params: peak, width-forward, width-backward), tunable but shipped with phonetics-literature defaults.
- Because dominance envelopes are C¹ and the parameter→contour map is C¹, the mouth is **provably continuous** — the popping bug class is eliminated by construction.
- `jaw` remains gated by the amplitude envelope and clamped to a max slew rate from real articulator velocity (≈12 cm/s scaled to face height): no physically impossible jumps, asserted in tests.

**Gate (Law 5):** implement linear blending first (1 day), then dominance (2 days). The QC "viseme discriminability at phone scale" metric (§Part VIII) decides whether dominance ships as default — the expectation, from phonetics, is a decisive win on closures.

### 4.3 Rasterization

- Deform canonical outer/inner lip closed cubic B-splines via FFD with control points at commissures, philtrum, cupid's bow, lower vermilion.
- **Hybrid texture:** FFD-warp the registered art sprite patch for the dominant viseme(s) into the deformed contour, blend by dominance — painted style, line weight, and shading preserved; geometry from the model, pixels from the art.
- Where art is absent, fill procedurally: oral cavity (dark, vertical gradient), teeth arc clipped to the upper contour when `jaw > 0.25`, tongue blob on DENTAL/RETROFLEX, lips with `lip`/`lip_shadow` gradient modulated by the baked shading map.
- Render at **4× supersample → LANCZOS down**. Contour evaluated in continuous coordinates, quantized only at the final downsample.
- Composite onto the head plate with **sub-pixel translation** (`Image.transform` AFFINE, fractional offset), never `int()`-rounded paste — integer rounding is itself a source of 1-px mouth jitter.
- Cache key: quantized 5-D parameter vector (jaw 1/64, others 1/32) in an LRU. Speech revisits similar shapes; hit rate stays high.

---

## Part V — Phase 3: Parametric eyes, lids, and living gaze

`engine/eye_model.py` (new). Rendered on the inpainted socket, in head space.

- **Eyeball:** sclera ellipse from the baked socket contour; iris + pupil disc at `(iris_c + saccade)`, clipped to the socket — saccades can't slide the iris onto the cheek.
- **Upper lid:** baked upper-lid polyline interpolated along an arc concentric with the eyeball, from open to fully-closed at the lower-lid contour. Skin fill + baked shading + lash line on the leading edge. At `blink = 1` the closed path IS the lower-lid contour — full iris occlusion is a geometric guarantee (D9 closed permanently), then asserted in QC.
- **Lower lid:** small rise on squint/smile, coupled to `pull`.
- **Coupling (D10):** lids track `eye_dy` at ≈0.6 gain; brow raise lifts the upper lid ≈0.15; blink pulls the brow down ≈0.08.
- **Blink dynamics:** replace the symmetric `sin^0.7` in `puppet.py:_blink_amount` with the measured human profile: close ~85 ms (cubic ease-in), hold ~25 ms, open ~180 ms (ease-out), ±8% jitter, independent 1-frame L/R offset. Keep the log-normal interval scheduler and 10% double-blink.
- **Living gaze (new):** micro-saccades (0.1–0.3° amplitude, 1–2 Hz Poisson) during holds so eyes never freeze; gaze targets driven by a simple saliency policy — look at the listener character on dialogue turns, glance to the reveal element 200–350 ms *before* pointing at it (anticipation is what reads as intelligence); pupil dilation +5–8% on emphasis words (from the prosody track). Cheap, procedural, and the difference between "puppet" and "person."
- 4× supersample, sub-pixel composite, same as the mouth.

---

## Part VI — Phase 4: Unified head assembly

`engine/bone_engine.py` — major rewrite. Delete the dead `_compose_head` / `_staged_head` / `_brow_patches` machinery and replace `render()` with ONE path (D4 root fix — one implementation cannot diverge):

```
1. body   = crossfade(headless[from], headless[to], eased_t)          # D2 fixed
2. head   = head_plate.copy()
             + brows(brow)                                            # polyline warp, not a patch
             + eyes_and_lids(blink, eye_dx, eye_dy, brow)             # Part V
             + mouth(param_vector)                                    # Part IV
3. M_pose = interp(xform[from], xform[to], eased_t)                   # slerp θ, lerp s/t → D1 fixed
   M_anim = roll(head_tilt + physics_overshoot)
             ∘ nod(head_nod) ∘ yaw(head_yaw) ∘ translate(sway, bounce)
   head   = head.transform(ONE composed affine, BICUBIC)              # single resample pass
4. body.alpha_composite(head, sub-pixel dest)
5. body.alpha_composite(occluder[from→to])                            # hands in front of face
6. squash & stretch (unchanged)
```

- `M_pose` uses the same eased `blend_t` as the body cross-fade — head travels in lockstep through every transition.
- Yaw parallax restored properly: features shift in head space before the transform (`face_dx` from yaw × head half-width) + horizontal cosine squash, replacing `face_dx = 0.0`.
- Exactly one affine resample for the whole head — no chained rotate→resize→transform, no cumulative softness.
- Hair shear retained, applied inside head space before the affine.
- Two-level LRU: face-channel key → head plate; (face key, quantized affine) → transformed head.

---

## Part VII — Phase 5: Phoneme-exact timing (the keystone)

`engine/align.py` (new). This phase fixes sync AND unlocks any voice (Part X), because timings derive from the waveform, never from TTS metadata.

### 7.1 Tier 1 — forced alignment (primary)

`torchaudio.pipelines.MMS_FA` CTC forced alignment against the actual TTS wav per turn. Hindi/Devanagari romanized with `uroman` first (as the MMS_FA recipe requires). Token-level spans; a **real phoneme inventory** (not the grapheme heuristic) maps to the 10 viseme classes — fixing D6's wrong shapes for ch, sh, th, aspirates, gemination, schwa deletion, and Devanagari conjuncts. Cached to `<turn>.align.json` keyed by audio hash: alignment runs once per turn, ever.

### 7.2 Tier 2 — waveform calibration (fallback + always-on correction)

- **Global A/V offset (D8):** cross-correlate the predicted jaw envelope against the audio onset envelope (librosa) over ±250 ms, apply argmax lag. Then DTW the two envelopes to correct slow drift within long turns.
- **Voiced/unvoiced + spectral-centroid gating:** snap bilabial closures to detected amplitude dips; forbid open-vowel visemes during silent frames.

### 7.3 Tier 3 — current even-split G2P, retained only as last resort, with Tier 2 offset still applied.

### 7.4 Sub-frame integration (D7)

`VisemeTrack.weights_at` sampled at 4 sub-frames per rendered frame, box-filtered — a 20 ms closure contributes ~60% weight instead of vanishing. Min-duration coalescing: events shorter than 1 sub-frame merge into the articulatorily dominant neighbor (bilabials win). Total viseme-weight mass conserved — asserted in tests.

### 7.5 Prosody-driven acting (new — this is where "correct" becomes "alive")

`engine/prosody.py` already extracts F0 + RMS. Currently it barely drives anything. Wire it into the acting layer:

- **Beat gestures on pitch accents:** detect F0 peaks (pitch accents); schedule micro head-nods and brow raises 50–100 ms *before* the accent (humans gesture ahead of stress). `engine/gestures.py` keyword triggers migrate from VTT timestamps to aligner timestamps — one timing source for everything.
- **Phrase-final behavior:** F0 declination at phrase end → slight head lower + settle; question rise → head tilt + brow raise.
- **Breath acting:** detect inter-phrase silences ≥ 400 ms → visible inhale (chest scale + slight head lift) *before* the next phrase, from the existing breathe channel. An agent that breathes before speaking reads as sentient.
- **Emphasis coupling:** RMS peaks scale gesture amplitude and pupil dilation. All gains conservative, all procedural, zero new art.

---

## Part VIII — Phase 6: The QC constitution (hard-fail gates on rendered pixels)

`tools/face_qc.py` + `jvmake verify-face`. Re-detects landmarks on rendered pixels and fails the render on violation. **All thresholds derived from FPS and face size, never literal frames/pixels.**

| Check | Assertion |
|-------|-----------|
| Registration | Mouth centroid + both iris centers within 0.6 px of the analytically predicted transformed position; head scale within 1%; roll within 0.3°. Swept over every pose × 21 cross-fade values of `blend_t`. |
| Single face | Exactly one face detected; connected components of the lip-color mask == 1; no lip-color pixels outside mouth contour + 2 px. Kills D2/D3 regressions. |
| Blink closure | At `blink = 1`: iris-color pixel count == 0; lid path covers 100% of the iris ellipse. |
| Temporal | Per-frame Δ of aperture, centroid, 5-D params below thresholds; no single-frame sign reversals (jerk metric); no 1-px centroid jitter during a held viseme. |
| A/V sync | Cross-correlation of rendered aperture vs. audio envelope peaks at |lag| ≤ 1 frame. |
| Sync confidence (new) | SyncNet-style sliding-window correlation between the mouth-region pixel sequence and the mel-spectrogram over the whole video — catches *local* desync (one bad turn) that a global correlation averages away. Pure numpy/librosa; no model download needed for the correlation variant. |
| Viseme discriminability (new) | Render each viseme class at phone scale (~420 px), compute pairwise contour distances; every class pair must exceed a separation threshold. This is the metric that gates the dominance model (§4.2) and the articulation gain (§Part XI). |
| Seam | Vertical gradient continuity across the neck band within tolerance; no alpha < 1 hole. |
| Rig sanity | Every pose RMS ≤ 0.8 px; version == 3. |

### Wiring

- `jvmake verify-face [--character X] [--strict]` → nonzero exit on any violation + an annotated diagnostic contact sheet (predicted vs. detected landmarks overlaid).
- Pre-flight in `jvmake setup/rig` (static sweep over poses × blends); post-render audit on sampled frames inside the render DAG. Failure fails the build.
- `agents/critic.py` upgraded: instead of its current heuristics, it consumes the QC metric JSON and renders a verdict + the diagnostic sheet. The critic becomes the QC's narrator, not a second opinion.

### The golden corpus (regression harness — what makes fixes permanent)

A fixed torture script, checked into `tests/golden_corpus/`: 12 lines covering every viseme class, every emotion tag, every pose transition, a hand-occluded pose, a question, and a long turn (drift test). Every commit that touches `engine/` or `pipeline/`:

1. Renders the corpus at 60 fps (cached; ~fast on M2 due to the LRU layers).
2. Runs the full QC table above.
3. Byte-compares frame hashes against the blessed ledger; any intentional visual change requires an explicit `jvmake bless` with a human-viewed diff strip.

`tests/test_face_registration.py`, `tests/test_lipsync_timing.py`, `tests/test_voice_identity.py` extend the existing `tests/test_naturalism.py` / `test_golden.py` style with the numeric invariants above.

---

## Part IX — Phase 7: Unify the unrigged path & delete dead code

- `engine/lipsync_pro.py` (`MouthBlender`, used at `compositor_v5.py:126` for unrigged characters) routes through the same `mouth_model` parameter vector — exactly one mouth implementation in the codebase.
- Retire `openness_track`/`shaped_openness`'s duplicate 5-class G2P in favor of the 10-class engine.
- Delete: `_crop_lid_feathered`, `_brow_patches` ellipse feathering, `_make_backing`, `_feathered_backing`, `_ring_clutter`, `_tone_match`, `_bake_visemes` (procedural ellipse mouths), `_bake_eyelids`, `blended_body`'s face-bearing variant, `_compose_head`, `_staged_head`.
- A CI grep-lint asserts these symbols never return.
- README updated with the v3 rig contract and the verify-face gate.

---

## Part X — Phase 8: Voice identity — your own voice, recorded once

`pipeline/voice_clone.py` + `pipeline/audio_source.py` (new).

### 10.1 Audio-source-agnostic contract

`voice.py` refactored behind one interface: `AudioSource.render(turn) -> { wav_path, sample_rate, duration }`. **Timings always come from `engine/align.py` on the returned wav — never from TTS metadata.** Three interchangeable backends, all lip-sync identically because Part VII derives timing from audio:

1. `EdgeTTSSource` — current behavior, kept as zero-setup default and fallback (a *bridge*, per Law 3).
2. `ClonedVoiceSource` — **your voice, synthesized per line. The recommended answer.**
3. `RecordedSource` — folder of hand-recorded wavs keyed by turn ID; the manual path, now merely an escape hatch.

### 10.2 Voice bank — record once, never per line

`assets/voices/<character>/`: `base.wav` (45–60 s neutral read) + `emotion/<name>.wav` (5–10 s each). `voice_bank.json` (schema + validator) maps character → clips. The script JSON's per-turn `emotion` field — which `voice.py` currently ignores for audio — becomes the automatic style selector. **Emotion per-line without recording per line.**

### 10.3 Engine choice (both licenses safe for monetized Shorts)

- **Primary: IndexTTS-2** (Apache-2.0) — zero-shot cloning with *disentangled* emotion reference: timbre from `base.wav`, emotion from the emotion clip. Runs on MPS.
- **Fallback: Chatterbox Multilingual** (MIT, Hindi supported, `chatterbox-mlx` Apple-Silicon-optimized) behind the same interface, config-selectable.
- **Excluded: XTTS-v2** — CPML license is non-commercial; unsafe for a monetized channel.
- Adapter isolates the engine; swapping models never touches the face pipeline.

### 10.4 Determinism & identity QC

- Fixed seed + pinned model revision + pinned reference clips ⇒ byte-identical audio for identical input. Asserted in tests.
- Cache keyed by `hash(text + character + emotion + engine + seed)` → each line synthesized once, ever. Re-renders free; M2 speed a non-issue.
- **Speaker-identity gate (hard):** cosine similarity of a speaker embedding between every generated line and `base.wav` must exceed threshold. Below → auto-retry with different seed, then fail loudly. A line can never drift off-voice.
- **Pronunciation gate (new):** run the MMS_FA aligner on the generated line against its own text; alignment confidence below threshold → the TTS mangled a word → retry with different seed, then flag for the recorded-wav escape hatch. Catches hallucinated syllables *before* they're published to a million people.
- Loudness normalized to integrated LUFS with true-peak limiting per line.
- `--voice-preview` renders the whole script's audio only, for a fast listen before committing to a video render.

### 10.5 Recording guide

`docs/VOICE_RECORDING.md`: phonetically balanced Hindi/Hinglish prompt sheet covering every viseme class, mic/level/room guidance, and an automatic input validator (clipping, noise floor, DC offset, sample rate) that rejects bad reference audio before it poisons every downstream line.

---

## Part XI — Phase 9: Delivery integrity — surviving the phone and the codec

Perfection must survive the phone screen and H.264, not just the PNG.

- **Phone-scale readability:** QC re-runs on frames downsampled to ~420 px wide (real Shorts viewport). Mouth aperture must remain discriminable between viseme classes at that scale — a calibrated **articulation gain** in shape space (never cartoon exaggeration), tuned until classes separate at phone scale while staying anatomically valid at 1080. Gated by the discriminability metric (§Part VIII).
- **Codec truth:** lips are the most saturated red region in frame — exactly where 4:2:0 chroma subsampling smears. QC decodes the final MP4 and re-runs registration + single-face + blink checks **on decoded pixels**. Encoder settings tuned for the face region (`-pix_fmt yuv420p`, high profile, tuned CRF, correct color metadata) and *verified*, not assumed.
- **Color metadata:** assert BT.709 primaries/transfer/matrix flags in the container (`ffprobe` check in the DAG) — wrong flags shift skin tones on every phone.
- **Loudness truth:** `pipeline/mixdown.py` targets −14 LUFS but never verifies the *muxed* file. Add a post-mux `ffmpeg loudnorm` (print_format json) measurement gate: integrated −14 ±0.5 LU, true peak ≤ −1.0 dBTP, LRA sane. Measured, not assumed.
- **A/V container check:** verify zero audio/video start-time offset in the muxed output (the classic invisible-in-PNG, obvious-on-YouTube bug).
- **Caption legibility gate:** captions rendered by `pipeline/caption_engine.py` re-checked at 420 px for WCAG-grade contrast against the sampled background region and minimum x-height; auto-bump stroke width if failing.

---

## Part XII — Phase 10: The channel factory (what "millions of subs" actually requires)

The face pipeline makes a video *watchable*; the factory makes a *channel*. All pieces exist in `factory/` — this phase wires them behind the QC constitution and gives the learning loop more to learn from.

### 12.1 The production DAG (extends `pipeline/buildgraph.py` / `pipeline/dag.py`)

```
script → voice (cached per line) → align (cached) → render (QC-gated)
       → encode → decoded-pixel QC → loudness QC → thumbnail forge
       → localize (optional) → publisher (upload or bundle) → flywheel ledger
```

- Every edge cached content-addressed; a script typo re-renders only affected turns.
- `jvmake ship <script.json>` runs the whole DAG; any QC failure stops the line with the diagnostic sheet. Nothing un-gated can reach YouTube. **The publisher refuses artifacts without a QC-pass manifest** — mechanically, not by convention.
- Batch mode (`factory/batch.py`) renders N scripts overnight on the M2; per-video wall-clock budget tracked and reported so throughput regressions are visible.

### 12.2 The learning flywheel (extends `factory/flywheel.py`)

The existing Thompson-sampling bandit learns over DNA genes (energy/hook/cut-rate). Extend the arm space with the new capabilities:

- **Thumbnail arms:** `factory/thumbnails.py` already scores frames; generate 2 candidate cards per video (frame choice × title treatment), publisher alternates, bandit learns CTR proxy.
- **Hook-style arms:** first-2-seconds treatment (cold open on reveal vs. question vs. character react) as a gene.
- **Voice-emotion intensity arm:** once Part X lands, emotion-reference intensity becomes tunable per video and learnable.
- Stats via the free YouTube Data API as today; the proxy (like-rate + comment-rate + view velocity vs. channel median) is retained. Week 1 it guesses; week 6 it knows.

### 12.3 Localization at scale (extends `factory/localizer.py`)

Because Part VII derives lip sync from audio, a re-dub **re-syncs automatically**: localize script → synthesize with the cloned voice (multilingual engines support it) → aligner produces new timings → same rig renders. One face pipeline, every language. Each localized output passes the identical QC constitution.

### 12.4 Operational runbook

- `docs/RUNBOOK.md`: failure taxonomy (rig failure / align failure / identity-gate failure / QC-gate failure / upload quota), each with the exact diagnostic artifact to open and the fix.
- Weekly automated jobs: determinism self-test, golden corpus render, cache-budget report, flywheel snapshot.
- Disk budget: content-addressed store capped (default 40 GB), golden corpus pinned, per-video artifacts pruned after publish + 14 days.

---

## Part XIII — Files

**New:** `engine/mouth_model.py`, `engine/eye_model.py`, `engine/head_transform.py`, `engine/align.py`, `engine/device.py`, `tools/face_qc.py`, `pipeline/voice_clone.py`, `pipeline/audio_source.py`, `assets/voices/voice_bank.json`, `requirements-voice.txt`, `docs/VOICE_RECORDING.md`, `docs/RUNBOOK.md`, `tests/golden_corpus/`, `tests/test_face_registration.py`, `tests/test_lipsync_timing.py`, `tests/test_voice_identity.py`

**Major rewrite:** `engine/bone_engine.py` (single head-assembly path), `tools/rig_builder.py` (multi-pose registration + inpaint bakes), `engine/rig.py` (v3 schema)

**Moderate:** `engine/pose_library.py` (headless bodies + occluders + xform interpolation), `engine/visemes.py` (phoneme inventory, sub-frame sampling, dominance blending, coalescing), `pipeline/puppet.py` (blink dynamics, 5-D params, lid coupling, micro-saccades), `engine/gestures.py` (aligner timestamps, prosody coupling), `engine/prosody.py` (pitch-accent detection), `jvmake.py` (verify-face, doctor, ship, bless), `pipeline/buildgraph.py` (QC-gated DAG), `factory/flywheel.py` (new arms), `factory/publisher.py` (QC-manifest requirement), `pipeline/mixdown.py` (post-mux loudness gate)

**Minor:** `pipeline/compositor_v5.py`, `pipeline/voice.py` (AudioSource refactor), `engine/lipsync_pro.py`, `pipeline/caption_engine.py` (legibility gate), `agents/critic.py` (consume QC metrics), `config/settings.py` (FPS 60 + de-hardcode), `requirements.txt`, `README.md`

---

## Part XIV — Order of work for the Terminal core (each step gated)

```
0. Env bootstrap + device.py + doctor + determinism self-test + FPS 60 de-hardcoding
1. Rig v3 registration & bakes            → registration QC alone proves D1/D2 fixed
2. Head assembly rewrite                  → verify-face static sweep passes
3. Mouth model (linear → dominance)       → discriminability gate decides default
4. Eye/lid model + living gaze            → blink-closure gate passes
5. Alignment (THE KEYSTONE)               → sync gate ≤1 frame; unlocks any voice
6. Full QC constitution + golden corpus   → all gates green, corpus blessed
7. Unify & delete dead code               → grep-lint green
8. Voice identity layer                   → identity + pronunciation gates green
9. Delivery integrity                     → decoded-pixel + loudness + color gates green
10. Factory wiring (ship DAG, flywheel arms, localization, runbook)
11. Tune against diagnostic sheets until every check passes WITH MARGIN
```

**Gate at each step:** `jvmake verify-face --strict` (and from step 6, the golden corpus) must pass before moving on.

**Sequencing note:** steps 0–7 fix the face and are fully useful with today's edge-tts voices. Step 8 swaps in your voice with no further face work, because step 5 made timing derive from audio. Record your reference clips any time before step 8. Step 10 turns a video machine into a channel machine — and by then every artifact that reaches YouTube has passed a constitution that no human needs to remember to enforce.

---

---

# THE SINGULARITY LAYER

Everything above makes the pipeline *correct*. Everything below makes it *transcendent*. Each Singularity part rides on the Terminal core's contracts (rig v3 landmarks, aligner timings, the QC constitution, the content-addressed DAG) — nothing here weakens a gate; every part adds new ones. Each is independently shippable and independently gated (Law 5), so the core never waits on the ceiling.

---

## Part XV — The 2.5D depth head: real parallax from flat art

The Terminal core shifts features for yaw (§Part VI). That reads as "good puppet." A million-sub channel needs "how is that flat art *turning*?" — Live2D-grade depth, procedurally, from the single painted head.

`engine/depth_head.py` (new):

- **Delaunay mesh over the head plate:** triangulate the 478 landmarks + hair/silhouette boundary points sampled from the alpha contour. The head becomes a warpable mesh, not a sprite.
- **Procedural depth proxy:** assign each vertex a depth from an anatomical prior fitted to the landmarks — nose tip nearest, ears/jawline farthest, forehead/cheeks interpolated on a face-topology heightfield. No neural depth model, no download: the MediaPipe canonical 3D face model *already ships z-coordinates* for all 478 points; use them, scaled to the art.
- **Yaw/pitch → per-vertex parallax:** vertex displacement = `depth(v) × sin(angle)` in head space, rendered as a piecewise-affine warp (scipy `griddata`/PiecewiseAffineTransform, cached per quantized angle). The nose leads the turn, the far cheek foreshortens, the near jawline swings wide — actual 2.5D rotation, ±12° range where flat art stays believable.
- **Occlusion-aware hair layers:** split hair alpha into front/back layers at the face contour during the rig bake; back hair parallaxes *less* than the face, front bangs parallax *more* — three-plane depth for free.
- **Dynamic shading:** a single baked normal-proxy map (from the depth heightfield) modulated by a fixed key-light direction → cheek/nose shading shifts subtly with yaw. One multiply, painted-style preserved, no relighting uncanny valley.
- **QC gates (new):** mesh fold-over detector (no triangle flips at any angle in the sweep); silhouette continuity (alpha boundary stays simple-connected); landmark re-detection on warped renders must recover the analytically predicted positions within 0.8 px across the full ±12° sweep.
- **Fallback contract:** `depth_head.enabled` flag; when off, Part VI's affine path renders identically to the Terminal core. The golden corpus is blessed separately per mode.

---

## Part XVI — The motion grammar: physics and the twelve principles, encoded

Human animators apply the Disney principles by instinct. Encode them as a procedural layer so every motion channel obeys them mechanically.

`engine/motion_grammar.py` (new) — a post-processor over ALL animation channels (head, brows, gaze, gestures, body) before rendering:

- **Spring-damper secondary motion:** hair strands and any loose costume element (baked as chains of 3–5 verlet points from the rig's hair layers) driven by head acceleration — the hair *follows through* and *overlaps* after a head turn, settles with critically-damped wobble. Deterministic (fixed timestep, seeded), cheap (dozens of points).
- **Anticipation injector:** any gesture or pose transition above a velocity threshold gets an automatic 2–4 frame counter-motion first (head dips before it rises, hand pulls back before it points). Amplitude proportional to the main motion, capped, C¹-blended.
- **Follow-through & settle:** every channel's keyframe target is reached via a slightly-underdamped second-order filter (tunable ζ ≈ 0.85) instead of pure easing — motions *arrive* with a 1–2% overshoot and settle, which is the difference between tweened and alive. The existing physics_overshoot on head roll generalizes to every channel.
- **Smear discipline:** on any frame where head angular velocity exceeds a threshold, apply directional motion blur to the head layer only (2–3 tap directional box in motion direction) — fast turns stop strobing at 60 fps.
- **OU-noise idle:** replace every sine-based idle (breathe wobble, sway) with seeded Ornstein–Uhlenbeck processes — mean-reverting noise that never visibly loops. Sines read as robotic within 10 seconds; OU never repeats.
- **Arc enforcement:** gaze and head targets travel along slight arcs (quadratic bow ∝ distance), never straight lines — straight-line interpolation is the single biggest "computer did this" tell.
- **QC gates (new):** jerk budget per channel (third derivative bounded); settle detector (no channel oscillates > 3 visible cycles); smear frames must never appear during a held pose; determinism ledger covers physics state.

---

## Part XVII — The affect engine: one nervous system, not per-channel triggers

Part VII.5 wires prosody to acting per-channel. The ceiling-breaker: a single continuous emotional state that ALL channels read from, so the whole body agrees about how the character feels — coherence is what audiences unconsciously read as "real."

`engine/affect.py` (new):

- **Valence–arousal state:** a 2-D continuous state per character, updated per frame by: the script's per-turn `emotion` tag (target), prosody (RMS/F0 push arousal), and dialogue events (being addressed, reveals). Follows targets through the §XVI second-order filter — emotions *transition*, never snap.
- **Channel mapping matrix:** one declarative table maps (valence, arousal) → resting mouth `pull`/`press` bias, brow height/inner-raise, lid openness, blink-rate multiplier, gaze-hold duration, head-tilt bias, gesture amplitude gain, breath rate/depth, and — once Part X lands — the voice-emotion reference intensity. Every channel reads the SAME state: an excited character blinks faster, breathes shallower, gestures bigger, holds gaze shorter, and *sounds* brighter, all from one number pair.
- **Listener reactions (multi-character scenes):** the non-speaking character's affect state tracks the speaker's with 300–500 ms lag and 0.4 gain → automatic reactive listening: nods on the speaker's pitch accents, brow raise on their reveals, gaze at the speaker with natural check-away saccades. Dead-eyed listeners are the #1 tell of automated dialogue; this kills it with zero new art.
- **Micro-expression grammar:** brief (120–200 ms) sub-threshold expression flickers on affect-state transitions (surprise onset before the smile), scheduled deterministically from the state trajectory.
- **QC gates (new):** affect-coherence audit — per-frame correlation between channel outputs and the state trajectory must exceed threshold (a smiling mouth with fear-brows fails the render); state continuity (bounded derivative, no snaps).

---

## Part XVIII — Multimodal coherence: the body you hear

Sound and image must agree at the millisecond level — audiences forgive a flat image long before they forgive incoherent audio.

Extends `pipeline/mixdown.py` + `engine/affect.py`:

- **Synthesized breath foley:** Part VII.5 schedules visible inhales; synthesize the matching breath *sound* (filtered noise burst shaped by breath depth from the affect state) at the exact aligned instant, −38 dB under dialogue. Seeing AND hearing the inhale is a subliminal "alive" signal almost no automated channel has.
- **Cloth/motion foley:** gesture onsets above an amplitude threshold trigger a soft cloth swish from a small seeded procedural synthesizer, panned to the character's screen position. Sub-audible consciously; missed when absent.
- **Room coherence:** all dialogue passes through one shared convolution with a subtle small-room impulse response (generated procedurally, seeded) so both characters exist in the SAME acoustic space — separately synthesized TTS lines otherwise sound like they were recorded on different planets, and everyone hears it without knowing why.
- **Ducking with lookahead:** music bed ducks 150 ms *before* dialogue onset (timings are known from the aligner — use them), recovers on phrase-final F0 declination. Ducking that anticipates reads as a human mix engineer.
- **Mouth de-click + emphasis micro-dynamics:** spectral de-click on TTS output; +0.5–1 dB micro-lift on aligner-identified emphasis words, matched to the pupil/gesture emphasis so audio and body stress the same syllables.
- **QC gates (new):** breath A/V co-occurrence (every visible inhale has its sound within ±1 frame); shared-reverb verification (cross-correlation of late-tail signatures between speakers); duck-timing assertion against aligner onsets.

---

## Part XIX — Retention intelligence: learning at the beat, not the video

The Terminal flywheel learns per-video. YouTube Analytics exposes per-moment audience retention — the flywheel must learn per-*beat*, which is 100× more signal from the same uploads.

Extends `factory/flywheel.py` + `agents/` :

- **Beat ledger:** the script JSON already has turns/beats; the ship DAG emits a `beats.json` mapping every script beat to its exact timestamp span in the final video (free — the aligner knows).
- **Retention-curve join:** pull the per-video audience-retention curve (free YouTube Analytics API), resample onto the beat ledger → every beat gets a retention delta. Aggregated across videos, every *type* of beat (hook style, reveal pacing, gesture density, emotion arc, formula-on-screen vs. spoken) accumulates evidence.
- **Beat-level bandit:** the Thompson sampler's arm space extends from video-level genes to beat-level policies: hook archetype, time-to-first-reveal, cut-rate curve shape, affect-arc template (§XVII gives every video a parameterized emotional trajectory — now it's learnable), caption density. Week 1 it guesses; week 6 it knows *which second* loses viewers and why.
- **Comment mining → topic engine:** `agents/` gains a comment miner (free Data API): cluster questions and misconceptions from comments; feed the scriptwriter a ranked queue of "what your actual audience is confused about." Combined with a JEE-syllabus coverage graph (topic dependency DAG, spaced-repetition scheduling of revisits), topic selection stops being guesswork forever.
- **Counterfactual discipline:** every learned policy change ships as an A/B arm first, never as a global flip; the ledger records which policy version produced every video, so regressions are attributable.
- **QC gate (new):** the publisher refuses upload if the beat ledger is missing or fails checksum against the muxed timeline — the learning loop's ground truth is gate-protected like everything else.

---

## Part XX — The performance engine: render speed is a creative feature

Iteration speed compounds into quality: a pipeline that renders 5× faster tunes 5× more. Target: **≤ 0.5× realtime** for a 60 s Short at 1080×1920@60 on the M2 (≤ 30 s wall-clock per render after caches warm).

- **Dirty-region rendering:** per frame, only the head box, gesture-swept regions, caption band, and particle deltas change; the background scene composite is reused. The frame becomes a sparse update, not a full repaint.
- **numpy compositing core:** replace per-frame PIL `alpha_composite` chains with a single vectorized premultiplied-alpha numpy pipeline over uint16 intermediates (no float boxing, no PIL object churn). PIL remains for I/O and text only.
- **Scene-parallel rendering:** scenes are independent given the rig and audio — `multiprocessing` over scenes with deterministic per-scene seeds; frame ledger hashes are order-independent by construction. M2 has 8 cores; use them.
- **Pipe to encoder:** frames stream to ffmpeg over stdin (rawvideo) — no PNG encode/decode round-trip on the critical path; PNG dumps become a `--debug-frames` mode.
- **Warm-cache contract:** LRU layers (head plates, transformed heads, mouth shapes, warp grids) sized so a typical script's working set fits in ~4 GB; hit-rate reported per render, regression-gated.
- **Perf gates (new):** wall-clock per video and per-stage budgets tracked in the ledger; CI fails a commit that regresses golden-corpus render time > 15%. Speed is a gated invariant, exactly like pixels.

---

## Part XXI — Adversarial QC: attack the renderer, don't just check it

The Terminal QC verifies known invariants. The Singularity QC goes looking for unknown failures — because at millions of views, one-in-ten-thousand-frames bugs WILL be screenshotted.

`tests/adversarial/` (new):

- **Property-based testing (`hypothesis`):** generate thousands of random-but-valid inputs — parameter vectors, alignment tracks, pose sequences, affect trajectories — and assert the invariants hold on ALL of them: mouth continuity, no fold-overs, blink closure, jerk budget, weight-mass conservation. Bugs are found by search, not by luck.
- **Metamorphic tests:** transformations with known output relations — time-reversed audio must produce time-reversed jaw envelope; a 2× slowed script must produce identical shapes at 2× spacing; muting a turn must produce a sealed resting mouth for its span. Each violated relation is a bug no example-based test would find.
- **Renderer fuzzing:** malformed scripts, zero-length turns, emoji in dialogue, 400-word run-ons, single-frame scenes, characters with 1 pose — the pipeline must fail *loudly and diagnosably* (typed errors with artifact paths), never render garbage silently. Silent garbage at scale is channel death.
- **Perceptual flicker metric:** temporal SSIM between consecutive frames in held moments — catches shimmer, cache-boundary popping, and dithering artifacts that per-landmark checks can't see. Threshold calibrated on blessed corpus renders.
- **Optical-flow sanity:** dense flow (cv2 Farnebäck) over rendered output; flow magnitude must be spatially coherent with commanded motion — a region moving when nothing commanded it (the historical "third mouth" class) is detected *generically*, forever, without knowing the bug in advance.
- **The nightly gauntlet:** hypothesis corpus + fuzz corpus + flicker + flow run nightly on the M2 (it's asleep anyway); failures land as diagnostic sheets in the runbook queue. The machine hunts its own bugs while you sleep.

---

## Part XXII — Order of work for the Singularity layer (each independently gated)

```
S0. Perf engine (§XX)                      → 0.5× realtime gate; makes every later step iterate faster — DO THIS FIRST
S1. Motion grammar (§XVI)                  → jerk/settle/smear gates green on golden corpus
S2. Affect engine (§XVII)                  → coherence audit green; listener reactions on dialogue corpus
S3. 2.5D depth head (§XV)                  → fold-over + silhouette + sweep gates green; A/B blessed vs. flat mode
S4. Multimodal coherence (§XVIII)          → breath-sync + shared-reverb + duck-timing gates green
S5. Adversarial QC (§XXI)                  → nightly gauntlet running clean for 7 consecutive nights
S6. Retention intelligence (§XIX)          → beat ledger in every upload; first beat-level posterior after 20 videos
```

**Dependency contract:** S0–S5 depend only on the Terminal core (steps 0–9). S6 depends on the ship DAG (step 10). Any Singularity part can be deferred without blocking publishing — the channel ships on the Terminal core from day one and gets *inevitably* better as each layer lands, each behind its own gate, each reversible by flag.

**The end state:** a machine that renders faster than realtime, moves with physics, feels with one nervous system, breathes audibly and visibly in sync, turns its head in true depth, hunts its own bugs nightly, and learns from every second of every viewer's attention — deterministic to the byte, free to the core, gated to the pixel. There is no ceiling above this because above this there is no ceiling.
