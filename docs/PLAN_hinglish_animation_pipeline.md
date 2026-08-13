# Plan — Hinglish Animation Pipeline Round

Status: **plan only, no code changes in this commit.**
Target branch for implementation: `hinglish-animation-pipeline`.

## Decisions locked in

- **You write the script.** No Gemini dependency on the main path; the existing
  Gemini topic-mode stays exactly as-is.
- **Voice:** Qwen3-TTS (Apache 2.0) as the primary clone engine — monetization-safe —
  with the existing `edge-tts` path as an always-available fallback.
- **F5-TTS is rejected:** CC-BY-NC (non-commercial) licensing would legally block
  channel monetization, and its Hindi quality is weak.
- **Priority:** maximum focus on visuals and movement perfection.

## Environment reality

The sandbox has Python 3 but **no ffmpeg and no torch**. So the round is engineered as
pure-Python logic plus frame-level code, built and test-verified in the sandbox
(`pytest` + `numpy` + `Pillow`), with full renders running on the MacBook M2 exactly as
they do today. `tests/conftest.py` already skips render-stack modules loudly.

---

## Workstream 1 — Raw-script formatter + one-command video

**Files:** `pipeline/scriptwriter.py` (extend), `jvmake.py` (new `video` command),
`tests/test_formatter.py` (new)

New `parse_raw_script(text) -> dialogue dict` in `scriptwriter.py`:

- **Speaker marks:** `C:` / `G:` (also accepts `Chintu:` / `Gudiya:`, case-insensitive).
- **Unmarked lines** alternate speakers automatically, starting with Gudiya
  (the question-asker).
- **Inline cues, all optional:**
  - `[curious]` emotion tags, validated against `VALID_EMOTIONS`, else inferred
  - `[board: E = mc^2]` produces explanation turns with visual elements
  - `[beat]` bumps `pause_after`
- **Emotion inference when untagged:** `?` → curious, `!` → enthusiastic/amazed
  (keyword-weighted), default explaining/neutral — mapped to the exact enums
  `EMOTION_BASELINE` in `puppet.py` consumes.
- **Shot auto-assignment** reusing the existing `VALID_SHOTS` grammar: question →
  `two_shot`, reveal/exclaim → `reaction_cut`/`reveal`, board turns →
  `fullscreen_explain`, with the ≥12-frame cut discipline respected downstream.
- Output goes through the existing `validate_dialogue()` self-heal path — **one schema,
  no parallel format.**

New CLI:

```
python3 jvmake.py video --script my_script.txt [--out ...]
```

format → save dialogue JSON → hand off to the existing render DAG.
`--format-only` inspects the JSON without rendering.

## Workstream 2 — Movement perfection (top priority)

**Files:** `pipeline/frame_qc.py` (NEW), `pipeline/delivery_qc.py` (extend),
`engine/motion_grammar.py` / `pipeline/puppet.py` (targeted polish),
`tests/test_frame_qc.py` (new)

`pipeline/frame_qc.py` — Ring 2, per-frame verification. A numpy/Pillow scanner run over
rendered frame sequences that detects:

- frozen/duplicate-frame runs longer than 0.7 s (hash + pixel-delta)
- black/blank frames
- teleports: any sprite/region jumping more than a threshold in px between consecutive
  frames without a registered cut
- layer seams at known body-part boundaries (alpha gap scan)
- caption safe-area collisions for 9:16 Shorts (top 220 px / bottom 320 px UI zones from
  `config/brand.py`, added there if absent)

It emits a frame-numbered fault report and is wired as a **mandatory stage before encode**
in the build graph.

**Lipsync-drift gate (< 40 ms)** in `delivery_qc.py`: compare each line's phoneme timeline
(from `engine/align.py` output already on the timeline) against the viseme track actually
rendered. Max drift per line must stay under 40 ms or the video is quarantined with the
offending line numbers.

**Movement polish audit + fixes.** Verify with unit tests — not hope — that the
already-written polish layers are actually wired into the frame path, and wire whatever is
dead:

- eyes lead head turns by 3–5 frames (`eye_model` ↔ `puppet.pose_at`)
- smear taps fire on fast head turns (`motion_grammar.smear_taps` → compositor)
- anticipation injection applied to gesture/head channel tracks
- listener coupling visible (nods/brow raises on the non-speaker)
- J/L cuts: audio leads picture by 2–4 frames on dialogue turns
  (`timeline`/`shot_sequencer`)
- minimum shot length of 12 frames enforced in `shot_sequencer`

Each item gets a small test in `tests/test_naturalism.py` or a new test file, so a
regression can never be silent again.

## Workstream 3 — Visual perfection for Shorts

**Files:** `tools/face_qc.py` (extend), `engine/post_production.py` (verify/complete),
`pipeline/compositor_v5.py` (verify), `config/brand.py`

- **Ring 1 asset gates** in `face_qc.py`: alpha-halo detection, consistent scale/anchor
  across the 17-pose banks (`character1.png`…`character2p.png`), mouth-region registration
  within 2 px across each character's viseme set. A bad sprite is rejected at build time.
- **Film-look grade** as the final compositor pass (lifted blacks, gentle S-curve, 2 %
  grain, subtle vignette) — complete/verify `post_production.py` and ensure it is actually
  in the frame path for Shorts renders.
- **Shorts framing:** confirm 1080×1920 layout, karaoke caption placement inside safe
  areas, hook framing for the first 2 seconds (speaker close-up + brow-raise entry gesture
  on turn 1).

## Workstream 4 — Voice: Qwen3-TTS clone + edge-tts fallback

**Files:** `pipeline/voice_clone.py` (add adapter), `pipeline/voice.py` (fallback routing),
`requirements-voice.txt`, docs note

- **Primary:** `Qwen3TTSAdapter` (Apache 2.0) behind the existing engine-adapter contract
  in `voice_clone.py`, the same interface as the IndexTTS-2 / Chatterbox adapters. 2026
  leader for Apple Silicon (MPS), clones from a ~3-second sample, native Hindi + English —
  ideal for Hinglish. Keeps every existing hard gate: speaker-identity cosine ≥ 0.72,
  pronunciation gate, −16 LUFS per line, deterministic seeds, content cache. Gudiya's voice
  = your sample plus a pitch/formant shift preset, or a second sample if provided.
- **Secondary:** Chatterbox Multilingual v3 adapter (MIT — also monetization-safe)
  completed/verified as engine option #2, since the adapter contract already exists in the
  repo. Engine selection via the `voice_bank.json` `"engine"` field.
- **Automatic fallback:** if `assets/voices/voice_bank.json` is missing or the clone stack
  isn't installed, the pipeline logs one loud warning and uses the existing edge-tts path
  unchanged. Renders never block on the voice stack.
- `requirements-voice.txt` gains the Qwen3-TTS install lines with Mac/MPS notes, and a docs
  note records the license reasoning so the monetization guarantee is on record.

## Workstream 5 — Verification in the sandbox

- `pip install numpy pillow pytest hypothesis`, then run the pure-logic suite (`tests/`)
  before and after changes.
- New tests:
  - `test_formatter.py` — marked / unmarked / cued scripts produce valid dialogue
  - `test_frame_qc.py` — synthetic frame sequences with planted faults must all be caught
  - lipsync-drift gate test with a synthetic phoneme/viseme pair
- Everything needing ffmpeg or torch (encode, TTS) is fallback-guarded and documented for
  the MacBook run: `python3 jvmake.py video --script your_script.txt`.

---

## Explicitly out of scope this round

- Gemini topic-mode changes (kept as-is)
- Gudiya sprite regeneration via image generation (asset work — separate round if wanted)
- YouTube upload automation

## Order of execution

1. Sandbox test baseline (install dev deps, run suite)
2. Workstream 1 — formatter + `video` command (unblocks your scripts immediately)
3. Workstream 2 — frame_qc + drift gate + movement wiring audit/fixes
4. Workstream 3 — asset gates + film grade + Shorts framing
5. Workstream 4 — Qwen3-TTS adapter + fallback
6. Full test pass, commit to `hinglish-animation-pipeline`, push + PR
