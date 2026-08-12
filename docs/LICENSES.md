# License Ledger

Phase −1 standing mechanism (Terminal Plan, Part II). Every external
tool, model, or dataset lands here **in the same change that
introduces it**, with its license and a verdict. Rule of the ledger:

- **OK** — permissive or weak-copyleft; fine even if JEEVidya monetizes.
- **BRIDGE** — permitted temporarily under Doctrine Law 3 with a named
  local successor; must not become load-bearing.
- **BANNED** — anything **CC-BY-NC** or otherwise non-commercial is
  banned outright (this rules out some Meta MMS checkpoints — see
  Gate 2). GPL is acceptable for *tools we run*, not for code we link
  into distributable artifacts.

Review trigger: any change to `requirements*.txt`, any new model
checkpoint, any new font/asset.

## Runtime (`requirements.txt`)

| Dependency | License | Verdict | Notes |
| --- | --- | --- | --- |
| Pillow | MIT-CMU (HPND) | OK | |
| matplotlib | Matplotlib License (PSF-based) | OK | |
| moviepy | MIT | OK | |
| numpy | BSD-3-Clause | OK | |
| Flask | BSD-3-Clause | OK | |
| pydub | MIT | OK | |
| rembg | MIT | OK | Bundled U2-Net weights are Apache-2.0 |
| numba | BSD-2-Clause | OK | |
| onnxruntime | MIT | OK | |
| mcp | MIT | OK | |
| imageio-ffmpeg | BSD-2-Clause | OK | Downloads an ffmpeg binary (LGPL build); we invoke it as a subprocess, we do not link it |
| audioop-lts | PSF-2.0 | OK | Stdlib backfill for 3.13+ |
| mediapipe | Apache-2.0 | OK | Face-landmark models are Apache-2.0 |
| opencv-contrib-python-headless | Apache-2.0 | OK | PyPI wheel excludes the non-free contrib modules |
| google-genai | Apache-2.0 | BRIDGE | Client is Apache-2.0 but the Gemini API is a free-tier cloud service (Law 3: "a free tier is a paid tool with a delay"). Successor: local LLM via the same `LLMClient` seam |
| edge-tts | LGPL-3.0 | BRIDGE | Library is LGPL (used as-is, unmodified, via import — permitted); the Edge TTS *service* is free-tier cloud. Named successor: IndexTTS-2 / Chatterbox (`requirements-voice.txt`), per Terminal Plan Part X |

## Voice stack (`requirements-voice.txt`)

| Dependency | License | Verdict | Notes |
| --- | --- | --- | --- |
| indextts (IndexTTS-2) | Apache-2.0 | OK | Primary local TTS engine; checkpoint license verified Apache-2.0 |
| chatterbox-tts | MIT | OK | Fallback engine; Hindi supported |
| soundfile | BSD-3-Clause | OK | |
| pyloudnorm | MIT | OK | |
| resemblyzer | Apache-2.0 | OK | Optional identity gate |
| torch | BSD-3-Clause | OK | |
| torchaudio | BSD-2-Clause | OK | |
| uroman | MIT | OK | |
| Meta MMS alignment checkpoints | **varies — some CC-BY-NC** | **BANNED unless verified** | Verify the specific checkpoint's card before download; NC checkpoints are banned (Gate 2) |

## Developer / CI (`requirements-dev.txt`)

| Dependency | License | Verdict | Notes |
| --- | --- | --- | --- |
| pytest | MIT | OK | |
| hypothesis | MPL-2.0 | OK | Weak copyleft, file-scoped; dev-only, never shipped |
| ruff | MIT | OK | |

## Assets and models not yet in the ledger

Character art, fonts, reference audio, and any future model weights
must be added here before they are committed or content-addressed
(Phase −1 artifact-versioning item). An asset with no ledger row is a
release blocker, not a footnote.
