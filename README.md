# JEEVidya — Automated AI Video Factory

A fully automated, high-retention Python video creator designed specifically for JEE/NEET educational YouTube Shorts.

## The Problem
Standard AI video generators dump random colors, unstructured facts, and hallucinatory math on screen. They produce "robotic" content that YouTube suppresses.

## The Solution (The Three-Layer Factory)
JEEVidya is built as a factory pipeline with strict, non-negotiable pedagogical rules:

1. **The Brain (Planning & Storyboard):**
   - **`config/prompts.py`**: Enforces "Geometry before Algebra", mandatory breathing room, and the "Why before the How".
   - **`pipeline/scriptwriter.py`**: Uses Gemini (via `google-genai`) to generate structured JSON storyboards ensuring one action = one spoken sentence.

2. **The Look (Brand & Generation):**
   - **`config/brand.py`**: A strict global visual system defining dark premium colors (Electric Cyan, Gold, Pink), opacity layers for visual hierarchy, and monospace typography.
   - **`engine/`**: A custom Matplotlib & PIL based renderer that bypasses the heavy limitations of Manim, providing direct control over geometric primitives, easing animations, and LaTeX rendering (without needing a local TeX distribution).

3. **The Assembly Line (Orchestration):**
   - **`pipeline/voice.py`**: High-quality Hindi TTS and subtitle timing extraction via `edge-tts`.
   - **`pipeline/caption_engine.py`**: Karaoke-style word-chunking captions.
   - **`pipeline/compositor.py`**: Final stitch using MoviePy.
   - **`generate.py`**: The master orchestrator script.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Export your Gemini API Key:
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```

## Usage
Run the pipeline by providing a physics or math concept:
```bash
python3 generate.py "How did Eratosthenes measure the Earth 2200 years ago?"
```

## Architecture
- `config/`: Brand identity, video settings, and LLM prompts.
- `engine/`: The core visual renderer (Canvas, Math, Easing, Animator).
- `pipeline/`: Workflow components (Script, Voice, Scene builder, Compositor).
- `assets/`: Static assets (fonts, background images if any).
- `.tmp/`: Intermediate build files (audio, frames).
- `output/`: Final rendered `.mp4` videos.

## Quality Control (QC)
This pipeline is designed for **Human-in-the-Loop** operation. While it automates 95% of the heavy lifting, you must:
1. Verify the generated JSON storyboard for factual accuracy before rendering.
2. Review the final MP4 to ensure the generated timing and subtitles are precise.

## MCP / Tasklet Setup
To connect an MCP client such as Tasklet to this workspace:

1. Open this workspace in VS Code.
2. Install the `@modelcontextprotocol/server-filesystem` server when prompted, or let VS Code use the workspace MCP config in `.vscode/mcp.json`.
3. In Tasklet, connect to the workspace MCP server and grant access to this repo folder.
4. If Tasklet requires a public URL instead of a local workspace bridge, run a tunnel for the MCP endpoint and use the `/mcp` path in the URL you provide.

The workspace MCP config exposes the repo through the standard filesystem MCP server, so Tasklet can read and search files without a custom Python server.
