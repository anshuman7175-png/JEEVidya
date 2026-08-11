"""
JEEVidya — Constants & Dead-Symbol Lint (Terminal Plan, Parts II §2.2 + IX)
═══════════════════════════════════════════════════════════════════════════
Two bug classes, made unrepresentable (Law 1):

1. PER-FRAME LITERALS: a constant tuned as "N frames" or "N px/frame"
   silently runs 2× fast when FPS moves 30 → 60. New per-frame literals
   in config/ are banned; time-dependent quantities must be per-second
   and converted via settings.frame_ms()/frames()/per_frame().

2. RESURRECTED DEAD CODE: symbols deleted in Phase 7 (the historical
   sources of the mouth-on-the-eyes bug) must never return. A grep-gate
   fails CI if any banned symbol is DEFINED again anywhere in the tree.

Run: python -m tools.lint_constants   (exit 0 = clean, 1 = violation)
"""
from __future__ import annotations

import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Symbols the Terminal plan deletes (Part IX). Definitions of these names
# must never reappear. Mentions in comments/strings are fine — we match
# actual def statements.
BANNED_DEFINITIONS = [
    "_feathered_backing",
    "_ring_clutter",
    "_tone_match",
    "_bake_visemes",
    "_bake_eyelids",
    "_crop_lid_feathered",
    "_make_backing",
    "_staged_head",
]

# Per-frame literal patterns banned in config/. The name itself encodes
# the violation: *_FRAMES / *_PER_FRAME assigned to a bare number.
_FRAME_LITERAL = re.compile(
    r"^\s*[A-Z_]*(?:FRAMES|PER_FRAME)\s*:\s*\w+\s*=\s*[\d.]+\s*(?:#.*)?$",
    re.MULTILINE)
# .cuda( must never appear (Part II §2.1: no CUDA anywhere).
_CUDA = re.compile(r"\.cuda\(")

SCAN_DIRS = ("engine", "pipeline", "tools", "config", "factory", "agents")
# Files allowed to *mention* banned names (this linter, and tests that
# assert their absence).
_SELF = os.path.abspath(__file__)


def code_only(src: str) -> str:
    """Return `src` with comments and string literals blanked out.

    Without this the linter flags its own doctrine: a docstring saying
    "`.cuda()` NEVER appears in this codebase" reads as a violation. A
    gate that reports false positives gets switched off, and a gate that
    is switched off protects nothing — so the match runs on real tokens.
    Line structure is preserved so reported line content stays truthful.
    """
    import io
    import tokenize
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Unparseable file: fall back to raw text. Better a false
        # positive than a silently unscanned file.
        return src

    lines = src.splitlines(keepends=True)
    out = list(lines)
    for tok in toks:
        if tok.type not in (tokenize.COMMENT, tokenize.STRING):
            continue
        (r0, c0), (r1, c1) = tok.start, tok.end
        for row in range(r0, r1 + 1):
            i = row - 1
            if i >= len(out):
                break
            line = out[i]
            start = c0 if row == r0 else 0
            end = c1 if row == r1 else len(line.rstrip("\n"))
            keep_nl = "\n" if line.endswith("\n") else ""
            body = line.rstrip("\n")
            body = body[:start] + " " * max(0, end - start) + body[end:]
            out[i] = body + keep_nl
    return "".join(out)


def _py_files():
    for d in SCAN_DIRS:
        root = os.path.join(PROJECT_ROOT, d)
        if not os.path.isdir(root):
            continue
        for dirpath, _, names in os.walk(root):
            for n in names:
                if n.endswith(".py"):
                    yield os.path.join(dirpath, n)


def run() -> int:
    violations: list[str] = []
    def_patterns = [
        (name, re.compile(rf"^\s*def\s+{re.escape(name)}\s*\(", re.MULTILINE))
        for name in BANNED_DEFINITIONS
    ]

    for path in _py_files():
        if os.path.abspath(path) == _SELF:
            continue
        rel = os.path.relpath(path, PROJECT_ROOT)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
        src = code_only(raw)

        for name, pat in def_patterns:
            if pat.search(src):
                violations.append(
                    f"{rel}: banned symbol `{name}` is defined again "
                    f"(deleted in Terminal Plan Part IX — it must stay dead)")

        if _CUDA.search(src):
            violations.append(f"{rel}: `.cuda(` found — this codebase is "
                              f"MPS/CPU only (Part II §2.1)")

        if rel.startswith("config") and rel.endswith("settings.py"):
            for m in _FRAME_LITERAL.finditer(src):
                line = m.group(0).strip()
                # The derived block at the bottom of settings.py assigns
                # from function calls, not literals — those don't match.
                violations.append(
                    f"{rel}: per-frame literal `{line}` — express it "
                    f"per-second and derive with settings.per_frame()")

    if violations:
        print("CONSTANTS LINT: FAIL")
        for v in violations:
            print(f"  ✗ {v}")
        return 1
    print("CONSTANTS LINT: OK — no per-frame literals, no resurrected "
          "dead symbols, no CUDA")
    return 0


if __name__ == "__main__":
    sys.exit(run())
