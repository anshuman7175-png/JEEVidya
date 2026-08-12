# CI activation

`ci.yml` in this folder is the complete GitHub Actions gate (lint + pure-logic
tests) described in ADR-0001. It lives here — not in `.github/workflows/` —
because the GitHub App that pushes from v0 does not have the `workflows`
permission needed to create workflow files.

To activate CI (one-time, ~30 seconds):

1. On GitHub, open this branch and navigate to `docs/ci/ci.yml`.
2. Copy its contents into a new file at `.github/workflows/ci.yml`
   (GitHub web editor: **Add file → Create new file**, path
   `.github/workflows/ci.yml`, paste, commit).
3. Optionally delete `docs/ci/ci.yml` afterwards — `.github/workflows/ci.yml`
   becomes the single source of truth.

Once the file exists in `.github/workflows/`, future edits to it must also be
made directly on GitHub (or from a locally authenticated clone), for the same
permission reason.
