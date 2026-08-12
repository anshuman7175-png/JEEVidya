"""Restore the uploaded viseme/mouth reference art from the tracked manifest.

`assets/` is gitignored, so the raw PNGs never survive a fresh clone. The
manifest at ``config/mouth_art_manifest.json`` is tracked and holds the source
URL plus a sha256 for every image, so the art can always be rebuilt byte-for-byte
without re-uploading anything.

Usage::

    python -m tools.fetch_mouth_art            # download anything missing/changed
    python -m tools.fetch_mouth_art --verify   # check only, never write
    python -m tools.fetch_mouth_art --force    # re-download everything
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(REPO_ROOT, "config", "mouth_art_manifest.json")


def load_manifest(path: str = MANIFEST_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as out:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    os.replace(tmp, dest)


def sync(verify_only: bool = False, force: bool = False) -> int:
    manifest = load_manifest()
    failures: list[str] = []

    for asset in manifest["assets"]:
        dest = os.path.join(REPO_ROOT, asset["file"])
        expected = asset["sha256"]

        have = os.path.exists(dest) and not force
        if have and sha256_file(dest) == expected:
            print(f"ok       {asset['id']}  {asset['file']}")
            continue

        if verify_only:
            state = "corrupt" if os.path.exists(dest) else "missing"
            print(f"{state:<8} {asset['id']}  {asset['file']}")
            failures.append(asset["id"])
            continue

        print(f"fetch    {asset['id']}  {asset['file']}")
        try:
            _download(asset["source_url"], dest)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"ERROR    {asset['id']}  {exc}", file=sys.stderr)
            failures.append(asset["id"])
            continue

        actual = sha256_file(dest)
        if actual != expected:
            print(
                f"ERROR    {asset['id']}  sha256 mismatch\n"
                f"         expected {expected}\n"
                f"         actual   {actual}",
                file=sys.stderr,
            )
            failures.append(asset["id"])

    total = len(manifest["assets"])
    if failures:
        print(f"\n{len(failures)}/{total} asset(s) unresolved: {', '.join(failures)}", file=sys.stderr)
        return 1

    print(f"\nall {total} mouth assets present and verified -> {manifest['dest_dir']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="report status without downloading")
    parser.add_argument("--force", action="store_true", help="re-download every asset")
    args = parser.parse_args()
    return sync(verify_only=args.verify, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
