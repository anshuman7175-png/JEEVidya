"""
JEEVidya V5 — Publisher (Tier 4)
════════════════════════════════
Two modes, auto-selected:

  UPLOAD  google-api-python-client + OAuth token present → schedule
          uploads via the YouTube Data API (free quota ≈ 6 videos/day:
          each upload costs 1600 of the 10k daily units).
  BUNDLE  otherwise → export ready-to-post folders (video + thumbnail +
          title/description/tags as post.txt) for manual upload.

Uploaded video IDs are appended to the flywheel ledger so Tier 4's
bandit can start learning from them.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from config import settings

CLIENT_SECRETS = os.path.join(settings.PROJECT_ROOT, "client_secrets.json")
TOKEN_PATH = os.path.join(settings.PROJECT_ROOT, ".cache", "yt_token.json")
EXPORT_DIR = os.path.join(settings.OUTPUT_DIR, "ready_to_post")
UPLOAD_COST_UNITS = 1600
DAILY_UNITS = 10_000
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


class GateRefusal(RuntimeError):
    """The publisher refused an artifact. Not a bug — the constitution."""


def admission_check(video: str, strict: bool = True) -> None:
    """The publisher's admission gate (Part XII §12.1 + Part XIX).

    Nothing un-gated reaches YouTube, mechanically rather than by
    convention:
      • a QC-pass manifest must exist, match the video's checksum, and
        report every delivery gate green (`pipeline/delivery_qc.py`);
      • a sealed beat ledger must exist and agree with the muxed
        timeline (`factory/beats.py`) — the learning loop's ground truth
        is gate-protected like the pixels are.

    `strict=False` (or JV_PUBLISH_UNGATED=1) downgrades to a loud warning
    for local dry runs; the batch/ship path never sets it.
    """
    if os.environ.get("JV_PUBLISH_UNGATED") == "1":
        strict = False
    problems: List[str] = []

    try:
        from pipeline.delivery_qc import verify_manifest
        ok, reason = verify_manifest(video)
    except Exception as e:                       # noqa: BLE001
        ok, reason = False, f"QC manifest unreadable: {e}"
    if not ok:
        problems.append(f"delivery QC: {reason}")

    try:
        from factory.beats import verify as verify_beats
        ok_b, reason_b = verify_beats(video)
    except Exception as e:                       # noqa: BLE001
        ok_b, reason_b = False, f"beat ledger unreadable: {e}"
    if not ok_b:
        problems.append(f"beat ledger: {reason_b}")

    if not problems:
        return
    msg = ("refusing to publish " + os.path.basename(video) + " — "
           + "; ".join(problems))
    if strict:
        raise GateRefusal(msg)
    print(f"  [Publish] WARNING (ungated): {msg}")


def can_upload() -> bool:
    try:
        import googleapiclient  # noqa: F401
        import google_auth_oauthlib  # noqa: F401
    except ImportError:
        return False
    return os.path.exists(CLIENT_SECRETS)


class Publisher:

    def publish_bundle(self, bundle_dir: str,
                       schedule_iso: Optional[str] = None,
                       strict: bool = True) -> Dict[str, Any]:
        """Publish one batch bundle (video.mp4 + meta.json + thumbnail)."""
        video = os.path.join(bundle_dir, "video.mp4")
        meta_path = os.path.join(bundle_dir, "meta.json")
        if not os.path.exists(video):
            raise FileNotFoundError(f"no video.mp4 in {bundle_dir}")
        meta: Dict[str, Any] = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        # The constitution, enforced before a single byte is uploaded.
        admission_check(video, strict=strict)

        if can_upload():
            return self._upload(video, bundle_dir, meta, schedule_iso)
        return self._export(video, bundle_dir, meta)

    # ─── Mode A: real upload ───────────────────────────────

    def _upload(self, video: str, bundle_dir: str, meta: Dict[str, Any],
                schedule_iso: Optional[str]) -> Dict[str, Any]:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload

        youtube = build("youtube", "v3", credentials=self._credentials())

        status: Dict[str, Any] = {"selfDeclaredMadeForKids": False}
        if schedule_iso:
            status.update({"privacyStatus": "private",
                           "publishAt": schedule_iso})
        else:
            status["privacyStatus"] = "public"

        body = {
            "snippet": {
                "title": meta.get("youtube_title", "JEEVidya Short")[:100],
                "description": meta.get("description", ""),
                "tags": meta.get("tags", [])[:30],
                "categoryId": "27",              # Education
            },
            "status": status,
        }
        request = youtube.videos().insert(
            part="snippet,status", body=body,
            media_body=MediaFileUpload(video, chunksize=-1, resumable=True))

        response = None
        while response is None:
            progress, response = request.next_chunk()
            if progress:
                print(f"  [Publish] upload {int(progress.progress() * 100)}%")
        video_id = response["id"]
        print(f"  [Publish] ✓ https://youtube.com/shorts/{video_id}")

        thumb = os.path.join(bundle_dir, "thumbnail.jpg")
        if os.path.exists(thumb):
            try:
                youtube.thumbnails().set(
                    videoId=video_id,
                    media_body=MediaFileUpload(thumb)).execute()
            except Exception as e:               # noqa: BLE001 — thumb optional
                print(f"  [Publish] thumbnail skipped: {e}")

        # Register with the flywheel so learning starts immediately
        try:
            from factory.flywheel import Flywheel
            Flywheel().register_video(video_id, meta)
        except Exception as e:                   # noqa: BLE001
            print(f"  [Publish] flywheel registration skipped: {e}")

        # Pin the (gate-verified) beat ledger so per-moment retention can
        # be joined onto it as soon as Analytics has data.
        try:
            from factory.retention import RetentionEngine
            RetentionEngine().register_from_video(
                video_id, video, arm=meta.get("policy_arm"))
        except Exception as e:                   # noqa: BLE001
            print(f"  [Publish] beat registration skipped: {e}")

        # Advance syllabus coverage so the scheduler stops re-proposing it.
        topic = meta.get("topic_id")
        if topic:
            try:
                from factory.syllabus import Syllabus
                Syllabus().mark_taught(str(topic), video_id)
            except Exception as e:               # noqa: BLE001
                print(f"  [Publish] syllabus update skipped: {e}")

        return {"mode": "upload", "video_id": video_id}

    def _credentials(self):
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow

        creds = None
        if os.path.exists(TOKEN_PATH):
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    CLIENT_SECRETS, SCOPES)
                creds = flow.run_local_server(port=0)
            os.makedirs(os.path.dirname(TOKEN_PATH), exist_ok=True)
            with open(TOKEN_PATH, "w", encoding="utf-8") as f:
                f.write(creds.to_json())
        return creds

    # ─── Mode B: manual bundle export ──────────────────────

    def _export(self, video: str, bundle_dir: str,
                meta: Dict[str, Any]) -> Dict[str, Any]:
        name = os.path.basename(bundle_dir.rstrip("/"))
        out = os.path.join(EXPORT_DIR, name)
        os.makedirs(out, exist_ok=True)
        import shutil
        shutil.copy2(video, os.path.join(out, "video.mp4"))
        thumb = os.path.join(bundle_dir, "thumbnail.jpg")
        if os.path.exists(thumb):
            shutil.copy2(thumb, os.path.join(out, "thumbnail.jpg"))
        with open(os.path.join(out, "post.txt"), "w", encoding="utf-8") as f:
            f.write(f"TITLE:\n{meta.get('youtube_title', '')}\n\n"
                    f"DESCRIPTION:\n{meta.get('description', '')}\n\n"
                    f"TAGS:\n{', '.join(meta.get('tags', []))}\n")
        print(f"  [Publish] ✓ ready-to-post bundle: {out}")
        return {"mode": "bundle", "path": out}

    # ─── Batch publishing with quota pacing ────────────────

    def publish_all(self, batch_dir: Optional[str] = None,
                    per_day: int = DAILY_UNITS // UPLOAD_COST_UNITS
                    ) -> List[Dict[str, Any]]:
        from factory.batch import BATCH_DIR
        batch_dir = batch_dir or BATCH_DIR
        results = []
        published = 0
        for name in sorted(os.listdir(batch_dir) if os.path.isdir(batch_dir)
                           else []):
            bundle = os.path.join(batch_dir, name)
            if not os.path.isfile(os.path.join(bundle, "video.mp4")):
                continue
            if can_upload() and published >= per_day:
                print(f"  [Publish] daily upload quota reached ({per_day}); "
                      "remaining bundles exported instead")
            try:
                if can_upload() and published < per_day:
                    results.append(self.publish_bundle(bundle))
                    published += 1
                    time.sleep(3)
                else:
                    video = os.path.join(bundle, "video.mp4")
                    admission_check(video)
                    meta_path = os.path.join(bundle, "meta.json")
                    meta = {}
                    if os.path.exists(meta_path):
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                    results.append(self._export(video, bundle, meta))
            except Exception as e:               # noqa: BLE001 — isolate bundles
                print(f"  [Publish] {name} failed: {e}")
                results.append({"mode": "error", "bundle": name,
                                "error": str(e)})
        return results
