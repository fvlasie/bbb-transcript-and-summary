#!/usr/bin/env python3
"""Package a published recording and send it to the transcript worker.

The post-publish hook calls this and returns. A missing upload is recorded as
pending-submit so the retrieve timer can try again. This process exits 0 so a
transcript failure does not fail the already-published presentation.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import http.client

from bbb_jobs import BBBConfig, blank_job, load_bbb_config, load_job, log_line, now_stamp, save_job
from bbb_publish import get_meeting_metadata
from worker_client import WorkerClient, WorkerError, valid_meeting_id, write_tar


class PermanentError(Exception):
    """Local problem that will not succeed on the next timer tick."""


def locate_source_audio(config: BBBConfig, meeting_id: str) -> Path:
    presentation = config.published_dir / "presentation" / meeting_id
    for relative in ("video/webcams.webm", "audio/audio.webm"):
        candidate = presentation / relative
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    raise PermanentError(f"No webcams.webm or audio.webm published for {meeting_id}")


def build_payload_tar(config: BBBConfig, meeting_id: str, tar_path: Path) -> None:
    source = locate_source_audio(config, meeting_id)
    staging = config.jobs_dir / "work" / meeting_id
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    opus_path = staging / "audio.opus"
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise PermanentError("ffmpeg is not installed")
    try:
        completed = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-nostdin",
                "-i",
                str(source),
                "-vn",
                "-ac",
                "1",
                "-c:a",
                "libopus",
                "-b:a",
                "32k",
                "-loglevel",
                "error",
                str(opus_path),
            ],
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except subprocess.TimeoutExpired as exc:
        raise PermanentError("ffmpeg timed out while extracting audio") from exc
    if completed.returncode != 0 or not opus_path.is_file() or opus_path.stat().st_size <= 0:
        detail = (completed.stderr or completed.stdout or "ffmpeg failed").strip()
        raise PermanentError(detail[-300:] or "ffmpeg failed")

    files = {"audio.opus": opus_path}
    events = config.raw_dir / meeting_id / "events.xml"
    if events.is_file():
        shutil.copyfile(events, staging / "events.xml")
        files["events.xml"] = staging / "events.xml"
    else:
        log_line(config, f"No events.xml for {meeting_id}; transcript will have no speaker names")

    manifest = {"meeting_id": meeting_id, "source": "bigbluebutton"}
    presentation = config.published_dir / "presentation" / meeting_id
    if presentation.is_dir():
        meeting_name, meeting_date = get_meeting_metadata(str(presentation))
        manifest["meeting_name"] = meeting_name
        manifest["meeting_date"] = meeting_date
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    files["manifest.json"] = manifest_path
    write_tar(tar_path, files)


def _cleanup_payload(config: BBBConfig, meeting_id: str, tar_path: Path) -> None:
    staging = config.jobs_dir / "work" / meeting_id
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    if tar_path.exists():
        tar_path.unlink()


def _record(config: BBBConfig, meeting_id: str, **updates) -> dict:
    job = load_job(config, meeting_id) or blank_job(meeting_id)
    job.update(updates)
    save_job(config, job)
    return job


def submit_meeting(meeting_id: str, config: BBBConfig, fresh: bool = False) -> dict | None:
    """Upload one meeting. fresh=True is the post-publish hook and can reopen a failed job."""
    if not valid_meeting_id(meeting_id):
        log_line(config, f"Refusing invalid meeting id {meeting_id!r}")
        return None

    existing = load_job(config, meeting_id)
    if existing and existing.get("status") == "failed" and not fresh:
        return existing

    tar_path = config.jobs_dir / "work" / f"{meeting_id}.tar"
    try:
        build_payload_tar(config, meeting_id, tar_path)
        result = WorkerClient(config.worker_url, config.worker_token).post_job(meeting_id, tar_path)
    except PermanentError as exc:
        log_line(config, f"Cannot submit {meeting_id}: {exc}")
        return _record(config, meeting_id, status="failed", error=str(exc)[:500], worker_status=None)
    except WorkerError as exc:
        if exc.status is not None and 400 <= exc.status < 500 and exc.status not in (408, 429):
            log_line(config, f"Worker rejected {meeting_id}: {exc}")
            return _record(config, meeting_id, status="failed", error=str(exc)[:500], worker_status=None)
        log_line(config, f"Upload failed for {meeting_id}: {exc}")
        return _record(
            config,
            meeting_id,
            status="pending-submit",
            error=str(exc)[:500],
            worker_status=None,
        )
    except (OSError, http.client.HTTPException) as exc:
        log_line(config, f"Upload failed for {meeting_id}: {exc}")
        return _record(
            config,
            meeting_id,
            status="pending-submit",
            error=str(exc)[:500],
            worker_status=None,
        )
    finally:
        _cleanup_payload(config, meeting_id, tar_path)

    job = load_job(config, meeting_id) or blank_job(meeting_id)
    if result.get("created"):
        job["attempts"] = 1 if fresh else int(job.get("attempts") or 0) + 1
    elif fresh and job.get("status") == "failed":
        job["attempts"] = int(job.get("attempts") or 0)
    job["status"] = "submitted"
    job["worker_status"] = result.get("status")
    job["error"] = None
    if not job.get("submitted_at"):
        job["submitted_at"] = now_stamp()
    save_job(config, job)
    log_line(
        config,
        f"Submitted {meeting_id} worker_status={job['worker_status']} attempts={job['attempts']}",
    )
    return job


def main(argv: list[str]) -> int:
    try:
        config = load_bbb_config()
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 0
    if len(argv) != 1:
        print("Usage: bbb_submit.py <meeting_id>", file=sys.stderr)
        return 0
    submit_meeting(argv[0], config, fresh=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
