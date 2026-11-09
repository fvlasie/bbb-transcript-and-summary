#!/usr/bin/env python3
"""Pull finished transcripts back onto the BigBlueButton host and publish them."""

import shutil
import sys
from pathlib import Path

import http.client

from bbb_jobs import BBBConfig, list_jobs, load_bbb_config, load_job, log_line, now_stamp, save_job
from bbb_publish import (
    generate_html_viewer,
    generate_rtf_from_md,
    get_meeting_metadata,
    own_published,
    register_custom_formats_native,
    summary_rtf_filename,
)
from bbb_submit import submit_meeting
from worker_client import RESULT_TAR_NAMES, WorkerClient, WorkerError, safe_extract_tar


class RetryableError(Exception):
    """The published recording is not ready to receive files yet."""


def publish_artifacts(config: BBBConfig, meeting_id: str, result_dir: Path) -> None:
    """Copy worker output into the presentation folder and register playback formats."""
    presentation = config.published_dir / "presentation" / meeting_id
    if not presentation.is_dir():
        raise RetryableError(f"Presentation directory is not ready: {presentation}")

    for name in ("transcript.txt", "transcript.vtt", "summary.md"):
        source = result_dir / name
        if not source.is_file() or source.stat().st_size <= 0:
            raise RetryableError(f"Result is missing {name}")

    transcript_dir = presentation / "transcript"
    summary_dir = presentation / "summary"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(result_dir / "transcript.txt", transcript_dir / "transcript.txt")
    shutil.copyfile(result_dir / "transcript.vtt", transcript_dir / "transcript.vtt")
    shutil.copyfile(result_dir / "summary.md", summary_dir / "summary.md")

    summary_text = (result_dir / "summary.md").read_text(encoding="utf-8").rstrip("\n")
    room_name, meeting_date = get_meeting_metadata(str(presentation))
    for stale in summary_dir.glob("Summary_*.rtf"):
        stale.unlink()
    rtf_filename = summary_rtf_filename(room_name, meeting_date)
    rtf_path = summary_dir / rtf_filename
    html_path = summary_dir / "summary.html"
    generate_rtf_from_md(summary_text, str(rtf_path), meeting_name=room_name, meeting_date=meeting_date)
    generate_html_viewer(summary_text, str(html_path), rtf_filename=rtf_filename, meeting_name=room_name, meeting_date=meeting_date)

    for path in (
        transcript_dir,
        transcript_dir / "transcript.txt",
        transcript_dir / "transcript.vtt",
        summary_dir,
        summary_dir / "summary.md",
        html_path,
        rtf_path,
    ):
        own_published(str(path))

    register_custom_formats_native(
        meeting_id=meeting_id,
        base_published_dir=str(config.published_dir),
        properties_path=str(config.properties_path),
    )
    summary_meta = config.published_dir / "summary" / meeting_id / "metadata.xml"
    transcript_meta = config.published_dir / "transcript" / meeting_id / "metadata.xml"
    if not summary_meta.is_file() or not transcript_meta.is_file():
        raise RetryableError("Playback metadata was not written")


def _result_complete(result_dir: Path) -> bool:
    return all(
        (result_dir / name).is_file() and (result_dir / name).stat().st_size > 0
        for name in ("transcript.txt", "transcript.vtt", "summary.md")
    )


def _already_published(config: BBBConfig, meeting_id: str) -> bool:
    presentation = config.published_dir / "presentation" / meeting_id
    transcript = presentation / "transcript" / "transcript.txt"
    summary = presentation / "summary" / "summary.html"
    return transcript.is_file() and summary.is_file()


def _mark_retrieved(config: BBBConfig, job: dict) -> None:
    job["status"] = "retrieved"
    job["worker_status"] = "ready"
    job["error"] = None
    job["retrieved_at"] = now_stamp()
    save_job(config, job)
    work = config.jobs_dir / "work" / job["meeting_id"]
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    log_line(config, f"Retrieved {job['meeting_id']}")


def _pull_and_publish(config: BBBConfig, job: dict, client: WorkerClient) -> None:
    meeting_id = job["meeting_id"]
    work = config.jobs_dir / "work" / meeting_id
    result_dir = work / "result"
    tar_path = work / "result.tar"
    try:
        if not _result_complete(result_dir):
            work.mkdir(parents=True, exist_ok=True)
            client.download_result(meeting_id, tar_path)
            if result_dir.exists():
                shutil.rmtree(result_dir)
            safe_extract_tar(tar_path, result_dir, set(RESULT_TAR_NAMES), required=set(RESULT_TAR_NAMES))
        publish_artifacts(config, meeting_id, result_dir)
        code = client.delete_job(meeting_id)
        if code not in (204, 404):
            raise WorkerError(f"Worker refused to delete the job ({code})", code)
    except RetryableError as exc:
        job["error"] = str(exc)[:500]
        save_job(config, job)
        log_line(config, f"Will retry publish for {meeting_id}: {exc}")
        return
    except (OSError, WorkerError, ValueError, http.client.HTTPException) as exc:
        job["error"] = str(exc)[:500]
        save_job(config, job)
        log_line(config, f"Retrieve failed for {meeting_id}: {exc}")
        return
    _mark_retrieved(config, job)


def _retry_or_give_up(config: BBBConfig, job: dict, remote_error: str | None, client: WorkerClient, submit) -> None:
    meeting_id = job["meeting_id"]
    message = (remote_error or "Worker failed")[:500]
    if int(job.get("attempts") or 0) >= config.max_attempts:
        job["status"] = "failed"
        job["worker_status"] = "failed"
        job["error"] = message
        save_job(config, job)
        try:
            client.delete_job(meeting_id)
        except (OSError, WorkerError, http.client.HTTPException) as exc:
            log_line(config, f"Could not delete failed worker job {meeting_id}: {exc}")
        log_line(config, f"Gave up on {meeting_id} after {job['attempts']} attempts: {message}")
        return
    log_line(config, f"Worker failed {meeting_id} (attempt {job.get('attempts')}): {message}")
    submit(meeting_id, config, fresh=False)


def advance_job(config: BBBConfig, meeting_id: str, submit=None) -> None:
    submit_fn = submit or submit_meeting
    job = load_job(config, meeting_id)
    if job is None or job.get("status") in ("retrieved", "failed"):
        return
    if job.get("status") == "pending-submit":
        submit_fn(meeting_id, config, fresh=False)
        return
    if job.get("status") != "submitted":
        return

    client = WorkerClient(config.worker_url, config.worker_token)
    try:
        status_code, remote = client.get_job(meeting_id)
    except (OSError, WorkerError, http.client.HTTPException) as exc:
        job["error"] = str(exc)[:500]
        save_job(config, job)
        log_line(config, f"Status check failed for {meeting_id}: {exc}")
        return

    if status_code == 404:
        if _already_published(config, meeting_id):
            _mark_retrieved(config, job)
        else:
            job["status"] = "pending-submit"
            job["worker_status"] = None
            job["error"] = "Worker no longer has this job"
            save_job(config, job)
            log_line(config, f"Worker lost {meeting_id}; will submit again")
        return

    remote_status = remote.get("status")
    job["worker_status"] = remote_status
    if remote_status in ("queued", "running"):
        job["error"] = None
        save_job(config, job)
        return
    if remote_status == "ready":
        save_job(config, job)
        _pull_and_publish(config, job, client)
        return
    if remote_status == "failed":
        _retry_or_give_up(config, job, remote.get("error"), client, submit_fn)
        return
    job["error"] = f"Unexpected worker status: {remote_status}"
    save_job(config, job)
    log_line(config, f"Unexpected status for {meeting_id}: {remote_status}")


def retrieve_pending(config: BBBConfig, submit=None) -> None:
    for job in list_jobs(config):
        meeting_id = str(job.get("meeting_id") or "")
        try:
            advance_job(config, meeting_id, submit=submit)
        except Exception as exc:
            log_line(config, f"Retrieve error for {meeting_id}: {exc}")


def main() -> int:
    try:
        config = load_bbb_config()
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    retrieve_pending(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
