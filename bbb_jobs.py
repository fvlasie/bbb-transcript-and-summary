#!/usr/bin/env python3
"""Local job records on the BigBlueButton host."""

import datetime
import json
import os
import urllib.parse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BBBConfig:
    worker_url: str
    worker_token: str
    jobs_dir: Path
    published_dir: Path
    raw_dir: Path
    properties_path: Path
    max_attempts: int = 3
    log_file: Path | None = None


def load_bbb_config() -> BBBConfig:
    url = os.environ.get("TRANSCRIPT_WORKER_URL", "").strip().rstrip("/")
    token = os.environ.get("TRANSCRIPT_WORKER_TOKEN", "").strip()
    if not url or not token:
        raise ValueError("TRANSCRIPT_WORKER_URL and TRANSCRIPT_WORKER_TOKEN are required")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("https", "http") or not parsed.hostname:
        raise ValueError("TRANSCRIPT_WORKER_URL must be an http(s) URL")
    try:
        max_attempts = max(1, int(os.environ.get("TRANSCRIPT_MAX_ATTEMPTS", "3")))
    except ValueError as exc:
        raise ValueError("TRANSCRIPT_MAX_ATTEMPTS must be an integer") from exc
    return BBBConfig(
        worker_url=url,
        worker_token=token,
        jobs_dir=Path(os.environ.get("TRANSCRIPT_JOBS_DIR", "/var/bigbluebutton/transcript-jobs")),
        published_dir=Path(os.environ.get("BBB_PUBLISHED_DIR", "/var/bigbluebutton/published")),
        raw_dir=Path(os.environ.get("BBB_RAW_DIR", "/var/bigbluebutton/recording/raw")),
        properties_path=Path(os.environ.get("BBB_WEB_PROPERTIES", "/etc/bigbluebutton/bbb-web.properties")),
        max_attempts=max_attempts,
        log_file=Path(os.environ.get("TRANSCRIPT_LOG_FILE", "/var/log/bigbluebutton/post_publish_transcribe.log")),
    )


def now_stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


def log_line(config: BBBConfig, message: str) -> None:
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    if config.log_file is None:
        return
    try:
        config.log_file.parent.mkdir(parents=True, exist_ok=True)
        with config.log_file.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError as exc:
        print(f"[{stamp}] Could not write {config.log_file}: {exc}", flush=True)


def job_path(config: BBBConfig, meeting_id: str) -> Path:
    return config.jobs_dir / f"{meeting_id}.json"


def blank_job(meeting_id: str) -> dict:
    return {
        "meeting_id": meeting_id,
        "status": "pending-submit",
        "worker_status": None,
        "error": None,
        "attempts": 0,
        "submitted_at": None,
        "retrieved_at": None,
        "updated_at": now_stamp(),
    }


def load_job(config: BBBConfig, meeting_id: str) -> dict | None:
    path = job_path(config, meeting_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def save_job(config: BBBConfig, job: dict) -> None:
    config.jobs_dir.mkdir(parents=True, exist_ok=True)
    job["updated_at"] = now_stamp()
    path = job_path(config, job["meeting_id"])
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def list_jobs(config: BBBConfig) -> list[dict]:
    if not config.jobs_dir.is_dir():
        return []
    jobs = []
    for path in sorted(config.jobs_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("meeting_id"):
            jobs.append(data)
    return jobs
