#!/usr/bin/env python3
"""One-at-a-time transcript worker.

Accepts a tar upload, runs transcribe_meeting.py, and serves the result tar.
Bind this to localhost and put HTTPS plus the BigBlueButton host allow-list
in front of it.
"""

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from worker_client import (
    JOB_TAR_NAMES,
    RESULT_TAR_NAMES,
    safe_extract_tar,
    valid_meeting_id,
    write_tar,
)

RESULT_FILES = ("transcript.txt", "transcript.vtt", "summary.md")
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "LLM_NUM_THREADS",
    "WHISPER_CPU_THREADS",
)

_current_proc_lock = threading.Lock()
_current_proc: subprocess.Popen | None = None


@dataclass
class Config:
    token: str
    bind: str
    port: int
    data_dir: Path
    max_upload_bytes: int
    processor: tuple[str, ...]
    job_timeout_seconds: int
    lock: threading.Lock


def iso_now() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


def load_config() -> Config:
    token = os.environ.get("TRANSCRIPT_WORKER_TOKEN", "").strip()
    if not token:
        raise SystemExit("TRANSCRIPT_WORKER_TOKEN is required")
    try:
        port = int(os.environ.get("TRANSCRIPT_PORT", "8787"))
        max_upload = int(os.environ.get("TRANSCRIPT_MAX_UPLOAD_BYTES", str(1024 * 1024 * 1024)))
        timeout = int(os.environ.get("TRANSCRIPT_JOB_TIMEOUT_SECONDS", "28800"))
    except ValueError as exc:
        raise SystemExit(f"Invalid numeric setting: {exc}") from exc
    script = Path(__file__).resolve().parent / "transcribe_meeting.py"
    processor_env = os.environ.get("TRANSCRIPT_PROCESSOR", "").strip()
    if processor_env:
        processor = tuple(processor_env.split())
    else:
        processor = (sys.executable, str(script))
    return Config(
        token=token,
        bind=os.environ.get("TRANSCRIPT_BIND", "127.0.0.1"),
        port=port,
        data_dir=Path(os.environ.get("TRANSCRIPT_DATA_DIR", "/var/transcript-worker")),
        max_upload_bytes=max_upload,
        processor=processor,
        job_timeout_seconds=timeout,
        lock=threading.Lock(),
    )


def _dir(config: Config, name: str) -> Path:
    path = config.data_dir / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs(config: Config) -> None:
    for name in ("inbox", "results", "status", "logs", "tmp", "models"):
        _dir(config, name)


def status_path(config: Config, meeting_id: str) -> Path:
    return config.data_dir / "status" / f"{meeting_id}.json"


def read_status(config: Config, meeting_id: str) -> dict | None:
    path = status_path(config, meeting_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def write_status(config: Config, status: dict) -> None:
    path = status_path(config, status["meeting_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def blank_status(meeting_id: str) -> dict:
    return {
        "meeting_id": meeting_id,
        "status": "queued",
        "error": None,
        "submitted_at": iso_now(),
        "started_at": None,
        "finished_at": None,
    }


def iter_statuses(config: Config) -> list[dict]:
    status_dir = config.data_dir / "status"
    if not status_dir.is_dir():
        return []
    found = []
    for path in status_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and valid_meeting_id(str(data.get("meeting_id", ""))):
            found.append(data)
    return found


def recover(config: Config) -> None:
    """Requeue jobs that were running when the process stopped, and adopt orphan inboxes."""
    known = set()
    with config.lock:
        for status in iter_statuses(config):
            known.add(status["meeting_id"])
            if status.get("status") == "running":
                status["status"] = "queued"
                status["started_at"] = None
                status["error"] = None
                write_status(config, status)
                logging.info("Requeued %s after restart", status["meeting_id"])
        inbox = config.data_dir / "inbox"
        if inbox.is_dir():
            for entry in inbox.iterdir():
                if entry.name in known or not valid_meeting_id(entry.name):
                    continue
                if (entry / "audio.opus").is_file():
                    write_status(config, blank_status(entry.name))
                    logging.info("Adopted inbox %s", entry.name)


def public_status(status: dict, created: bool | None = None) -> dict:
    body = {
        "meeting_id": status.get("meeting_id"),
        "status": status.get("status"),
        "error": status.get("error"),
        "submitted_at": status.get("submitted_at"),
        "started_at": status.get("started_at"),
        "finished_at": status.get("finished_at"),
    }
    if created is not None:
        body["created"] = created
    return body


def _results_ready(config: Config, meeting_id: str) -> bool:
    result_dir = config.data_dir / "results" / meeting_id
    for name in RESULT_FILES:
        path = result_dir / name
        if not path.is_file() or path.stat().st_size <= 0:
            return False
    return True


def accept_upload(config: Config, meeting_id: str, tar_path: Path) -> tuple[int, dict]:
    """Store a job tar. A running or ready job is left untouched."""
    staging = config.data_dir / "tmp" / f"stage-{uuid.uuid4().hex}"
    try:
        safe_extract_tar(tar_path, staging, set(JOB_TAR_NAMES), required={"audio.opus"})
        opus = staging / "audio.opus"
        if not opus.is_file() or opus.stat().st_size <= 0:
            raise ValueError("audio.opus is empty")
        manifest_path = staging / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError("manifest.json is not valid JSON") from exc
            if isinstance(manifest, dict) and manifest.get("meeting_id") not in (None, meeting_id):
                raise ValueError("manifest meeting_id does not match X-Meeting-Id")
    except ValueError:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    with config.lock:
        current = read_status(config, meeting_id)
        if current and current.get("status") in ("running", "ready"):
            shutil.rmtree(staging, ignore_errors=True)
            return 200, public_status(current, created=False)

        created = current is None or current.get("status") == "failed"
        inbox = config.data_dir / "inbox" / meeting_id
        if inbox.exists():
            shutil.rmtree(inbox)
        staging.rename(inbox)
        results = config.data_dir / "results" / meeting_id
        if results.exists():
            shutil.rmtree(results)
        log_path = config.data_dir / "logs" / f"{meeting_id}.log"
        if log_path.exists():
            log_path.unlink()

        if current and current.get("status") == "queued":
            current["error"] = None
            write_status(config, current)
            logging.info("Replaced queued upload for %s", meeting_id)
            return 200, public_status(current, created=False)

        status = blank_status(meeting_id)
        write_status(config, status)
        logging.info("Queued %s", meeting_id)
        return 202, public_status(status, created=created)


def claim_next(config: Config) -> str | None:
    with config.lock:
        queued = [item for item in iter_statuses(config) if item.get("status") == "queued"]
        if not queued:
            return None
        queued.sort(key=lambda item: item.get("submitted_at") or "")
        chosen = queued[0]
        chosen["status"] = "running"
        chosen["started_at"] = iso_now()
        chosen["error"] = None
        chosen["finished_at"] = None
        write_status(config, chosen)
        return chosen["meeting_id"]


def requeue(config: Config, meeting_id: str) -> None:
    with config.lock:
        status = read_status(config, meeting_id)
        if not status or status.get("status") != "running":
            return
        status["status"] = "queued"
        status["started_at"] = None
        write_status(config, status)


def finish_job(config: Config, meeting_id: str, ok: bool, error: str | None) -> None:
    with config.lock:
        status = read_status(config, meeting_id)
        if not status:
            return
        if ok and _results_ready(config, meeting_id):
            status["status"] = "ready"
            status["error"] = None
        else:
            status["status"] = "failed"
            status["error"] = (error or "Processing failed")[:500]
        status["finished_at"] = iso_now()
        write_status(config, status)
        logging.info("Finished %s status=%s", meeting_id, status["status"])


def resolve_threads() -> int:
    raw = os.environ.get("CPU_THREAD_LIMIT")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return max(1, (os.cpu_count() or 2) - 2)


def processor_env(config: Config) -> dict[str, str]:
    env = os.environ.copy()
    threads = str(resolve_threads())
    env["CPU_THREAD_LIMIT"] = os.environ.get("CPU_THREAD_LIMIT", threads)
    for name in THREAD_ENV_VARS:
        env.setdefault(name, threads)
    env.setdefault("DEBUG_SUMMARY", "0")
    env.setdefault("HF_HOME", str(config.data_dir / "models"))
    env.setdefault("LLM_MODEL_FAMILY", "gemma-4")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def kill_current_processor() -> None:
    global _current_proc
    with _current_proc_lock:
        proc = _current_proc
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        proc.wait(timeout=10)


def run_processor(config: Config, meeting_id: str, stop: threading.Event) -> tuple[bool, str] | None:
    """Run one job. None means the server is stopping and the job was not judged."""
    global _current_proc
    if stop.is_set():
        return None
    inbox = config.data_dir / "inbox" / meeting_id
    results = config.data_dir / "results" / meeting_id
    if results.exists():
        shutil.rmtree(results)
    results.mkdir(parents=True, exist_ok=True)
    if not (inbox / "audio.opus").is_file():
        return False, "audio.opus is missing from the job inbox"

    log_path = config.data_dir / "logs" / f"{meeting_id}.log"
    command = list(config.processor) + ["--work-dir", str(inbox), "--output-dir", str(results)]
    logging.info("Starting processor for %s", meeting_id)
    with log_path.open("ab") as log_handle:
        log_handle.write(f"\n--- {iso_now()} {meeting_id} ---\n".encode())
        proc = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=processor_env(config),
            start_new_session=True,
        )
        with _current_proc_lock:
            _current_proc = proc
        try:
            timeout = config.job_timeout_seconds or None
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logging.error("Processor timed out for %s", meeting_id)
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=10)
            return False, "Processing timed out"
        finally:
            with _current_proc_lock:
                if _current_proc is proc:
                    _current_proc = None

    if stop.is_set():
        return None
    if returncode != 0:
        return False, f"processor exited with status {returncode}"
    if not _results_ready(config, meeting_id):
        return False, "processor exited without the transcript and summary files"
    return True, ""


def process_loop(config: Config, stop: threading.Event) -> None:
    while not stop.is_set():
        meeting_id = claim_next(config)
        if meeting_id is None:
            stop.wait(1.0)
            continue
        try:
            outcome = run_processor(config, meeting_id, stop)
        except Exception as exc:
            logging.exception("Processor crashed for %s", meeting_id)
            outcome = (False, f"processor crashed: {exc}")
        if outcome is None:
            requeue(config, meeting_id)
            continue
        ok, error = outcome
        finish_job(config, meeting_id, ok, error)


def delete_job(config: Config, meeting_id: str) -> int:
    with config.lock:
        status = read_status(config, meeting_id)
        if status is None:
            return 404
        if status.get("status") in ("queued", "running"):
            return 409
        status_path(config, meeting_id).unlink(missing_ok=True)
        for relative in (
            config.data_dir / "inbox" / meeting_id,
            config.data_dir / "results" / meeting_id,
        ):
            if relative.exists():
                shutil.rmtree(relative, ignore_errors=True)
        log_path = config.data_dir / "logs" / f"{meeting_id}.log"
        if log_path.exists():
            log_path.unlink()
    logging.info("Deleted %s", meeting_id)
    return 204


def build_result_tar(config: Config, meeting_id: str, dest: Path) -> None:
    result_dir = config.data_dir / "results" / meeting_id
    files = {name: result_dir / name for name in RESULT_FILES}
    write_tar(dest, files)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "transcript-worker"

    def log_message(self, fmt: str, *args) -> None:
        logging.info("%s - " + fmt, self.address_string(), *args)

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, code: int, payload: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    def _authorized(self) -> bool:
        config: Config = self.server.config  # type: ignore[attr-defined]
        header = self.headers.get("Authorization", "")
        presented = header[7:] if header.startswith("Bearer ") else ""
        expected = config.token
        if not presented or len(presented) != len(expected):
            return False
        import hmac
        return hmac.compare_digest(presented.encode(), expected.encode())

    def _parts(self) -> list[str]:
        return [part for part in urlparse(self.path).path.split("/") if part]

    def _read_body(self, config: Config) -> Path | None:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self._json(411, {"error": "Content-Length is required"})
            return None
        try:
            length = int(raw_length)
        except ValueError:
            self._json(400, {"error": "Content-Length is not an integer"})
            return None
        if length <= 0 or length > config.max_upload_bytes:
            self._json(413, {"error": "Upload is empty or larger than TRANSCRIPT_MAX_UPLOAD_BYTES"})
            return None
        temporary = config.data_dir / "tmp" / f"upload-{uuid.uuid4().hex}.tar"
        remaining = length
        try:
            with temporary.open("wb") as handle:
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    handle.write(chunk)
                    remaining -= len(chunk)
        except OSError:
            temporary.unlink(missing_ok=True)
            self._json(500, {"error": "Could not store the upload"})
            return None
        if remaining:
            temporary.unlink(missing_ok=True)
            self._json(400, {"error": "Upload ended early"})
            return None
        return temporary

    def do_GET(self) -> None:  # noqa: N802
        parts = self._parts()
        if parts == ["v1", "health"]:
            self._json(200, {"status": "ok"})
            return
        if not self._authorized():
            self._json(401, {"error": "Unauthorized"})
            return
        config: Config = self.server.config  # type: ignore[attr-defined]
        if len(parts) == 3 and parts[0] == "v1" and parts[1] == "jobs":
            meeting_id = parts[2]
            if not valid_meeting_id(meeting_id):
                self._json(400, {"error": "Invalid meeting id"})
                return
            status = read_status(config, meeting_id)
            if status is None:
                self._json(404, {"error": "Job not found"})
                return
            self._json(200, public_status(status))
            return
        if len(parts) == 4 and parts[0] == "v1" and parts[1] == "jobs" and parts[3] == "result":
            meeting_id = parts[2]
            if not valid_meeting_id(meeting_id):
                self._json(400, {"error": "Invalid meeting id"})
                return
            status = read_status(config, meeting_id)
            if status is None:
                self._json(404, {"error": "Job not found"})
                return
            if status.get("status") != "ready" or not _results_ready(config, meeting_id):
                self._json(409, {"error": "Result is not ready"})
                return
            dest = config.data_dir / "tmp" / f"result-{uuid.uuid4().hex}.tar"
            try:
                build_result_tar(config, meeting_id, dest)
                payload = dest.read_bytes()
            finally:
                dest.unlink(missing_ok=True)
            self._bytes(200, payload, "application/x-tar")
            return
        self._json(404, {"error": "Not found"})

    def _discard_body(self) -> None:
        raw_length = self.headers.get("Content-Length")
        try:
            remaining = int(raw_length or "0")
        except ValueError:
            return
        while remaining > 0:
            chunk = self.rfile.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            remaining -= len(chunk)

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            self._discard_body()
            self._json(401, {"error": "Unauthorized"})
            return
        parts = self._parts()
        if parts != ["v1", "jobs"]:
            self._discard_body()
            self._json(404, {"error": "Not found"})
            return
        meeting_id = self.headers.get("X-Meeting-Id", "").strip()
        if not valid_meeting_id(meeting_id):
            self._discard_body()
            self._json(400, {"error": "X-Meeting-Id is missing or invalid"})
            return
        config: Config = self.server.config  # type: ignore[attr-defined]
        tar_path = self._read_body(config)
        if tar_path is None:
            return
        try:
            code, body = accept_upload(config, meeting_id, tar_path)
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return
        finally:
            tar_path.unlink(missing_ok=True)
        self._json(code, body)

    def do_DELETE(self) -> None:  # noqa: N802
        if not self._authorized():
            self._json(401, {"error": "Unauthorized"})
            return
        parts = self._parts()
        if len(parts) != 3 or parts[0] != "v1" or parts[1] != "jobs":
            self._json(404, {"error": "Not found"})
            return
        meeting_id = parts[2]
        if not valid_meeting_id(meeting_id):
            self._json(400, {"error": "Invalid meeting id"})
            return
        config: Config = self.server.config  # type: ignore[attr-defined]
        code = delete_job(config, meeting_id)
        if code == 404:
            self._json(404, {"error": "Job not found"})
            return
        if code == 409:
            self._json(409, {"error": "Job is still queued or running"})
            return
        self._bytes(204, b"", "application/json")


class TranscriptServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    config: Config


def make_server(config: Config) -> TranscriptServer:
    ensure_dirs(config)
    recover(config)
    server = TranscriptServer((config.bind, config.port), Handler)
    server.config = config
    return server


def serve(config: Config) -> None:
    stop = threading.Event()
    server = make_server(config)
    worker = threading.Thread(target=process_loop, args=(config, stop), name="transcript-queue", daemon=True)
    worker.start()

    def handle_signal(signum, _frame) -> None:
        logging.info("Received signal %s", signum)
        stop.set()
        kill_current_processor()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    logging.info("Transcript worker listening on %s:%s", config.bind, server.server_address[1])
    try:
        server.serve_forever()
    finally:
        stop.set()
        kill_current_processor()
        server.server_close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    serve(load_config())


if __name__ == "__main__":
    main()
