#!/usr/bin/env python3
"""HTTP client and tar helpers shared by the worker and the BigBlueButton host."""

import http.client
import json
import os
import re
import tarfile
import urllib.parse
from pathlib import Path

MEETING_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
JOB_TAR_NAMES = {"audio.opus", "events.xml", "manifest.json"}
RESULT_TAR_NAMES = {"transcript.txt", "transcript.vtt", "summary.md"}


class WorkerError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def valid_meeting_id(value: str | None) -> bool:
    return bool(value) and bool(MEETING_ID_RE.fullmatch(value))


def write_tar(path: Path, files: dict[str, Path]) -> None:
    """Write an uncompressed tar whose member names are the dict keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as archive:
        for arcname, source in files.items():
            if not valid_member_name(arcname):
                raise ValueError(f"Refusing to archive {arcname}")
            archive.add(source, arcname=arcname, recursive=False)


def valid_member_name(name: str) -> bool:
    return name == os.path.basename(name) and name not in ("", ".", "..")


def safe_extract_tar(
    tar_path: Path,
    dest: Path,
    allowed: set[str],
    required: set[str] | None = None,
) -> None:
    """Extract only the named regular files. Reject links and path members."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:") as archive:
        members = archive.getmembers()
        if len(members) > len(allowed) + 2:
            raise ValueError("Tar archive has too many members")
        seen: set[str] = set()
        for member in members:
            name = member.name
            if name in ("", ".", "./"):
                continue
            if not valid_member_name(name) or name not in allowed:
                raise ValueError(f"Unexpected tar member: {name}")
            if member.issym() or member.islnk() or not member.isfile():
                raise ValueError(f"Tar member is not a regular file: {name}")
            if member.size < 0 or member.size > 1024 * 1024 * 1024:
                raise ValueError(f"Tar member size is not allowed: {name}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Missing tar payload: {name}")
            target = dest / name
            with target.open("wb") as handle:
                remaining = member.size
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    handle.write(chunk)
                    remaining -= len(chunk)
            if target.stat().st_size != member.size:
                raise ValueError(f"Short read for {name}")
            seen.add(name)
    missing = (required or set()) - seen
    if missing:
        raise ValueError("Tar archive is missing " + ", ".join(sorted(missing)))


def api_request(
    method: str,
    url: str,
    token: str,
    *,
    body_path: Path | None = None,
    timeout: int = 600,
    extra_headers: dict[str, str] | None = None,
) -> tuple[int, bytes]:
    """Perform one request. A file body is streamed with Content-Length set."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("https", "http") or not parsed.hostname:
        raise WorkerError(f"Unsupported worker URL: {url}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if parsed.scheme == "https":
        connection = http.client.HTTPSConnection(parsed.hostname, port, timeout=timeout)
    else:
        connection = http.client.HTTPConnection(parsed.hostname, port, timeout=timeout)

    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    headers = {"Authorization": f"Bearer {token}", "Connection": "close"}
    if extra_headers:
        headers.update(extra_headers)

    body = None
    try:
        if body_path is not None:
            headers["Content-Length"] = str(body_path.stat().st_size)
            headers.setdefault("Content-Type", "application/x-tar")
            body = body_path.open("rb")
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        payload = response.read()
        return response.status, payload
    finally:
        if body is not None:
            body.close()
        connection.close()


def _decode_json(status: int, payload: bytes, ok: tuple[int, ...]) -> dict:
    try:
        parsed = json.loads(payload.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerError(f"Worker returned a non-JSON response ({status})", status) from exc
    if not isinstance(parsed, dict):
        raise WorkerError(f"Worker returned an unexpected response ({status})", status)
    if status not in ok:
        message = str(parsed.get("error") or f"Worker request failed ({status})")
        raise WorkerError(message, status)
    return parsed


class WorkerClient:
    def __init__(self, base_url: str, token: str, timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _url(self, suffix: str) -> str:
        return self.base_url + suffix

    def post_job(self, meeting_id: str, tar_path: Path) -> dict:
        status, payload = api_request(
            "POST",
            self._url("/v1/jobs"),
            self.token,
            body_path=tar_path,
            timeout=self.timeout,
            extra_headers={"X-Meeting-Id": meeting_id, "Content-Type": "application/x-tar"},
        )
        return _decode_json(status, payload, ok=(200, 202))

    def get_job(self, meeting_id: str) -> tuple[int, dict]:
        status, payload = api_request(
            "GET",
            self._url(f"/v1/jobs/{meeting_id}"),
            self.token,
            timeout=min(self.timeout, 60),
        )
        if status == 404:
            return 404, {}
        return status, _decode_json(status, payload, ok=(200,))

    def download_result(self, meeting_id: str, dest_tar: Path) -> None:
        status, payload = api_request(
            "GET",
            self._url(f"/v1/jobs/{meeting_id}/result"),
            self.token,
            timeout=self.timeout,
        )
        if status != 200:
            message = payload.decode("utf-8", errors="replace")[:300]
            try:
                parsed = json.loads(message or "{}")
                if isinstance(parsed, dict) and parsed.get("error"):
                    message = str(parsed["error"])
            except json.JSONDecodeError:
                pass
            raise WorkerError(message or f"Result download failed ({status})", status)
        dest_tar.parent.mkdir(parents=True, exist_ok=True)
        dest_tar.write_bytes(payload)

    def delete_job(self, meeting_id: str) -> int:
        status, payload = api_request(
            "DELETE",
            self._url(f"/v1/jobs/{meeting_id}"),
            self.token,
            timeout=min(self.timeout, 60),
        )
        if status not in (204, 404, 409):
            message = payload.decode("utf-8", errors="replace")[:300]
            raise WorkerError(message or f"Delete failed ({status})", status)
        return status
