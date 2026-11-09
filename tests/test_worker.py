#!/usr/bin/env python3
"""Exercise the worker API and the BigBlueButton retrieve step without Whisper."""

import io
import json
import logging
import py_compile
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bbb_jobs import BBBConfig, load_job, save_job  # noqa: E402
from bbb_retrieve import advance_job  # noqa: E402
from bbb_submit import submit_meeting  # noqa: E402
from worker_client import (  # noqa: E402
    RESULT_TAR_NAMES,
    WorkerClient,
    WorkerError,
    api_request,
    safe_extract_tar,
    write_tar,
)
from worker_server import (  # noqa: E402
    Config,
    ensure_dirs,
    kill_current_processor,
    make_server,
    process_loop,
    read_status,
    recover,
    write_status,
)

logging.basicConfig(level=logging.WARNING)


def write_fake_processor(path: Path, release: Path | None = None, fail_file: Path | None = None, order_file: Path | None = None) -> None:
    path.write_text(
        "\n".join([
            "#!/usr/bin/env python3",
            "import sys, time",
            "from pathlib import Path",
            "args = sys.argv[1:]",
            "work = Path(args[args.index('--work-dir') + 1])",
            "out = Path(args[args.index('--output-dir') + 1])",
            f"release = {str(release) if release else ''!r}",
            f"fail_file = {str(fail_file) if fail_file else ''!r}",
            f"order_file = {str(order_file) if order_file else ''!r}",
            "if release:",
            "    while not Path(release).exists():",
            "        time.sleep(0.05)",
            "if fail_file and Path(fail_file).exists():",
            "    sys.stderr.write('forced failure\\n')",
            "    sys.exit(1)",
            "out.mkdir(parents=True, exist_ok=True)",
            "(out / 'transcript.txt').write_text('hello transcript\\n', encoding='utf-8')",
            "(out / 'transcript.vtt').write_text('WEBVTT\\n\\n', encoding='utf-8')",
            "(out / 'summary.md').write_text('## Executive Summary\\nHello\\n', encoding='utf-8')",
            "if order_file:",
            "    with open(order_file, 'a', encoding='utf-8') as handle:",
            "        handle.write(work.name + '\\n')",
            "",
        ]),
        encoding="utf-8",
    )


def worker_config(data_dir: Path, processor: tuple[str, ...], max_upload: int = 1024 * 1024) -> Config:
    return Config(
        token="test-token",
        bind="127.0.0.1",
        port=0,
        data_dir=data_dir,
        max_upload_bytes=max_upload,
        processor=processor,
        job_timeout_seconds=30,
        lock=threading.Lock(),
    )


class RunningWorker:
    def __init__(self, config: Config):
        self.config = config
        self.stop = threading.Event()
        self.server = make_server(config)
        self.http_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker_thread = threading.Thread(target=process_loop, args=(config, self.stop), daemon=True)
        self.http_thread.start()
        self.worker_thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.client = WorkerClient(self.url, config.token, timeout=30)

    def close(self) -> None:
        self.stop.set()
        kill_current_processor()
        self.server.shutdown()
        self.server.server_close()
        self.worker_thread.join(timeout=5)
        self.http_thread.join(timeout=5)


def make_job_tar(path: Path, meeting_id: str) -> None:
    staging = path.parent / f"files-{meeting_id}"
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "audio.opus").write_bytes(b"not-a-real-opus")
    (staging / "manifest.json").write_text(json.dumps({"meeting_id": meeting_id}), encoding="utf-8")
    (staging / "events.xml").write_text("<events></events>", encoding="utf-8")
    write_tar(path, {
        "audio.opus": staging / "audio.opus",
        "manifest.json": staging / "manifest.json",
        "events.xml": staging / "events.xml",
    })


def wait_status(client: WorkerClient, meeting_id: str, wanted: set[str], timeout: float = 10) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        code, body = client.get_job(meeting_id)
        last = (code, body)
        if code == 200 and body.get("status") in wanted:
            return body
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for {wanted}: {last}")


def bbb_config(root: Path, url: str, max_attempts: int = 3) -> BBBConfig:
    root.mkdir(parents=True, exist_ok=True)
    properties = root / "bbb-web.properties"
    properties.write_text("bigbluebutton.web.serverURL=https://bbb.example\n", encoding="utf-8")
    return BBBConfig(
        worker_url=url,
        worker_token="test-token",
        jobs_dir=root / "jobs",
        published_dir=root / "published",
        raw_dir=root / "raw",
        properties_path=properties,
        max_attempts=max_attempts,
        log_file=root / "retrieve.log",
    )


def write_presentation(published: Path, meeting_id: str, room_name: str = "Faculty Meeting") -> None:
    presentation = published / "presentation" / meeting_id
    presentation.mkdir(parents=True)
    # XML-escape the room name so ElementTree returns the raw characters.
    from xml.sax.saxutils import escape
    xml_name = escape(room_name)
    presentation.joinpath("metadata.xml").write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<recording>
  <id>{meeting_id}</id>
  <state>published</state>
  <published>true</published>
  <start_time>1700000000000</start_time>
  <end_time>1700003600000</end_time>
  <participants>2</participants>
  <meeting>
    <meetingName>{xml_name}</meetingName>
  </meeting>
  <meta>
    <meetingName>{xml_name}</meetingName>
  </meta>
  <playback>
    <format>presentation</format>
    <link>https://bbb.example/playback/presentation/{meeting_id}</link>
    <duration>60</duration>
  </playback>
</recording>
""",
        encoding="utf-8",
    )


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="transcript-worker-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def start_worker(self, *, release: Path | None = None, fail_file: Path | None = None, order_file: Path | None = None, max_upload: int = 1024 * 1024) -> RunningWorker:
        script = self.tmp / "fake_processor.py"
        write_fake_processor(script, release, fail_file, order_file)
        config = worker_config(self.tmp / "data", (sys.executable, str(script)), max_upload=max_upload)
        worker = RunningWorker(config)
        self.addCleanup(worker.close)
        return worker

    def test_sources_parse_and_hook_syntax(self):
        for name in (
            "transcribe_meeting.py",
            "worker_server.py",
            "worker_client.py",
            "bbb_publish.py",
            "bbb_jobs.py",
            "bbb_submit.py",
            "bbb_retrieve.py",
        ):
            py_compile.compile(str(ROOT / name), doraise=True)
        subprocess.run(["bash", "-n", str(ROOT / "01_transcribe_whisper.sh")], check=True)

    def test_health_and_auth(self):
        worker = self.start_worker()
        status, payload = api_request("GET", worker.url + "/v1/health", "nope", timeout=5)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(payload)["status"], "ok")
        tar_path = self.tmp / "job.tar"
        make_job_tar(tar_path, "meeting-1")
        with self.assertRaises(WorkerError) as caught:
            WorkerClient(worker.url, "wrong-token", timeout=5).post_job("meeting-1", tar_path)
        self.assertEqual(caught.exception.status, 401)

    def test_rejects_traversal_and_id_mismatch(self):
        worker = self.start_worker()
        bad = self.tmp / "bad.tar"
        opus = self.tmp / "audio.opus"
        opus.write_bytes(b"audio")
        with tarfile.open(bad, "w") as archive:
            archive.add(opus, arcname="audio.opus")
            info = tarfile.TarInfo("../escape.txt")
            payload = b"nope"
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        with self.assertRaises(WorkerError) as caught:
            worker.client.post_job("meeting-1", bad)
        self.assertEqual(caught.exception.status, 400)
        self.assertFalse((self.tmp / "escape.txt").exists())

        mismatch = self.tmp / "mismatch.tar"
        staging = self.tmp / "mismatch-files"
        staging.mkdir()
        (staging / "audio.opus").write_bytes(b"audio")
        (staging / "manifest.json").write_text(json.dumps({"meeting_id": "other-meeting"}), encoding="utf-8")
        write_tar(mismatch, {"audio.opus": staging / "audio.opus", "manifest.json": staging / "manifest.json"})
        with self.assertRaises(WorkerError) as caught:
            worker.client.post_job("meeting-1", mismatch)
        self.assertEqual(caught.exception.status, 400)

    def test_oversize_upload(self):
        worker = self.start_worker(max_upload=32)
        tar_path = self.tmp / "job.tar"
        make_job_tar(tar_path, "meeting-1")
        self.assertGreater(tar_path.stat().st_size, 32)
        with self.assertRaises(WorkerError) as caught:
            worker.client.post_job("meeting-1", tar_path)
        self.assertEqual(caught.exception.status, 413)

    def test_queue_order_and_running_post_is_idempotent(self):
        release = self.tmp / "release"
        order = self.tmp / "order.txt"
        worker = self.start_worker(release=release, order_file=order)
        first = self.tmp / "a.tar"
        second = self.tmp / "b.tar"
        make_job_tar(first, "job-a")
        make_job_tar(second, "job-b")
        created = worker.client.post_job("job-a", first)
        self.assertTrue(created["created"])
        self.assertEqual(created["status"], "queued")
        running = wait_status(worker.client, "job-a", {"running"})
        self.assertEqual(running["status"], "running")
        again = worker.client.post_job("job-a", first)
        self.assertFalse(again["created"])
        self.assertEqual(again["status"], "running")
        delete_code = worker.client.delete_job("job-a")
        self.assertEqual(delete_code, 409)
        queued = worker.client.post_job("job-b", second)
        self.assertTrue(queued["created"])
        release.write_text("go", encoding="utf-8")
        wait_status(worker.client, "job-a", {"ready"})
        wait_status(worker.client, "job-b", {"ready"})
        self.assertEqual(order.read_text(encoding="utf-8").splitlines(), ["job-a", "job-b"])
        result = self.tmp / "result.tar"
        worker.client.download_result("job-a", result)
        extracted = self.tmp / "result"
        safe_extract_tar(result, extracted, set(RESULT_TAR_NAMES), required=set(RESULT_TAR_NAMES))
        self.assertEqual((extracted / "transcript.txt").read_text(encoding="utf-8"), "hello transcript\n")

    def test_failed_job_can_be_replaced(self):
        fail_file = self.tmp / "fail"
        fail_file.write_text("1", encoding="utf-8")
        worker = self.start_worker(fail_file=fail_file)
        tar_path = self.tmp / "job.tar"
        make_job_tar(tar_path, "meeting-1")
        worker.client.post_job("meeting-1", tar_path)
        failed = wait_status(worker.client, "meeting-1", {"failed"})
        self.assertIn("processor exited", failed["error"])
        fail_file.unlink()
        replaced = worker.client.post_job("meeting-1", tar_path)
        self.assertTrue(replaced["created"])
        self.assertEqual(replaced["status"], "queued")
        ready = wait_status(worker.client, "meeting-1", {"ready"})
        self.assertIsNone(ready["error"])

    def test_restart_requeues_running_job(self):
        config = worker_config(self.tmp / "data", (sys.executable, "unused"))
        ensure_dirs(config)
        write_status(config, {
            "meeting_id": "meeting-1",
            "status": "running",
            "error": None,
            "submitted_at": "2026-01-01T00:00:00+00:00",
            "started_at": "2026-01-01T00:00:01+00:00",
            "finished_at": None,
        })
        inbox = config.data_dir / "inbox" / "meeting-1"
        inbox.mkdir(parents=True)
        (inbox / "audio.opus").write_bytes(b"audio")
        recover(config)
        status = read_status(config, "meeting-1")
        self.assertEqual(status["status"], "queued")
        self.assertIsNone(status["started_at"])

    def test_publish_retrieve_and_remote_delete(self):
        worker = self.start_worker()
        meeting_id = "meeting-1"
        tar_path = self.tmp / "job.tar"
        make_job_tar(tar_path, meeting_id)
        worker.client.post_job(meeting_id, tar_path)
        wait_status(worker.client, meeting_id, {"ready"})

        room_name = "Tom & <Jerry>"
        config = bbb_config(self.tmp / "bbb", worker.url)
        write_presentation(config.published_dir, meeting_id, room_name)
        save_job(config, {
            "meeting_id": meeting_id,
            "status": "submitted",
            "worker_status": "queued",
            "error": None,
            "attempts": 1,
            "submitted_at": "2026-01-01T00:00:00+00:00",
            "retrieved_at": None,
            "updated_at": "2026-01-01T00:00:00+00:00",
        })
        advance_job(config, meeting_id)
        job = load_job(config, meeting_id)
        self.assertEqual(job["status"], "retrieved")
        presentation = config.published_dir / "presentation" / meeting_id
        html = (presentation / "summary" / "summary.html").read_text(encoding="utf-8")
        self.assertIn("Tom &amp; &lt;Jerry&gt;", html)
        self.assertNotIn("<Jerry>", html)
        self.assertIn("Hello", html)
        self.assertTrue((presentation / "transcript" / "transcript.txt").is_file())
        self.assertTrue((presentation / "transcript" / "transcript.vtt").is_file())
        rtfs = list((presentation / "summary").glob("Summary_*.rtf"))
        self.assertEqual(len(rtfs), 1)
        self.assertIn("Tom_&_Jerry", rtfs[0].name)
        for fmt in ("summary", "transcript"):
            meta = (config.published_dir / fmt / meeting_id / "metadata.xml").read_text(encoding="utf-8")
            self.assertIn(f"https://bbb.example/presentation/{meeting_id}/{fmt}/", meta)
            self.assertIn("Tom", meta)
        self.assertEqual(worker.client.get_job(meeting_id)[0], 404)

    def test_worker_failure_retries_then_gives_up(self):
        fail_file = self.tmp / "fail"
        fail_file.write_text("1", encoding="utf-8")
        worker = self.start_worker(fail_file=fail_file)
        meeting_id = "meeting-1"
        tar_path = self.tmp / "job.tar"
        make_job_tar(tar_path, meeting_id)
        worker.client.post_job(meeting_id, tar_path)
        wait_status(worker.client, meeting_id, {"failed"})

        config = bbb_config(self.tmp / "bbb", worker.url, max_attempts=3)
        save_job(config, {
            "meeting_id": meeting_id,
            "status": "submitted",
            "worker_status": "running",
            "error": None,
            "attempts": 1,
            "submitted_at": "2026-01-01T00:00:00+00:00",
            "retrieved_at": None,
            "updated_at": "2026-01-01T00:00:00+00:00",
        })
        calls = []

        def fake_submit(mid, cfg, fresh=False):
            calls.append((mid, fresh))
            job = load_job(cfg, mid)
            job["attempts"] = int(job["attempts"]) + 1
            job["status"] = "submitted"
            save_job(cfg, job)

        advance_job(config, meeting_id, submit=fake_submit)
        self.assertEqual(calls, [(meeting_id, False)])
        self.assertEqual(load_job(config, meeting_id)["attempts"], 2)
        self.assertEqual(worker.client.get_job(meeting_id)[0], 200)

        config.max_attempts = 1
        job = load_job(config, meeting_id)
        job["attempts"] = 1
        job["status"] = "submitted"
        save_job(config, job)
        advance_job(config, meeting_id, submit=fake_submit)
        self.assertEqual(load_job(config, meeting_id)["status"], "failed")
        self.assertEqual(len(calls), 1)
        self.assertEqual(worker.client.get_job(meeting_id)[0], 404)

    def test_missing_audio_is_permanent_and_network_error_retries(self):
        config = bbb_config(self.tmp / "bbb", "http://127.0.0.1:9")
        job = submit_meeting("meeting-1", config, fresh=True)
        self.assertEqual(job["status"], "failed")
        self.assertIn("webcams.webm", job["error"])

        with mock.patch("bbb_submit.build_payload_tar"), mock.patch("bbb_submit.WorkerClient") as client_cls:
            client_cls.return_value.post_job.side_effect = ConnectionRefusedError("down")
            retried = submit_meeting("meeting-2", config, fresh=True)
        self.assertEqual(retried["status"], "pending-submit")

        from worker_client import WorkerError
        with mock.patch("bbb_submit.build_payload_tar"), mock.patch("bbb_submit.WorkerClient") as client_cls:
            client_cls.return_value.post_job.side_effect = WorkerError("bad tar", status=400)
            rejected = submit_meeting("meeting-3", config, fresh=True)
        self.assertEqual(rejected["status"], "failed")

    def test_retrieve_retries_pending_submit(self):
        config = bbb_config(self.tmp / "bbb", "http://127.0.0.1:9")
        save_job(config, {
            "meeting_id": "meeting-9",
            "status": "pending-submit",
            "worker_status": None,
            "error": "down",
            "attempts": 0,
            "submitted_at": None,
            "retrieved_at": None,
            "updated_at": "2026-01-01T00:00:00+00:00",
        })
        calls = []

        def fake_submit(meeting_id, cfg, fresh=False):
            calls.append((meeting_id, fresh))
            job = load_job(cfg, meeting_id)
            job["status"] = "submitted"
            save_job(cfg, job)

        advance_job(config, "meeting-9", submit=fake_submit)
        self.assertEqual(calls, [("meeting-9", False)])
        self.assertEqual(load_job(config, "meeting-9")["status"], "submitted")


if __name__ == "__main__":
    unittest.main()
