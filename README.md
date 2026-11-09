# BigBlueButton 3.0 Transcription & AI Summarization

Speaker-attributed transcripts and AI summaries for BigBlueButton 3.0 recordings. Whisper and the local Gemma model run on a separate CPU worker. The BigBlueButton host only sends the recording and, when the job is ready, writes the files back into the published presentation.

Playback links stay on the BigBlueButton host:

- `https://<bbb-host>/presentation/<meeting_id>/summary/summary.html`
- `https://<bbb-host>/presentation/<meeting_id>/transcript/transcript.txt`

The presentation recording is available as soon as BigBlueButton finishes publishing. Summary and transcript links show up in Greenlight after the retrieve step writes `metadata.xml`.

## Flow

```text
[BBB recording published]
        │
        ▼
[01_transcribe_whisper.sh]
        │  extract mono Opus, send events.xml and metadata
        ▼
[Worker POST /v1/jobs]  ──►  one meeting at a time
        │                     Whisper large-v3-turbo (CPU, int8)
        │                     then local Gemma summary
        ▼
[bbb-transcript-retrieve.timer]
        │  download transcript.txt, transcript.vtt, summary.md
        │  render summary.html and RTF
        │  register summary and transcript playback formats
        ▼
[Delete the copy on the worker]
```

The post-publish hook returns as soon as the upload is accepted. It does not wait for transcription.

## Repository layout

| File | Where it runs | Role |
| :--- | :--- | :--- |
| `01_transcribe_whisper.sh` | BigBlueButton | Post-publish hook. Submits the job and exits. |
| `bbb_submit.py` | BigBlueButton | Extracts Opus audio and uploads the job tar. |
| `bbb_retrieve.py` | BigBlueButton | Polls the worker and publishes finished artifacts. |
| `bbb_publish.py` | BigBlueButton | HTML, RTF, and playback `metadata.xml`. Stdlib only. |
| `bbb_jobs.py` | BigBlueButton | Local job records under `transcript-jobs/`. |
| `worker_client.py` | Both | HTTPS client and safe tar extract. |
| `worker_server.py` | Worker | Queue and HTTP API. One job at a time. |
| `transcribe_meeting.py` | Worker | Whisper, speaker alignment, Gemma summary. |
| `deploy/` | Both | systemd units, nginx site, env examples. |

## Worker API

The worker listens on localhost. Put HTTPS and a firewall allow-list for the BigBlueButton host in front of it.

| Request | Meaning |
| :--- | :--- |
| `POST /v1/jobs` | Tar of `audio.opus`, optional `events.xml`, optional `manifest.json`. Header `X-Meeting-Id`. |
| `GET /v1/jobs/<id>` | `queued`, `running`, `ready`, or `failed`. |
| `GET /v1/jobs/<id>/result` | Tar of `transcript.txt`, `transcript.vtt`, and `summary.md`. |
| `DELETE /v1/jobs/<id>` | Remove the audio and results after the BigBlueButton host has them. |
| `GET /v1/health` | Liveness, no token. |

A second upload of a queued, running, or ready meeting returns the current job. A failed meeting can be submitted again. The bearer token is required on every route except health.

## Job records on BigBlueButton

`/var/bigbluebutton/transcript-jobs/<meeting_id>.json`

| Status | Meaning |
| :--- | :--- |
| `pending-submit` | The upload did not reach the worker. The timer retries. |
| `submitted` | The worker has the job. |
| `retrieved` | Files are published and the worker copy was deleted. |
| `failed` | Missing audio, a rejected upload, or the attempt limit was reached. |

Publishing the same recording again (`fresh` submit from the hook) opens a failed job and runs it once more.

## Manual run on the worker

```bash
python3 transcribe_meeting.py --work-dir /path/to/job --output-dir /path/to/out
```

The work directory needs `audio.opus` (or `webcams.webm` / `audio.webm`) and may include `events.xml`.

## Tests

```bash
python3 tests/test_worker.py -v
```

The tests use a stand-in processor. They do not load Whisper or Gemma.

## Install

Host setup, the nginx allow-list, and the systemd units are in [`install.md`](./install.md).
