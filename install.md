# Install

One BigBlueButton server submits recordings. One CPU worker transcribes and summarizes them, one meeting at a time. Whisper stays on CPU (`large-v3-turbo`, int8). Gemma stays a local GGUF.

Use the same bearer token on both hosts. Generate it once:

```bash
openssl rand -hex 32
```

## 1. Worker host

Install build tools. `llama-cpp-python` compiles on first install.

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip build-essential cmake

sudo useradd --system --home /var/transcript-worker --shell /usr/sbin/nologin transcript || true
sudo mkdir -p /var/transcript-worker /opt/transcript-worker
sudo chown transcript:transcript /var/transcript-worker

sudo python3 -m venv /opt/transcript-worker/venv
sudo /opt/transcript-worker/venv/bin/pip install --upgrade pip
sudo /opt/transcript-worker/venv/bin/pip install -r requirements-worker.txt

sudo cp transcribe_meeting.py worker_server.py worker_client.py /opt/transcript-worker/
sudo cp deploy/transcript-worker.service /etc/systemd/system/
sudo cp deploy/transcript-worker.env.example /etc/transcript-worker.env
sudo chmod 600 /etc/transcript-worker.env
```

Edit `/etc/transcript-worker.env` and set `TRANSCRIPT_WORKER_TOKEN`. Leave `TRANSCRIPT_BIND` on `127.0.0.1`. The worker uses all but two CPU cores unless you set `CPU_THREAD_LIMIT`.

Confirm the imports, then start the service:

```bash
sudo -u transcript HF_HOME=/var/transcript-worker/models \
  /opt/transcript-worker/venv/bin/python -c 'from faster_whisper import WhisperModel; print("whisper ok")'

sudo systemctl daemon-reload
sudo systemctl enable --now transcript-worker
curl -fsS http://127.0.0.1:8787/v1/health
```

Put nginx (or Caddy) on 443 in front of port 8787. `deploy/nginx-transcript-worker.conf` is a starting point: TLS, `client_max_body_size 1024m`, and `allow` only the BigBlueButton host. A two-hour Opus upload at 32 kbit/s is about 30 MB; the cap is there so a stray video upload cannot fill the disk.

On first real meeting the worker downloads the Whisper model and, unless `LLAMA_MODEL_PATH` is set, a Gemma GGUF into `/var/transcript-worker/models`. The machine needs outbound HTTPS for that download. 16 GB of RAM is the practical floor; 32 GB leaves room for a long meeting.

## 2. BigBlueButton host

The hook and the retrieve timer use the system Python. They do not need the speech virtualenv.

```bash
sudo apt install -y ffmpeg

sudo mkdir -p /usr/local/bigbluebutton/core/scripts/post_publish/lib
sudo cp 01_transcribe_whisper.sh /usr/local/bigbluebutton/core/scripts/post_publish/
sudo cp bbb_submit.py bbb_retrieve.py bbb_jobs.py bbb_publish.py worker_client.py \
  /usr/local/bigbluebutton/core/scripts/post_publish/lib/
sudo chmod 755 /usr/local/bigbluebutton/core/scripts/post_publish/01_transcribe_whisper.sh
sudo chown -R bigbluebutton:bigbluebutton \
  /usr/local/bigbluebutton/core/scripts/post_publish/01_transcribe_whisper.sh \
  /usr/local/bigbluebutton/core/scripts/post_publish/lib

sudo mkdir -p /var/bigbluebutton/transcript-jobs /var/bigbluebutton/published/summary /var/bigbluebutton/published/transcript
sudo chown bigbluebutton:bigbluebutton /var/bigbluebutton/transcript-jobs
sudo chown -R bigbluebutton:bigbluebutton /var/bigbluebutton/published/summary /var/bigbluebutton/published/transcript

sudo cp deploy/bbb-transcript-worker.env.example /etc/bigbluebutton/transcript-worker.env
sudo chown root:bigbluebutton /etc/bigbluebutton/transcript-worker.env
sudo chmod 640 /etc/bigbluebutton/transcript-worker.env
```

Edit `/etc/bigbluebutton/transcript-worker.env`. `TRANSCRIPT_WORKER_URL` is the nginx HTTPS URL with no path. `TRANSCRIPT_WORKER_TOKEN` matches the worker.

`post_publish.rb.example` (or the live `post_publish.rb`) should call the hook and ignore its exit status. The presentation is already published:

```ruby
logger.info("Submitting transcript job for meeting: #{meeting_id}")
system("/usr/local/bigbluebutton/core/scripts/post_publish/01_transcribe_whisper.sh #{meeting_id}")
```

Install the retrieve timer:

```bash
sudo cp deploy/bbb-transcript-retrieve.service deploy/bbb-transcript-retrieve.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now bbb-transcript-retrieve.timer
```

The timer runs once a minute as `bigbluebutton`. It uploads anything still in `pending-submit`, downloads ready results, renders `summary.html` and the RTF, registers the playback formats, and deletes the worker copy.

## 3. Try one recording

On the BigBlueButton host, after a recording has published:

```bash
sudo -u bigbluebutton bash -lc 'set -a; . /etc/bigbluebutton/transcript-worker.env; set +a; python3 /usr/local/bigbluebutton/core/scripts/post_publish/lib/bbb_submit.py MEETING_ID'
```

Watch the worker journal until the job is `ready`:

```bash
sudo journalctl -u transcript-worker -f
```

Then run one retrieve on the BigBlueButton host:

```bash
sudo systemctl start bbb-transcript-retrieve.service
sudo journalctl -u bbb-transcript-retrieve.service -n 50
```

Confirm these URLs load, and that Greenlight lists summary and transcript for that recording:

- `https://<bbb-host>/presentation/MEETING_ID/summary/summary.html`
- `https://<bbb-host>/presentation/MEETING_ID/transcript/transcript.txt`

A 90-minute class is on the order of 35 minutes of Whisper on the old Xeon path, then the Gemma summary. A second recording waits in the queue until the first finishes. The links appear when that job reaches the front and the timer publishes it.

## 4. After a few real meetings

The BigBlueButton host no longer needs `/opt/speech_env` or `/var/bigbluebutton/hf_cache`. Remove them once you are satisfied the worker is keeping up. Failed jobs are recorded in `/var/bigbluebutton/transcript-jobs/<meeting_id>.json` and in `/var/log/bigbluebutton/post_publish_transcribe.log`. Publishing the recording again submits the meeting once more.

Prompts live in `transcribe_meeting.py`: the chunk system prompt inside `generate_summary`, and `final_system_prompt` just below it. The Whisper `initial_prompt` is in `write_transcript`.
