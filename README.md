# BigBlueButton 3.0 Native Transcription & AI Summarization Pipeline

A lightweight, fully native post-processing pipeline for BigBlueButton (BBB) 3.0 that automatically generates speaker-attributed transcripts and formatted AI summaries upon recording completion.

Artifacts are delivered directly via the default BBB presentation playback layer (`/presentation/<meeting_id>/`), requiring zero manual Nginx route changes.

---

## 🌟 Key Features

* **Local Speech-to-Text (`faster-whisper`)**: High-speed, int8-quantized `large-v3-turbo` model executing at ~0.4x real-time on CPU.
* **Speaker Attribution**: Cross-references BBB `events.xml` timestamps with audio segments to automatically identify speaker names in transcript logs.
* **AI Meeting Summaries**: Generates structured Markdown, HTML, and dynamic downloadable RTF summaries via Cloudflare Workers AI.
* **Native BBB Integration**: Registers custom formats (`summary` and `transcript`) directly into BBB metadata so front-ends like Greenlight automatically display valid playback links.
* **Zero Nginx Configuration**: Files are hosted inside standard presentation folders, avoiding routing conflicts and custom web server hacks.

---

## 🏗️ Repository Structure

| File | Description |
| :--- | :--- |
| **`01_transcribe_whisper.sh`** | The BBB post-publish hook script called automatically by BigBlueButton when a recording finishes. |
| **`transribe_meeting.py`** | Core Python script handling Whisper transcription, speaker timeline alignment, LLM summarization, HTML/RTF rendering, and XML metadata registration. |
| **`install.md`** | **Step-by-step installation guide**, covering virtual environment setup, system dependencies, permissions, and BBB post-publish script integration. |
| **`README.md`** | High-level architecture overview and repository documentation (this file). |

---

## 🔄 Workflow Overview

```text
[BBB Recording Ends]
         │
         ▼
[01_transcribe_whisper.sh] ──► Runs Python Venv pipeline
         │
         ├── 1. Parse timeline from events.xml (Speaker attribution)
         ├── 2. Transcribe WebM using faster-whisper (large-v3-turbo)
         ├── 3. Output WebVTT & Speaker TXT -> /presentation/<meeting_id>/transcript/
         ├── 4. Summarize via Cloudflare Workers AI -> HTML & RTF
         └── 5. Register native playback URLs in metadata.xml
         │
         ▼
[Published Artifacts] Available natively at https://bbb.yourdomain.com/presentation/<meeting_id>/...
```
---

## 🚀 Quick Setup & Installation

Detailed instructions for dependencies, virtual environments, permissions, and BBB post-publish registration are documented in **[`install.md`](./install.md)**.

### General Highlights:
1. Python dependencies (e.g., `faster-whisper`, `markdown`) are isolated in `/opt/speech_env/`.
2. Post-publish script `01_transcribe_whisper.sh` is symlinked or placed into `/var/bigbluebutton/recording/publish/post_publish/`.
3. All generated artifacts are set with `bigbluebutton:bigbluebutton` ownership and `0644` permissions for web access.

---

## 🛠️ Usage / Manual Execution

To manually test or re-process a past meeting recording:

```bash
# Form: /opt/speech_env/bin/python3 transcribe_meeting.py <MEETING_ID>
/opt/speech_env/bin/python3 transcribe_meeting.py 2187398764890732491623487923249-972314691

## 📄 License
Distributed under the MIT License. Feel free to modify and adapt for your institutional BigBlueButton deployment.
