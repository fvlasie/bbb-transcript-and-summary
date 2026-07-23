#!/usr/bin/env /opt/speech_env/bin/python3
import sys
import os
import glob
import time
import json
import urllib.request
import urllib.error
from faster_whisper import WhisperModel

os.environ["HF_HUB_CACHE"] = "/var/bigbluebutton/hf_cache"

def format_timestamp(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    msecs = int((seconds % 1) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{msecs:03d}"

def generate_summary(transcript_text: str) -> str:
    """
    Sends the transcript to Cloudflare Workers AI REST API.
    Uses @cf/meta/llama-3-8b-instruct on CF's free tier.
    """
    account_id = os.environ.get("CF_ACCOUNT_ID")
    api_token = os.environ.get("CF_API_TOKEN")

    if not account_id or not api_token:
        return "[Error: CF_ACCOUNT_ID or CF_API_TOKEN environment variables not set]"

    model = "@cf/meta/llama-3-8b-instruct"
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"

    system_prompt = "You are an AI executive assistant. Summarize meeting transcripts accurately and concisely."
    user_prompt = (
        "Summarize the following meeting transcript.\n\n"
        "Formatting guidelines:\n"
        "1. Executive Summary (2-3 sentences)\n"
        "2. Key Discussion Points (bullet points)\n"
        "3. Decisions Made\n"
        "4. Action Items & Next Steps\n\n"
        f"TRANSCRIPT:\n{transcript_text}"
    )

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 1024
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json"
            }
        )
        with urllib.request.urlopen(req, timeout=45) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            if res_data.get("success"):
                return res_data.get("result", {}).get("response", "Error: Empty response.")
            else:
                errors = res_data.get("errors", [])
                return f"[Cloudflare API Error: {errors}]"

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        return f"[HTTP Error {e.code}: {error_body}]"
    except Exception as e:
        return f"[Error generating summary: {e}]"

def process_recording(meeting_id):
    publish_dir = f"/var/bigbluebutton/published/presentation/{meeting_id}"
    
    if not os.path.exists(publish_dir):
        print(f"[Error] Publish directory not found: {publish_dir}")
        sys.exit(1)

    audio_candidates = (
        glob.glob(f"{publish_dir}/video/webcams.webm") +
        glob.glob(f"{publish_dir}/audio/audio.webm") +
        glob.glob(f"{publish_dir}/audio/audio.wav")
    )

    if not audio_candidates:
        print(f"[Error] No supported audio/video files found for meeting {meeting_id}")
        sys.exit(1)

    audio_file = audio_candidates[0]
    print(f"[Transcribe] Processing file: {audio_file}")

    model = WhisperModel("medium", device="cpu", compute_type="int8", cpu_threads=6)

    start_time = time.time()
    segments, info = model.transcribe(audio_file, beam_size=5, vad_filter=True)

    vtt_path = f"{publish_dir}/transcript.vtt"
    txt_path = f"{publish_dir}/transcript.txt"
    summary_path = f"{publish_dir}/transcript_summary.txt"

    full_transcript = []

    with open(vtt_path, "w", encoding="utf-8") as vtt_file, \
         open(txt_path, "w", encoding="utf-8") as txt_file:
        
        vtt_file.write("WEBVTT\n\n")

        for segment in segments:
            start_str = format_timestamp(segment.start)
            end_str = format_timestamp(segment.end)
            text = segment.text.strip()
            
            vtt_file.write(f"{start_str} --> {end_str}\n{text}\n\n")
            txt_file.write(f"[{start_str}] {text}\n")
            full_transcript.append(f"[{start_str}] {text}")

    print(f"[Transcribe] Finished in {time.time() - start_time:.2f}s.")

    # --- Summarization Step ---
    print("[Summary] Generating meeting summary via Cloudflare Workers AI...")
    raw_text = "\n".join(full_transcript)
    summary = generate_summary(raw_text)

    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write(summary)

    print(f"[Summary] Saved to: {summary_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 transcribe_meeting.py <meeting_id>")
        sys.exit(1)

    process_recording(sys.argv[1])        "prompt": prompt,
        "stream": False
    }

    try:
        req = urllib.request.Request(
            url, 
            data=json.dumps(payload).encode("utf-8"), 
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data.get("response", "Error: Empty response from LLM.")
    except Exception as e:
        return f"[Error generating summary: {e}]"

def process_recording(meeting_id):
    publish_dir = f"/var/bigbluebutton/published/presentation/{meeting_id}"
    
    if not os.path.exists(publish_dir):
        print(f"[Error] Publish directory not found: {publish_dir}")
        sys.exit(1)

    audio_candidates = (
        glob.glob(f"{publish_dir}/video/webcams.webm") +
        glob.glob(f"{publish_dir}/audio/audio.webm") +
        glob.glob(f"{publish_dir}/audio/audio.wav")
    )

    if not audio_candidates:
        print(f"[Error] No supported audio/video files found for meeting {meeting_id}")
        sys.exit(1)

    audio_file = audio_candidates[0]
    print(f"[Transcribe] Processing file: {audio_file}")

    model = WhisperModel("medium", device="cpu", compute_type="int8", cpu_threads=6)

    start_time = time.time()
    segments, info = model.transcribe(audio_file, beam_size=5, vad_filter=True)

    vtt_path = f"{publish_dir}/transcript.vtt"
    txt_path = f"{publish_dir}/transcript.txt"
    summary_path = f"{publish_dir}/transcript_summary.txt"

    full_transcript = []

    with open(vtt_path, "w", encoding="utf-8") as vtt_file, \
         open(txt_path, "w", encoding="utf-8") as txt_file:
        
        vtt_file.write("WEBVTT\n\n")

        for segment in segments:
            start_str = format_timestamp(segment.start)
            end_str = format_timestamp(segment.end)
            text = segment.text.strip()
            
            vtt_file.write(f"{start_str} --> {end_str}\n{text}\n\n")
            txt_file.write(f"[{start_str}] {text}\n")
            full_transcript.append(f"[{start_str}] {text}")

    print(f"[Transcribe] Finished in {time.time() - start_time:.2f}s.")

    # --- Summarization Step ---
    print("[Summary] Sending transcript to LLM...")
    raw_text = "\n".join(full_transcript)
    summary = generate_summary(raw_text)

    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write(summary)

    print(f"[Summary] Saved to: {summary_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 transcribe_meeting.py <meeting_id>")
        sys.exit(1)

    process_recording(sys.argv[1])
