#!/usr/bin/env /opt/speech_env/bin/python3
import sys
import os
import glob
import time
from faster_whisper import WhisperModel
# file location: /usr/local/bigbluebutton/core/scripts/transcribe_meeting.py
# Direct HuggingFace model cache to the large storage volume
os.environ["HF_HUB_CACHE"] = "/var/bigbluebutton/hf_cache"

def format_timestamp(seconds: float) -> str:
    """Formats seconds to WebVTT timestamp format (HH:MM:SS.mmm)"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    msecs = int((seconds % 1) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{msecs:03d}"

def process_recording(meeting_id):
    publish_dir = f"/var/bigbluebutton/published/presentation/{meeting_id}"
    
    if not os.path.exists(publish_dir):
        print(f"[Error] Publish directory not found: {publish_dir}")
        sys.exit(1)

    # Locate meeting audio/video sources
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

    # Load faster-whisper with CPU optimizations (INT8, 6 threads)
    model = WhisperModel("medium", device="cpu", compute_type="int8", cpu_threads=6)

    start_time = time.time()
    segments, info = model.transcribe(audio_file, beam_size=5, vad_filter=True)

    vtt_path = f"{publish_dir}/transcript.vtt"
    txt_path = f"{publish_dir}/transcript.txt"

    with open(vtt_path, "w", encoding="utf-8") as vtt_file, \
         open(txt_path, "w", encoding="utf-8") as txt_file:
        
        vtt_file.write("WEBVTT\n\n")

        for segment in segments:
            start_str = format_timestamp(segment.start)
            end_str = format_timestamp(segment.end)
            text = segment.text.strip()
            
            # Write WebVTT caption line
            vtt_file.write(f"{start_str} --> {end_str}\n{text}\n\n")
            
            # Write plain text transcript line
            txt_file.write(f"[{start_str}] {text}\n")

    elapsed = time.time() - start_time
    print(f"[Transcribe] Completed in {elapsed:.2f}s! Artifacts generated:\n  - {vtt_path}\n  - {txt_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 transcribe_meeting.py <meeting_id>")
        sys.exit(1)

    process_recording(sys.argv[1])
