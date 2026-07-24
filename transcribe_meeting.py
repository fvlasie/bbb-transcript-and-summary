import sys
import os
import re
import json
import urllib.request
import xml.etree.ElementTree as ET
from faster_whisper import WhisperModel

import xml.etree.ElementTree as ET

def parse_bbb_speaker_timeline(events_xml_path: str):
    """
    Parses BBB 3.x events.xml to map speaker names to audio timestamps (0.0s relative to audio.webm).
    """
    if not os.path.exists(events_xml_path):
        print(f"[Speaker Map] Warning: {events_xml_path} not found.")
        return []

    try:
        tree = ET.parse(events_xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[Speaker Map] Error parsing XML: {e}")
        return []

    # 1. Map participant IDs to Display Names
    # Check PARTICIPANT module join events
    participants = {}
    for event in root.findall(".//event[@module='PARTICIPANT'][@eventname='ParticipantJoinEvent']"):
        u_id = event.findtext("userId")
        name = event.findtext("name")
        if u_id and name:
            participants[u_id] = name

    # Check VOICE module join events (sometimes callernumber or participant is used)
    for event in root.findall(".//event[@module='VOICE'][@eventname='ParticipantJoinedEvent']"):
        p_id = event.findtext("participant")
        caller_name = event.findtext("callername")
        if p_id and caller_name:
            participants[p_id] = caller_name

    # 2. Find the exact StartRecordingEvent timestamp (t0 for audio.webm)
    start_rec_event = root.find(".//event[@module='VOICE'][@eventname='StartRecordingEvent']")
    if start_rec_event is None or start_rec_event.find("timestampUTC") is None:
        print("[Speaker Map] Error: Could not find StartRecordingEvent timestampUTC.")
        return []

    t0_ms = float(start_rec_event.findtext("timestampUTC"))

    # 3. Track talking state and build intervals
    active_talkers = {} # participant_id -> start_sec
    timeline = []

    for event in root.findall(".//event[@module='VOICE'][@eventname='ParticipantTalkingEvent']"):
        p_id = event.findtext("participant")
        is_talking = event.findtext("talking", "false").lower() == "true"
        ts_utc_node = event.find("timestampUTC")

        if not p_id or ts_utc_node is None:
            continue

        ts_ms = float(ts_utc_node.text)
        # Calculate seconds from start of audio.webm
        rel_sec = max(0.0, (ts_ms - t0_ms) / 1000.0)

        if is_talking:
            active_talkers[p_id] = rel_sec
        else:
            if p_id in active_talkers:
                start_sec = active_talkers.pop(p_id)
                speaker_name = participants.get(p_id, "Unknown Speaker")
                timeline.append({
                    "start": start_sec,
                    "end": rel_sec,
                    "name": speaker_name
                })

    # Close any talkers who were still talking when recording ended
    for p_id, start_sec in active_talkers.items():
        speaker_name = participants.get(p_id, "Unknown Speaker")
        timeline.append({
            "start": start_sec,
            "end": start_sec + 5.0, # Default padding
            "name": speaker_name
        })

    print(f"[Speaker Map] Successfully mapped {len(timeline)} voice intervals for {len(participants)} participants.")
    return timeline

def get_speaker_for_timestamp(start_sec: float, end_sec: float, timeline: list) -> str:
    """
    Finds the speaker active during [start_sec, end_sec] with fuzzy tolerance.
    """
    if not timeline:
        return ""

    midpoint = (start_sec + end_sec) / 2.0
    
    # 1. Direct overlap check
    for interval in timeline:
        if interval["start"] <= midpoint <= interval["end"]:
            return interval["name"]

    # 2. Fuzzy match within a 1.5-second buffer (handles slight VAD drift)
    for interval in timeline:
        if not (end_sec + 1.5 < interval["start"] or start_sec - 1.5 > interval["end"]):
            return interval["name"]

    return ""

def get_best_active_cf_model(account_id: str, api_token: str) -> str:
    catalog_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/models/search?task=Text%20Generation"
    try:
        req = urllib.request.Request(catalog_url, headers={"Authorization": f"Bearer {api_token}"})
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            if res_data.get("success"):
                models = res_data.get("result", [])
                valid_models = [
                    m["name"] for m in models 
                    if m.get("name", "").startswith("@cf/") and not m.get("description", "").lower().startswith("deprecated")
                ]
                if valid_models:
                    priority_order = ["llama-3.3-70b", "llama-3.1-70b", "qwen2.5-72b", "llama-3.2-3b", "glm-4"]
                    for preferred in priority_order:
                        for m in valid_models:
                            if preferred in m:
                                print(f"[Catalog] Selected model: {m}")
                                return m
                    return valid_models[0]
    except Exception as e:
        print(f"[Warning] Catalog query failed ({e}). Using default fallback.")

    return "@cf/meta/llama-3.3-70b-instruct-fp8-fast"


def generate_summary(transcript_text: str) -> str:
    words = transcript_text.strip().split()
    if len(words) < 8:
        return (
            "## Executive Summary\n"
            "No active meeting discussion recorded (empty transcript or automated prompt).\n\n"
            "## Key Discussion Points\nNone\n\n"
            "## Decisions Made\nNone\n\n"
            "## Action Items & Next Steps\nNone"
        )

    account_id = os.environ.get("CF_ACCOUNT_ID")
    api_token = os.environ.get("CF_API_TOKEN")

    if not account_id or not api_token:
        return "[Error: CF_ACCOUNT_ID or CF_API_TOKEN environment variables not set]"

    model = get_best_active_cf_model(account_id, api_token)
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"

    system_prompt = (
        "You are an executive assistant. Your task is to provide a clean, accurate meeting summary based on speaker-attributed transcripts. "
        "CRITICAL INSTRUCTIONS:\n"
        "1. Do NOT include conversational intros or preambles (e.g., 'Here is a summary'). Start directly with '## Executive Summary'.\n"
        "2. Do NOT inflate brief off-hand remarks, mic checks, or automated prompts into major discussion points.\n"
        "3. Attribute key points or action items to specific speakers if mentioned in the transcript.\n"
        "4. If a section has no meaningful content, explicitly write 'None'."
    )

    user_prompt = (
        "Summarize the following speaker-tagged meeting transcript accurately.\n\n"
        "Formatting guidelines:\n"
        "## Executive Summary\n"
        "## Key Discussion Points\n"
        "## Decisions Made\n"
        "## Action Items & Next Steps\n\n"
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
                raw_summary = res_data.get("result", {}).get("response", "")
                cleaned = re.sub(
                    r"^(here is (a|the) summary[^\n]*\n?|sure[^\n]*\n?|certainly[^\n]*\n?)", 
                    "", 
                    raw_summary, 
                    flags=re.IGNORECASE
                ).strip()
                return cleaned
            else:
                return f"[Cloudflare API Error: {res_data.get('errors')}]"

    except Exception as e:
        return f"[Error generating summary: {e}]"


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 transcribe_meeting.py <audio_file_path> <output_dir> [events_xml_path]")
        sys.exit(1)

    audio_path = sys.argv[1]
    output_dir = sys.argv[2]
    events_xml_path = sys.argv[3] if len(sys.argv) > 3 else ""

    print(f"[Whisper] Loading model 'medium'...")
    model = WhisperModel("medium", device="cpu", compute_type="int8", download_root="/var/bigbluebutton/hf_cache")

    print(f"[Whisper] Transcribing {audio_path}...")
    segments, _ = model.transcribe(audio_path, vad_filter=True, language="en")

    # Parse speaker timeline from events.xml
    timeline = parse_bbb_speaker_timeline(events_xml_path) if events_xml_path else []

    full_transcript = []
    vtt_lines = ["WEBVTT\n"]

    for segment in segments:
        speaker = get_speaker_for_timestamp(segment.start, segment.end, timeline)
        speaker_prefix = f"[{speaker}]: " if speaker else ""
        
        text_line = f"{speaker_prefix}{segment.text.strip()}"
        full_transcript.append(text_line)

        # Format WebVTT timestamps (HH:MM:SS.mmm)
        start_fmt = f"{int(segment.start//3600):02d}:{int((segment.start%3600)//60):02d}:{segment.start%60:06.3f}"
        end_fmt = f"{int(segment.end//3600):02d}:{int((segment.end%3600)//60):02d}:{segment.end%60:06.3f}"
        
        vtt_lines.append(f"{start_fmt} --> {end_fmt}\n{text_line}\n")

    full_text = "\n".join(full_transcript)

    # Save transcript files
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "transcript.txt"), "w") as f:
        f.write(full_text)

    with open(os.path.join(output_dir, "transcript.vtt"), "w") as f:
        f.write("\n".join(vtt_lines))

    print("[Summary] Generating AI meeting summary...")
    summary = generate_summary(full_text)

    with open(os.path.join(output_dir, "transcript_summary.txt"), "w") as f:
        f.write(summary)

    print("[Complete] Processing finished successfully.")

if __name__ == "__main__":
    main()
