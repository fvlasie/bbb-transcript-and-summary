import sys
import os
import copy
import shutil
import re
import json
import urllib.request
import xml.etree.ElementTree as ET
import markdown
from faster_whisper import WhisperModel

import markdown  # Or python-markdown / simple regex if no third-party libs installed

def generate_rtf_from_md(md_text: str, rtf_path: str):
    """
    Converts Markdown text into a native RTF file using pandoc if available,
    or a simple formatted RTF template fallback.
    """
    if shutil.which("pandoc"):
        try:
            import subprocess
            subprocess.run(
                ["pandoc", "-f", "markdown", "-t", "rtf", "-o", rtf_path],
                input=md_text.encode("utf-8"),
                check=True
            )
            print(f"[RTF] Generated {rtf_path} via Pandoc.")
            return
        except Exception as e:
            print(f"[RTF Warning] Pandoc conversion failed: {e}. Falling back to basic RTF.")

    # Fallback basic RTF wrapper if pandoc isn't installed
    rtf_content = (
        "{\\rtf1\\ansi\\deff0\n"
        "{\\fonttbl{\\f0\\froman Gentium Book Plus;}}\n"
        "\\f0\\fs24\n"
    )
    for line in md_text.splitlines():
        if line.startswith("## "):
            rtf_content += f"\\b\\fs28 {line[3:]}\\b0\\fs24\\par\n"
        elif line.startswith("### "):
            rtf_content += f"\\b\\fs26 {line[4:]}\\b0\\fs24\\par\n"
        elif line.startswith("- "):
            rtf_content += f"\\bullet  {line[2:]}\\par\n"
        elif line.strip() == "":
            rtf_content += "\\par\n"
        else:
            rtf_content += f"{line}\\par\n"
    rtf_content += "}"

    with open(rtf_path, "w", encoding="utf-8") as f:
        f.write(rtf_content)


def generate_html_viewer(md_text: str, html_path: str, meeting_id: str):
    """
    Renders Markdown summary into a stylized HTML paper document with a downloadable RTF button.
    """
    # Simple Markdown-to-HTML parsing rules
    html_body = []
    in_list = False

    for line in md_text.splitlines():
        clean_line = line.strip()
        
        # Handle lists
        if clean_line.startswith("- ") or clean_line.startswith("* "):
            if not in_list:
                html_body.append("<ul>")
                in_list = True
            item_text = clean_line[2:]
            # Bold parsing (**text**)
            item_text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", item_text)
            html_body.append(f"  <li>{item_text}</li>")
            continue
        else:
            if in_list:
                html_body.append("</ul>")
                in_list = False

        # Headings & paragraphs
        if clean_line.startswith("## "):
            html_body.append(f"<h2>{clean_line[3:]}</h2>")
        elif clean_line.startswith("### "):
            html_body.append(f"<h3>{clean_line[4:]}</h3>")
        elif clean_line == "":
            continue
        else:
            text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", clean_line)
            html_body.append(f"<p>{text}</p>")

    if in_list:
        html_body.append("</ul>")

    body_content = "\n".join(html_body)

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Meeting Summary - {meeting_id}</title>
    <style>
        @import url('https://fonts.bunny.net/css?family=gentium-book-plus:400,400i,700,700i');

        * {{
            box-sizing: border-box;
        }}

        body {{
            background-color: #6f7c8d;
            color: #2d3748;
            font-family: 'Gentium Book Plus', Georgia, serif;
            font-size: 19px;
            line-height: 1.6;
            margin: 0;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
        }}

        .paper-container {{
            background: #ffffff;
            width: 100%;
            max-width: 820px;
            padding: 60px 70px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.03);
            position: relative;
        }}

        .toolbar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #edf2f7;
            padding-bottom: 20px;
            margin-bottom: 35px;
        }}

        .meeting-title {{
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #718096;
            font-weight: 700;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}

        .btn-download {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background-color: #2b6cb0;
            color: #ffffff;
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 14px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-weight: 600;
            transition: background-color 0.2s ease;
        }}

        .btn-download:hover {{
            background-color: #2c5282;
        }}

        h2 {{
            font-size: 26px;
            color: #1a202c;
            margin-top: 35px;
            margin-bottom: 15px;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 6px;
        }}

        h3 {{
            font-size: 21px;
            color: #2d3748;
            margin-top: 25px;
            margin-bottom: 10px;
        }}

        p {{
            margin-bottom: 18px;
        }}

        ul {{
            margin-top: 5px;
            margin-bottom: 20px;
            padding-left: 25px;
        }}

        li {{
            margin-bottom: 8px;
        }}

        strong {{
            color: #1a202c;
        }}
    </style>
</head>
<body>
    <div class="paper-container">
        <div class="toolbar">
            <span class="meeting-title">Meeting Summary</span>
            <a href="summary.rtf" download class="btn-download">
                <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                    <path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z"/>
                    <path d="M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708l3 3z"/>
                </svg>
                Download RTF
            </a>
        </div>
        <div class="content">
            {body_content}
        </div>
    </div>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_template)

def parse_bbb_speaker_timeline(events_xml_path: str):
    """
    Parses BBB 3.x events.xml to map speaker names to audio timestamps 
    relative to StartRecordingEvent (t0).
    """
    if not os.path.exists(events_xml_path):
        print(f"[Speaker Map] Warning: XML not found at {events_xml_path}. Skipping speaker attribution.")
        return []

    try:
        tree = ET.parse(events_xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[Speaker Map] Error parsing XML: {e}")
        return []

    # 1. Map participant IDs to Display Names
    participants = {}
    for event in root.findall(".//event[@module='PARTICIPANT'][@eventname='ParticipantJoinEvent']"):
        u_id = event.findtext("userId")
        name = event.findtext("name")
        if u_id and name:
            participants[u_id] = name

    for event in root.findall(".//event[@module='VOICE'][@eventname='ParticipantJoinedEvent']"):
        p_id = event.findtext("participant")
        caller_name = event.findtext("callername")
        if p_id and caller_name:
            participants[p_id] = caller_name

    # 2. Find StartRecordingEvent timestampUTC (t0)
    start_rec_event = root.find(".//event[@module='VOICE'][@eventname='StartRecordingEvent']")
    if start_rec_event is None or start_rec_event.find("timestampUTC") is None:
        print("[Speaker Map] Warning: StartRecordingEvent missing in XML. Skipping speaker attribution.")
        return []

    t0_ms = float(start_rec_event.findtext("timestampUTC"))

    # 3. Extract talking intervals
    active_talkers = {}
    timeline = []

    for event in root.findall(".//event[@module='VOICE'][@eventname='ParticipantTalkingEvent']"):
        p_id = event.findtext("participant")
        is_talking = event.findtext("talking", "false").lower() == "true"
        ts_utc_node = event.find("timestampUTC")

        if not p_id or ts_utc_node is None:
            continue

        ts_ms = float(ts_utc_node.text)
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

    for p_id, start_sec in active_talkers.items():
        speaker_name = participants.get(p_id, "Unknown Speaker")
        timeline.append({"start": start_sec, "end": start_sec + 5.0, "name": speaker_name})

    print(f"[Speaker Map] Successfully mapped {len(timeline)} talking segments from XML.")
    return timeline


def get_speaker_for_timestamp(start_sec: float, end_sec: float, timeline: list) -> str:
    """Finds speaker active during [start_sec, end_sec] with fuzzy buffer."""
    if not timeline:
        return ""

    midpoint = (start_sec + end_sec) / 2.0
    for interval in timeline:
        if interval["start"] <= midpoint <= interval["end"]:
            return interval["name"]

    for interval in timeline:
        if not (end_sec + 1.5 < interval["start"] or start_sec - 1.5 > interval["end"]):
            return interval["name"]

    return ""

def get_best_active_cf_model(account_id: str, api_token: str) -> str:
    """Queries Cloudflare Workers AI model catalog dynamically."""
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
                                print(f"[Catalog] Selected Cloudflare model: {m}")
                                return m
                    return valid_models[0]
    except Exception as e:
        print(f"[Warning] Catalog query failed ({e}). Falling back to default model.")

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
        "You are an executive assistant analyzing speaker-attributed transcripts.\n\n"
        "GOAL: Produce a faithful, comprehensive summary that captures both main topics and subtle, critical side remarks, technical details, or quick decisions.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Do NOT include conversational intros or preambles. Start directly with '## Executive Summary'.\n"
        "2. COMPREHENSION & DETAIL:\n"
        "   - Treat all spoken content as potentially significant, including side comments, quick clarifications, or niche/technical points.\n"
        "   - Do NOT omit arcane terminology, specific proper nouns, or short off-hand statements—preserve them as stated.\n"
        "3. EXECUTIVE SUMMARY:\n"
        "   - Provide a high-level overview (2-4 sentences) summarizing the main purpose or core themes of the discussion.\n"
        "   - If the transcript consists purely of casual banter, summarize the general nature of the conversation instead of writing 'None'.\n"
        "4. KEY DISCUSSION POINTS:\n"
        "   - Detail specific topics, claims, observations, or brief tangential points made during the session.\n"
        "   - Attribute key observations to specific speakers when clear from the text.\n"
        "5. DECISIONS & ACTION ITEMS:\n"
        "   - List any explicit or implied decisions, agreements, or follow-ups.\n"
        "   - If truly no decisions or action items were stated, write 'None' under those specific headings."
    )

    user_prompt = f"Summarize the following speaker-tagged meeting transcript accurately:\n\n{transcript_text}"

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
            headers={"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
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

def register_custom_formats_native(
    meeting_id: str,
    base_published_dir: str = "/var/bigbluebutton/published"
):
    """
    Creates native published format directories for 'summary' and 'transcript'
    by reading (and NOT modifying) the primary presentation metadata.xml.
    """
    domain = get_server_domain()
    # Guarantee protocol prefix
    if not domain.startswith("http://") and not domain.startswith("https://"):
        domain = f"https://{domain}"

    pres_dir = os.path.join(base_published_dir, "presentation", meeting_id)
    pres_meta_path = os.path.join(pres_dir, "metadata.xml")

    if not os.path.exists(pres_meta_path):
        print(f"[Metadata Error] Primary metadata file not found at {pres_meta_path}")
        return

    if os.path.getsize(pres_meta_path) == 0:
        print(f"[Metadata Error] Primary metadata file at {pres_meta_path} is empty.")
        return

    try:
        # READ ONLY: Parse primary presentation metadata.xml
        tree = ET.parse(pres_meta_path)
        root = tree.getroot()

        # Extract elements to copy into the new native format XMLs
        meeting_elem = root.find("meeting")
        meta_elem = root.find("meta")
        start_time = root.findtext("start_time", default="")
        end_time = root.findtext("end_time", default="")
        participants = root.findtext("participants", default="0")
        
        playback = root.find("playback")
        duration = playback.findtext("duration", default="0") if playback is not None else "0"

        # Point links directly to the presentation subfolders Nginx already serves
        formats = {
            "summary": f"{domain}/presentation/{meeting_id}/summary/summary.html",
            "transcript": f"{domain}/presentation/{meeting_id}/transcript/transcript.txt"
        }

        # Build published/summary/<meeting_id>/metadata.xml and published/transcript/<meeting_id>/metadata.xml
        for fmt_name, fmt_link in formats.items():
            fmt_dir = os.path.join(base_published_dir, fmt_name, meeting_id)
            os.makedirs(fmt_dir, exist_ok=True)

            new_root = ET.Element("recording")
            ET.SubElement(new_root, "id").text = meeting_id
            ET.SubElement(new_root, "state").text = "published"
            ET.SubElement(new_root, "published").text = "true"
            if start_time: ET.SubElement(new_root, "start_time").text = start_time
            if end_time: ET.SubElement(new_root, "end_time").text = end_time
            ET.SubElement(new_root, "participants").text = participants

            if meeting_elem is not None:
                new_root.append(meeting_elem)

            if meta_elem is not None:
                new_root.append(meta_elem)

            new_playback = ET.SubElement(new_root, "playback")
            ET.SubElement(new_playback, "format").text = fmt_name
            ET.SubElement(new_playback, "link").text = fmt_link
            ET.SubElement(new_playback, "processing_time").text = "0"
            ET.SubElement(new_playback, "duration").text = duration
            ET.SubElement(new_playback, "size").text = "0"

            meta_file_path = os.path.join(fmt_dir, "metadata.xml")
            new_tree = ET.ElementTree(new_root)
            
            # Atomic write strictly for the NEW format metadata file
            temp_meta_path = meta_file_path + ".tmp"
            ET.indent(new_tree, space="  ")
            new_tree.write(temp_meta_path, encoding="UTF-8", xml_declaration=True)
            shutil.move(temp_meta_path, meta_file_path)

            shutil.chown(fmt_dir, user="bigbluebutton", group="bigbluebutton")
            shutil.chown(meta_file_path, user="bigbluebutton", group="bigbluebutton")
            os.chmod(fmt_dir, 0o755)
            os.chmod(meta_file_path, 0o644)

            print(f"[Success] Created native format for '{fmt_name}' at {meta_file_path}")

    except ET.ParseError as pe:
        print(f"[Metadata Error] Malformed XML encountered in primary metadata: {pe}")
    except Exception as e:
        print(f"[Metadata Error] Failed to process {meeting_id}: {e}")

def get_server_domain() -> str:
    """
    Reads the BigBlueButton server URL directly from bbb-web.properties.
    Raises RuntimeError if the file or key is missing.
    """
    prop_path = "/etc/bigbluebutton/bbb-web.properties"
    
    if os.path.exists(prop_path):
        try:
            with open(prop_path, "r") as f:
                for line in f:
                    if line.strip().startswith("bigbluebutton.web.serverURL="):
                        url = line.split("=", 1)[1].strip()
                        if url:
                            return url.rstrip("/")
        except Exception as e:
            raise RuntimeError(f"Failed to read {prop_path}: {e}")

    raise RuntimeError(f"Could not resolve server URL. Key 'bigbluebutton.web.serverURL' missing or file not found at {prop_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 transcribe_meeting.py <meeting_id>")
        sys.exit(1)

    meeting_id = sys.argv[1]
    base_published_dir = "/var/bigbluebutton/published"
    
    # Input webm source directory (from default presentation playback)
    pres_dir = os.path.join(base_published_dir, "presentation", meeting_id)
    events_xml_path = f"/var/bigbluebutton/recording/raw/{meeting_id}/events.xml"
    
    # Destination directories inside presentation folder (Nginx serves these natively)
    transcript_dir = os.path.join(pres_dir, "transcript")
    summary_dir = os.path.join(pres_dir, "summary")
    
    # Ensure destination directories exist
    os.makedirs(transcript_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    # Check for audio/video source file
    webm_path = os.path.join(pres_dir, "video", "webcams.webm")
    if not os.path.exists(webm_path):
        fallback_audio = os.path.join(pres_dir, "audio", "audio.webm")
        if os.path.exists(fallback_audio):
            webm_path = fallback_audio
        else:
            print(f"[Error] No audio/video WebM file found for meeting {meeting_id} under {pres_dir}")
            sys.exit(1)

    print(f"--- Processing AI Pipeline for Recording: {meeting_id} ---")

    # 1. Parse speaker timeline from events.xml
    timeline = parse_bbb_speaker_timeline(events_xml_path) if os.path.exists(events_xml_path) else []

    # 2. Transcribe published WebM directly using faster-whisper
    print(f"[Whisper] Loading model 'large-v3-turbo'...")
    model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8", download_root="/var/bigbluebutton/hf_cache")

    print(f"[Whisper] Transcribing WebM: {webm_path}...")
    segments, _ = model.transcribe(webm_path, vad_filter=True, language="en")

    full_transcript = []
    vtt_lines = ["WEBVTT\n"]
    current_speaker = None

    # 3. Format WebVTT and TXT with speaker attribution
    for segment in segments:
        raw_text = segment.text.strip()

        # Filter out automated BBB voice prompt
        if "you are currently the only person" in raw_text.lower():
            continue

        speaker = get_speaker_for_timestamp(segment.start, segment.end, timeline)

        # --- TXT Formatting (Grouped by Speaker with Timecodes) ---
        hrs = int(segment.start // 3600)
        mins = int((segment.start % 3600) // 60)
        secs = int(segment.start % 60)
        time_str = f"[{hrs:02d}:{mins:02d}:{secs:02d}]" if hrs > 0 else f"[{mins:02d}:{secs:02d}]"

        if speaker and speaker != current_speaker:
            if full_transcript:
                full_transcript.append("")  # Blank line separator
            full_transcript.append(f"[{speaker}]")
            current_speaker = speaker

        full_transcript.append(f"{time_str}: {raw_text}")

        # --- WebVTT Formatting ---
        vtt_prefix = f"[{speaker}]: " if speaker else ""
        vtt_text = f"{vtt_prefix}{raw_text}"

        start_fmt = f"{int(segment.start//3600):02d}:{int((segment.start%3600)//60):02d}:{segment.start%60:06.3f}"
        end_fmt = f"{int(segment.end//3600):02d}:{int((segment.end%3600)//60):02d}:{segment.end%60:06.3f}"

        vtt_lines.append(f"{start_fmt} --> {end_fmt}\n{vtt_text}\n")

    full_text = "\n".join(full_transcript)

    # 4. Save transcript artifacts
    txt_path = os.path.join(transcript_dir, "transcript.txt")
    vtt_path = os.path.join(transcript_dir, "transcript.vtt")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text + "\n")

    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(vtt_lines) + "\n")

    # Apply permissions to transcript artifacts
    for path in [transcript_dir, txt_path, vtt_path]:
        shutil.chown(path, user="bigbluebutton", group="bigbluebutton")
        os.chmod(path, 0o755 if os.path.isdir(path) else 0o644)

    # 5. Summarize via Cloudflare Workers AI & save summary artifacts
    print("[Summary] Generating AI meeting summary...")
    summary = generate_summary(full_text)

    summary_md_path = os.path.join(summary_dir, "summary.md")
    summary_html_path = os.path.join(summary_dir, "summary.html")
    summary_rtf_path = os.path.join(summary_dir, "summary.rtf")

    # Write Markdown
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    # Generate HTML & RTF
    generate_html_viewer(summary, summary_html_path, meeting_id)
    generate_rtf_from_md(summary, summary_rtf_path)

    # Apply permissions across all summary artifacts
    for path in [summary_dir, summary_md_path, summary_html_path, summary_rtf_path]:
        if os.path.exists(path):
            shutil.chown(path, user="bigbluebutton", group="bigbluebutton")
            os.chmod(path, 0o755 if os.path.isdir(path) else 0o644)

    # 6. Register native published format directories
    print("[Pipeline] Registering native format directories...")
    register_custom_formats_native(
        meeting_id=meeting_id,
        base_published_dir=base_published_dir
    )

    print(f"[Pipeline] Finished processing {meeting_id} successfully.")

if __name__ == "__main__":
    main()        caller_name = event.findtext("callername")
        if p_id and caller_name:
            participants[p_id] = caller_name

    # 2. Find StartRecordingEvent timestampUTC (t0)
    start_rec_event = root.find(".//event[@module='VOICE'][@eventname='StartRecordingEvent']")
    if start_rec_event is None or start_rec_event.find("timestampUTC") is None:
        print("[Speaker Map] Warning: StartRecordingEvent missing in XML. Skipping speaker attribution.")
        return []

    t0_ms = float(start_rec_event.findtext("timestampUTC"))

    # 3. Extract talking intervals
    active_talkers = {}
    timeline = []

    for event in root.findall(".//event[@module='VOICE'][@eventname='ParticipantTalkingEvent']"):
        p_id = event.findtext("participant")
        is_talking = event.findtext("talking", "false").lower() == "true"
        ts_utc_node = event.find("timestampUTC")

        if not p_id or ts_utc_node is None:
            continue

        ts_ms = float(ts_utc_node.text)
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

    for p_id, start_sec in active_talkers.items():
        speaker_name = participants.get(p_id, "Unknown Speaker")
        timeline.append({"start": start_sec, "end": start_sec + 5.0, "name": speaker_name})

    print(f"[Speaker Map] Successfully mapped {len(timeline)} talking segments from XML.")
    return timeline


def get_speaker_for_timestamp(start_sec: float, end_sec: float, timeline: list) -> str:
    """Finds speaker active during [start_sec, end_sec] with fuzzy buffer."""
    if not timeline:
        return ""

    midpoint = (start_sec + end_sec) / 2.0
    for interval in timeline:
        if interval["start"] <= midpoint <= interval["end"]:
            return interval["name"]

    for interval in timeline:
        if not (end_sec + 1.5 < interval["start"] or start_sec - 1.5 > interval["end"]):
            return interval["name"]

    return ""

def get_best_active_cf_model(account_id: str, api_token: str) -> str:
    """Queries Cloudflare Workers AI model catalog dynamically."""
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
                                print(f"[Catalog] Selected Cloudflare model: {m}")
                                return m
                    return valid_models[0]
    except Exception as e:
        print(f"[Warning] Catalog query failed ({e}). Falling back to default model.")

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
        "You are an executive assistant analyzing speaker-attributed transcripts.\n\n"
        "GOAL: Produce a faithful, comprehensive summary that captures both main topics and subtle, critical side remarks, technical details, or quick decisions.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Do NOT include conversational intros or preambles. Start directly with '## Executive Summary'.\n"
        "2. COMPREHENSION & DETAIL:\n"
        "   - Treat all spoken content as potentially significant, including side comments, quick clarifications, or niche/technical points.\n"
        "   - Do NOT omit arcane terminology, specific proper nouns, or short off-hand statements—preserve them as stated.\n"
        "3. EXECUTIVE SUMMARY:\n"
        "   - Provide a high-level overview (2-4 sentences) summarizing the main purpose or core themes of the discussion.\n"
        "   - If the transcript consists purely of casual banter, summarize the general nature of the conversation instead of writing 'None'.\n"
        "4. KEY DISCUSSION POINTS:\n"
        "   - Detail specific topics, claims, observations, or brief tangential points made during the session.\n"
        "   - Attribute key observations to specific speakers when clear from the text.\n"
        "5. DECISIONS & ACTION ITEMS:\n"
        "   - List any explicit or implied decisions, agreements, or follow-ups.\n"
        "   - If truly no decisions or action items were stated, write 'None' under those specific headings."
    )

    user_prompt = f"Summarize the following speaker-tagged meeting transcript accurately:\n\n{transcript_text}"

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
            headers={"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
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

def register_custom_formats_native(
    meeting_id: str,
    base_published_dir: str = "/var/bigbluebutton/published"
):
    """
    Creates native published format directories for 'summary' and 'transcript'
    by reading (and NOT modifying) the primary presentation metadata.xml.
    """
    domain = get_server_domain()
    pres_dir = os.path.join(base_published_dir, "presentation", meeting_id)
    pres_meta_path = os.path.join(pres_dir, "metadata.xml")

    if not os.path.exists(pres_meta_path):
        print(f"[Metadata Error] Primary metadata file not found at {pres_meta_path}")
        return

    if os.path.getsize(pres_meta_path) == 0:
        print(f"[Metadata Error] Primary metadata file at {pres_meta_path} is empty.")
        return

    try:
        # READ ONLY: Parse primary presentation metadata.xml
        tree = ET.parse(pres_meta_path)
        root = tree.getroot()

        # Extract elements to copy into the new native format XMLs
        meeting_elem = root.find("meeting")
        meta_elem = root.find("meta")
        start_time = root.findtext("start_time", default="")
        end_time = root.findtext("end_time", default="")
        participants = root.findtext("participants", default="0")
        
        playback = root.find("playback")
        duration = playback.findtext("duration", default="0") if playback is not None else "0"

        formats = {
            "summary": f"{domain}/presentation/{meeting_id}/transcript_summary.txt",
            "transcript": f"{domain}/presentation/{meeting_id}/transcript.txt"
        }

        # Build published/summary/<meeting_id>/metadata.xml and published/transcript/<meeting_id>/metadata.xml
        for fmt_name, fmt_link in formats.items():
            fmt_dir = os.path.join(base_published_dir, fmt_name, meeting_id)
            os.makedirs(fmt_dir, exist_ok=True)

            new_root = ET.Element("recording")
            ET.SubElement(new_root, "id").text = meeting_id
            ET.SubElement(new_root, "state").text = "published"
            ET.SubElement(new_root, "published").text = "true"
            if start_time: ET.SubElement(new_root, "start_time").text = start_time
            if end_time: ET.SubElement(new_root, "end_time").text = end_time
            ET.SubElement(new_root, "participants").text = participants

            if meeting_elem is not None:
                new_root.append(meeting_elem)

            if meta_elem is not None:
                new_root.append(meta_elem)

            new_playback = ET.SubElement(new_root, "playback")
            ET.SubElement(new_playback, "format").text = fmt_name
            ET.SubElement(new_playback, "link").text = fmt_link
            ET.SubElement(new_playback, "processing_time").text = "0"
            ET.SubElement(new_playback, "duration").text = duration
            ET.SubElement(new_playback, "size").text = "0"

            meta_file_path = os.path.join(fmt_dir, "metadata.xml")
            new_tree = ET.ElementTree(new_root)
            
            # Atomic write strictly for the NEW format metadata file
            temp_meta_path = meta_file_path + ".tmp"
            ET.indent(new_tree, space="  ")
            new_tree.write(temp_meta_path, encoding="UTF-8", xml_declaration=True)
            shutil.move(temp_meta_path, meta_file_path)

            shutil.chown(fmt_dir, user="bigbluebutton", group="bigbluebutton")
            shutil.chown(meta_file_path, user="bigbluebutton", group="bigbluebutton")
            os.chmod(fmt_dir, 0o755)
            os.chmod(meta_file_path, 0o644)

            print(f"[Success] Created native format for '{fmt_name}' at {meta_file_path}")

    except ET.ParseError as pe:
        print(f"[Metadata Error] Malformed XML encountered in primary metadata: {pe}")
    except Exception as e:
        print(f"[Metadata Error] Failed to process {meeting_id}: {e}")

def get_server_domain() -> str:
    """
    Reads the BigBlueButton server URL directly from bbb-web.properties.
    Raises RuntimeError if the file or key is missing.
    """
    prop_path = "/etc/bigbluebutton/bbb-web.properties"
    
    if os.path.exists(prop_path):
        try:
            with open(prop_path, "r") as f:
                for line in f:
                    # Look for the serverURL property key
                    if line.strip().startswith("bigbluebutton.web.serverURL="):
                        url = line.split("=", 1)[1].strip()
                        if url:
                            return url.rstrip("/")
        except Exception as e:
            raise RuntimeError(f"Failed to read {prop_path}: {e}")

    raise RuntimeError(f"Could not resolve server URL. Key 'bigbluebutton.web.serverURL' missing or file not found at {prop_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 transcribe_meeting.py <meeting_id>")
        sys.exit(1)

    meeting_id = sys.argv[1]
    base_published_dir = "/var/bigbluebutton/published"
    
    # Standard BigBlueButton 3.0 directory paths constructed from meeting_id
    output_dir = os.path.join(base_published_dir, "presentation", meeting_id)
    events_xml_path = f"/var/bigbluebutton/recording/raw/{meeting_id}/events.xml"
    
    # Check for audio/video source file (check webcams.webm first, fallback to audio.webm if present)
    webm_path = os.path.join(output_dir, "video", "webcams.webm")
    if not os.path.exists(webm_path):
        fallback_audio = os.path.join(output_dir, "audio", "audio.webm")
        if os.path.exists(fallback_audio):
            webm_path = fallback_audio
        else:
            print(f"[Error] No audio/video WebM file found for meeting {meeting_id} under {output_dir}")
            sys.exit(1)

    print(f"--- Processing AI Pipeline for Recording: {meeting_id} ---")

    # 1. Parse speaker timeline from events.xml
    timeline = parse_bbb_speaker_timeline(events_xml_path) if os.path.exists(events_xml_path) else []

    # 2. Transcribe published WebM directly using faster-whisper
    print(f"[Whisper] Loading model 'large-v3-turbo'...")
    model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8", download_root="/var/bigbluebutton/hf_cache")

    print(f"[Whisper] Transcribing WebM: {webm_path}...")
    segments, _ = model.transcribe(webm_path, vad_filter=True, language="en")

    full_transcript = []
    vtt_lines = ["WEBVTT\n"]
    current_speaker = None

    # 3. Format WebVTT and TXT with speaker attribution
    for segment in segments:
        raw_text = segment.text.strip()

        # Filter out automated BBB voice prompt
        if "you are currently the only person" in raw_text.lower():
            continue

        speaker = get_speaker_for_timestamp(segment.start, segment.end, timeline)

        # --- TXT Formatting (Grouped by Speaker with Timecodes) ---
        # Format HH:MM:SS or MM:SS
        hrs = int(segment.start // 3600)
        mins = int((segment.start % 3600) // 60)
        secs = int(segment.start % 60)
        time_str = f"[{hrs:02d}:{mins:02d}:{secs:02d}]" if hrs > 0 else f"[{mins:02d}:{secs:02d}]"

        # Emit speaker header only when speaker changes
        if speaker and speaker != current_speaker:
            if full_transcript:
                full_transcript.append("")  # Blank line separator between speakers
            full_transcript.append(f"[{speaker}]")
            current_speaker = speaker

        # Append timecoded text line
        full_transcript.append(f"{time_str}: {raw_text}")

        # --- WebVTT Formatting (HH:MM:SS.mmm) ---
        vtt_prefix = f"[{speaker}]: " if speaker else ""
        vtt_text = f"{vtt_prefix}{raw_text}"

        start_fmt = f"{int(segment.start//3600):02d}:{int((segment.start%3600)//60):02d}:{segment.start%60:06.3f}"
        end_fmt = f"{int(segment.end//3600):02d}:{int((segment.end%3600)//60):02d}:{segment.end%60:06.3f}"

        vtt_lines.append(f"{start_fmt} --> {end_fmt}\n{vtt_text}\n")

    full_text = "\n".join(full_transcript)

    # 4. Save output files to published presentation directory
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "transcript.txt"), "w") as f:
        f.write(full_text + "\n")

    with open(os.path.join(output_dir, "transcript.vtt"), "w") as f:
        f.write("\n".join(vtt_lines) + "\n")

    # 5. Summarize via Cloudflare Workers AI
    print("[Summary] Generating AI meeting summary...")
    summary = generate_summary(full_text)

    with open(os.path.join(output_dir, "transcript_summary.txt"), "w") as f:
        f.write(summary + "\n")

    # 6. Register native published format directories
    print("[Pipeline] Registering native format directories...")
    register_custom_formats_native(
        meeting_id=meeting_id,
        base_published_dir=base_published_dir
    )

    print(f"[Pipeline] Finished processing {meeting_id} successfully.")

if __name__ == "__main__":
    main()            participants[p_id] = caller_name

    # 2. Find StartRecordingEvent timestampUTC (t0)
    start_rec_event = root.find(".//event[@module='VOICE'][@eventname='StartRecordingEvent']")
    if start_rec_event is None or start_rec_event.find("timestampUTC") is None:
        print("[Speaker Map] Warning: StartRecordingEvent missing in XML. Skipping speaker attribution.")
        return []

    t0_ms = float(start_rec_event.findtext("timestampUTC"))

    # 3. Extract talking intervals
    active_talkers = {}
    timeline = []

    for event in root.findall(".//event[@module='VOICE'][@eventname='ParticipantTalkingEvent']"):
        p_id = event.findtext("participant")
        is_talking = event.findtext("talking", "false").lower() == "true"
        ts_utc_node = event.find("timestampUTC")

        if not p_id or ts_utc_node is None:
            continue

        ts_ms = float(ts_utc_node.text)
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

    for p_id, start_sec in active_talkers.items():
        speaker_name = participants.get(p_id, "Unknown Speaker")
        timeline.append({"start": start_sec, "end": start_sec + 5.0, "name": speaker_name})

    print(f"[Speaker Map] Successfully mapped {len(timeline)} talking segments from XML.")
    return timeline


def get_speaker_for_timestamp(start_sec: float, end_sec: float, timeline: list) -> str:
    """Finds speaker active during [start_sec, end_sec] with fuzzy buffer."""
    if not timeline:
        return ""

    midpoint = (start_sec + end_sec) / 2.0
    for interval in timeline:
        if interval["start"] <= midpoint <= interval["end"]:
            return interval["name"]

    for interval in timeline:
        if not (end_sec + 1.5 < interval["start"] or start_sec - 1.5 > interval["end"]):
            return interval["name"]

    return ""


def get_best_active_cf_model(account_id: str, api_token: str) -> str:
    """Queries Cloudflare Workers AI model catalog dynamically."""
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
                                print(f"[Catalog] Selected Cloudflare model: {m}")
                                return m
                    return valid_models[0]
    except Exception as e:
        print(f"[Warning] Catalog query failed ({e}). Falling back to default model.")

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
        "1. Do NOT include conversational intros or preambles. Start directly with '## Executive Summary'.\n"
        "2. Do NOT inflate brief off-hand remarks or automated prompts into major discussion points.\n"
        "3. Attribute key points or action items to specific speakers if mentioned in the transcript.\n"
        "4. If a section has no meaningful content, explicitly write 'None'."
    )

    user_prompt = f"Summarize the following speaker-tagged meeting transcript accurately:\n\n{transcript_text}"

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
            headers={"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
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

def register_formats_in_metadata(output_dir: str, meeting_id: str):
    """
    Registers custom formats (summary and transcript) in metadata.xml
    following standard BigBlueButton <playback><format> schema so Greenlight
    and Scalelite can parse them automatically.
    """
    meta_path = os.path.join(output_dir, "metadata.xml")
    if not os.path.exists(meta_path):
        print(f"[Metadata] Warning: metadata.xml not found at {meta_path}")
        return

    try:
        tree = ET.parse(meta_path)
        root = tree.getroot()

        # Derive domain from existing playback link
        base_url = "https://bbb.spots.edu"
        playback_node = root.find("playback")
        if playback_node is not None:
            # Check existing format links first
            link_node = playback_node.find(".//url") or playback_node.find("link")
            if link_node is not None and link_node.text:
                match = re.match(r"(https?://[^/]+)", link_node.text)
                if match:
                    base_url = match.group(1)

        summary_url = f"{base_url}/presentation/{meeting_id}/transcript_summary.txt"
        transcript_url = f"{base_url}/presentation/{meeting_id}/transcript.txt"
        vtt_url = f"{base_url}/presentation/{meeting_id}/transcript.vtt"

        # 1. Update <meta> node for custom internal/API metadata querying
        meta_node = root.find("meta")
        if meta_node is None:
            meta_node = ET.SubElement(root, "meta")

        def set_or_update(parent, tag, value):
            node = parent.find(tag)
            if node is None:
                node = ET.SubElement(parent, tag)
            node.text = value

        set_or_update(meta_node, "summary-url", summary_url)
        set_or_update(meta_node, "transcript-url", transcript_url)
        set_or_update(meta_node, "caption-vtt-url", vtt_url)

        # 2. Add nested <format> blocks inside <playback>
        if playback_node is not None:
            # Collect existing format types (checking nested <type> tags)
            existing_types = []
            for fmt in playback_node.findall("format"):
                t_node = fmt.find("type")
                if t_node is not None and t_node.text:
                    existing_types.append(t_node.text)

            def add_format_block(fmt_type: str, fmt_url: str):
                if fmt_type not in existing_types:
                    fmt_elem = ET.SubElement(playback_node, "format")
                    
                    type_elem = ET.SubElement(fmt_elem, "type")
                    type_elem.text = fmt_type
                    
                    url_elem = ET.SubElement(fmt_elem, "url")
                    url_elem.text = fmt_url
                    
                    # Optional: Greenlight accepts length/processing time
                    len_elem = ET.SubElement(fmt_elem, "length")
                    len_elem.text = "0"

            add_format_block("summary", summary_url)
            add_format_block("transcript", transcript_url)

        # Format XML output neatly (Python 3.9+)
        if hasattr(ET, "indent"):
            ET.indent(tree, space="  ")

        tree.write(meta_path, encoding="UTF-8", xml_declaration=True)
        print("[Metadata] Successfully registered nested formats in metadata.xml.")

    except Exception as e:
        print(f"[Metadata] Error updating metadata.xml: {e}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 transcribe_meeting.py <webm_file_path> <output_dir> [events_xml_path]")
        sys.exit(1)

    webm_path = sys.argv[1]
    output_dir = sys.argv[2]
    events_xml_path = sys.argv[3] if len(sys.argv) > 3 else ""

    if not os.path.exists(webm_path):
        print(f"[Error] WebM file not found at: {webm_path}")
        sys.exit(1)

    # 1. Parse speaker timeline from events.xml
    timeline = parse_bbb_speaker_timeline(events_xml_path) if events_xml_path else []

    # 2. Transcribe published WebM directly using faster-whisper
    print(f"[Whisper] Loading model 'medium'...")
    model = WhisperModel("medium", device="cpu", compute_type="int8", download_root="/var/bigbluebutton/hf_cache")

    print(f"[Whisper] Transcribing published WebM: {webm_path}...")
    segments, _ = model.transcribe(webm_path, vad_filter=True, language="en")

    full_transcript = []
    vtt_lines = ["WEBVTT\n"]

    # 3. Format WebVTT and TXT with speaker attribution
    for segment in segments:
        speaker = get_speaker_for_timestamp(segment.start, segment.end, timeline)
        speaker_prefix = f"[{speaker}]: " if speaker else ""
        
        raw_text = segment.text.strip()
        
        # Filter out automated BBB voice prompt
        if "you are currently the only person" in raw_text.lower():
            continue

        text_line = f"{speaker_prefix}{raw_text}"
        full_transcript.append(text_line)

        # WebVTT timestamp formatting (HH:MM:SS.mmm)
        start_fmt = f"{int(segment.start//3600):02d}:{int((segment.start%3600)//60):02d}:{segment.start%60:06.3f}"
        end_fmt = f"{int(segment.end//3600):02d}:{int((segment.end%3600)//60):02d}:{segment.end%60:06.3f}"
        
        vtt_lines.append(f"{start_fmt} --> {end_fmt}\n{text_line}\n")

    full_text = "\n".join(full_transcript)

    # 4. Save output files to published directory
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "transcript.txt"), "w") as f:
        f.write(full_text + "\n")

    with open(os.path.join(output_dir, "transcript.vtt"), "w") as f:
        f.write("\n".join(vtt_lines) + "\n")

    # 5. Summarize via Cloudflare Workers AI
    print("[Summary] Generating AI meeting summary...")
    summary = generate_summary(full_text)

    with open(os.path.join(output_dir, "transcript_summary.txt"), "w") as f:
        f.write(summary + "\n")

    print("[Complete] Processing finished successfully.")

    # 6. Register formats in metadata.xml
    meeting_id = os.path.basename(os.path.normpath(output_dir))
    if os.path.exists(os.path.join(output_dir, "metadata.xml")):
        print(f"[Metadata] Updating metadata for meeting {meeting_id} at: {output_dir}")
        register_formats_in_metadata(output_dir, meeting_id)
    else:
        print(f"[Metadata] Warning: metadata.xml not found in {output_dir}. Skipping registration.")

if __name__ == "__main__":
    main()
