#!/usr/local/bigbluebutton/core/scripts/post_publish/lib/transcribe_meeting.py
import sys
import os
import shutil
import re
import json
import warnings
import xml.etree.ElementTree as ET
import markdown
from faster_whisper import WhisperModel
from huggingface_hub import hf_hub_download

# Suppress harmless HuggingFace deprecation UserWarnings in logs
warnings.filterwarnings("ignore", category=UserWarning)

_llama_import_error = None
try:
    from llama_cpp import Llama
except Exception as e:
    Llama = None
    _llama_import_error = str(e)


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


def _summarize_llama(llm, system_prompt: str, transcript_text: str, is_chunk: bool = False) -> str:
    """
    Internal helper: Summarize transcript using an already-loaded Llama model.
    is_chunk=True adjusts prompt for chunk summaries.
    Raises RuntimeError if context window is exceeded (caller can handle retry).
    """
    if is_chunk:
        user_prompt = f"Summarize this section of a meeting transcript (this is one section of a longer meeting):\n\n{transcript_text}"
    else:
        user_prompt = f"Summarize this meeting transcript:\n\n{transcript_text}"

    # Format prompt using Llama 2 Chat template
    formatted_prompt = (
        f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
    )

    response = llm(
        formatted_prompt,
        max_tokens=1024,
        temperature=0.7,
        top_p=0.95,
        stop=["[/INST]"]
    )

    raw_summary = response.get("choices", [{}])[0].get("text", "").strip()
    
    if not raw_summary:
        return ""

    # Clean up any preamble
    cleaned = re.sub(
        r"^(here is (a|the) summary[^\n]*\n?|sure[^\n]*\n?|certainly[^\n]*\n?)", 
        "", 
        raw_summary, 
        flags=re.IGNORECASE
    ).strip()
    
    return cleaned


def generate_summary(transcript_text: str) -> str:
    """
    Generates a meeting summary using a local Llama model.
    Automatically downloads the GGUF model from Hugging Face if not found locally (same as Whisper).
    For large transcripts, chunks and summarizes sections, then combines into final summary.
    """
    words = transcript_text.strip().split()
    if len(words) < 8:
        return (
            "## Executive Summary\n"
            "No meeting discussion recorded (empty transcript).\n\n"
            "## Key Discussion Points\nNone\n\n"
            "## Decisions Made\nNone\n\n"
            "## Action Items & Next Steps\nNone"
        )

    # Check for local Llama model
    if Llama is None:
        err_detail = f" ({_llama_import_error})" if _llama_import_error else ""
        return f"[Error: llama_cpp library not available{err_detail}. Install in venv: /opt/speech_env/bin/pip install llama-cpp-python]"

    # Calculate dynamic context window based on transcript size
    # Use 1.5 tokens per word (conservative for transcripts with speaker labels)
    # Add 15% safety margin to account for tokenizer variations
    transcript_tokens = int(len(words) * 1.5 * 1.15)
    system_prompt_tokens = 600  # System prompt overhead (conservative)
    response_buffer_tokens = 1024  # Space for model response
    format_overhead = 100  # Template wrapper + margins
    total_needed = transcript_tokens + system_prompt_tokens + response_buffer_tokens + format_overhead
    
    # Max safe context is 16384, but leave 2000-token buffer for safety
    MAX_CONTEXT = 16384
    SAFE_USABLE_CONTEXT = 14000
    needs_chunking = total_needed > SAFE_USABLE_CONTEXT
    
    # Round up to nearest power of 2 for efficient allocation, bounded 2048-16384
    n_ctx = 2048
    while n_ctx < total_needed and n_ctx < MAX_CONTEXT:
        n_ctx *= 2
    
    print(f"[Llama] Transcript estimate: {transcript_tokens} tokens → Using n_ctx={n_ctx}")
    if needs_chunking:
        print(f"[Llama] Transcript exceeds safe limit of {SAFE_USABLE_CONTEXT} tokens; will chunk and summarize sections.")

    model_path = os.environ.get("LLAMA_MODEL_PATH")
    hf_cache = os.environ.get("HF_HOME", "/var/bigbluebutton/hf_cache")

    try:
        if model_path and os.path.exists(model_path):
            print(f"[Llama] Loading model from path: {model_path}...")
            llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=4, verbose=False)
        else:
            # Use hf_hub_download to get cached path without re-downloading every time
            repo_id = os.environ.get("LLAMA_REPO_ID", "TheBloke/Mistral-7B-Instruct-v0.1-GGUF")
            filename = os.environ.get("LLAMA_FILENAME", "mistral-7b-instruct-v0.1.Q4_K_M.gguf")
            print(f"[Llama] Resolving model '{repo_id}' ({filename}) from cache or HuggingFace Hub...")
            
            # Download once (if missing), then always load from cache
            model_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=hf_cache,
                local_dir_use_symlinks=False
            )
            print(f"[Llama] Loading from cache: {model_file}...")
            llm = Llama(model_path=model_file, n_ctx=n_ctx, n_threads=4, verbose=False)

        system_prompt = (
            "You are an executive assistant analyzing speaker-attributed meeting transcripts.\n\n"
            "GOAL: Produce a faithful, comprehensive summary that captures both main topics and subtle, critical side remarks, technical details, or quick decisions.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Do NOT include conversational intros or preambles. Start directly with '## Executive Summary'.\n"
            "   - Do NOT use the terms 'the transcript' or 'the meeting transcript' but use the term 'meeting' or 'session' depending on the nature of the content.\n"
            "2. COMPREHENSION & DETAIL:\n"
            "   - Treat all spoken content as potentially significant, including side comments, quick clarifications, or niche/technical points.\n"
            "   - Do NOT omit arcane terminology, specific proper nouns, or short off-hand statements—preserve them as stated.\n"
            "3. EXECUTIVE SUMMARY:\n"
            "   - Provide a high-level overview (2-4 sentences) summarizing the main purpose or core themes of the discussion.\n"
            "   - If the meeting consists purely of casual banter, summarize the general nature of the conversation instead of writing 'None'.\n"
            "4. KEY DISCUSSION POINTS:\n"
            "   - Detail specific topics, claims, observations, or brief tangential points made during the session.\n"
            "   - Attribute key observations to specific speakers when clear from the text.\n"
            "5. DECISIONS & ACTION ITEMS:\n"
            "   - List any explicit or implied decisions, agreements, or follow-ups.\n"
            "   - If truly no decisions or action items were stated, write 'None' under those specific headings."
        )

        # Handle chunking if transcript is too large
        if needs_chunking:
            print(f"[Llama] Splitting transcript into chunks for processing...")
            
            # Conservative chunk size: ~6000 words (~10,350 tokens with 1.725 ratio + overhead)
            # This stays safely under the 14000-token usable context window
            chunk_size_words = 6000
            transcript_words = transcript_text.split()
            chunks = []
            
            for i in range(0, len(transcript_words), chunk_size_words):
                chunk = " ".join(transcript_words[i:i+chunk_size_words])
                chunks.append(chunk)
            
            print(f"[Llama] Processing {len(chunks)} chunks...")
            chunk_summaries = []
            current_chunk_size = chunk_size_words
            
            for idx, chunk in enumerate(chunks, 1):
                print(f"[Llama] Summarizing chunk {idx}/{len(chunks)}...")
                
                # Retry mechanism: if chunk exceeds context, automatically reduce size and resplit
                max_retries = 3
                retry_count = 0
                chunk_summary = None
                
                while retry_count < max_retries and chunk_summary is None:
                    try:
                        chunk_summary = _summarize_llama(llm, system_prompt, chunk, is_chunk=True)
                        if chunk_summary:
                            chunk_summaries.append(chunk_summary)
                            break
                    except Exception as e:
                        error_msg = str(e)
                        # Check if it's a context window error
                        if "exceed" in error_msg.lower() and "context" in error_msg.lower():
                            retry_count += 1
                            if retry_count < max_retries:
                                # Reduce chunk size by 30% and re-chunk
                                print(f"[Llama] ⚠️  Chunk {idx} exceeded context ({error_msg})")
                                print(f"[Llama] Reducing chunk size from {current_chunk_size} to {int(current_chunk_size * 0.7)} words (retry {retry_count}/{max_retries})...")
                                
                                current_chunk_size = int(current_chunk_size * 0.7)
                                
                                # Re-split remaining chunks with smaller size
                                chunk_words = chunk.split()
                                smaller_chunks = []
                                for j in range(0, len(chunk_words), current_chunk_size):
                                    smaller_chunk = " ".join(chunk_words[j:j+current_chunk_size])
                                    smaller_chunks.append(smaller_chunk)
                                
                                # Summarize smaller sub-chunks
                                for sub_idx, smaller_chunk in enumerate(smaller_chunks, 1):
                                    try:
                                        sub_summary = _summarize_llama(llm, system_prompt, smaller_chunk, is_chunk=True)
                                        if sub_summary:
                                            chunk_summaries.append(sub_summary)
                                    except Exception as sub_e:
                                        print(f"[Llama] Error processing sub-chunk {sub_idx}: {sub_e}")
                                
                                chunk_summary = "retry_complete"  # Mark as handled
                            else:
                                print(f"[Llama] ❌ Chunk {idx} failed after {max_retries} retries: {error_msg}")
                                chunk_summary = ""  # Skip this chunk on final failure
                        else:
                            # Non-context-window error, re-raise
                            raise
            
            # Combine chunk summaries and create final summary
            if not chunk_summaries:
                return "[Error: Failed to generate chunk summaries]"
            
            combined_summaries = "\n\n---\n\n".join(chunk_summaries)
            print(f"[Llama] Creating final summary from {len(chunk_summaries)} chunk summaries...")
            
            final_system_prompt = (
                "You are an executive assistant synthesizing summaries from multiple sections of a single meeting.\n\n"
                "GOAL: Combine these section summaries into one coherent, executive-level meeting summary.\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. Do NOT include conversational intros or preambles. Start directly with '## Executive Summary'.\n"
                "2. Eliminate redundancy while preserving all distinct topics, decisions, and action items.\n"
                "3. EXECUTIVE SUMMARY: Provide 2-4 sentences covering the main themes.\n"
                "4. KEY DISCUSSION POINTS: Consolidate key topics from all sections.\n"
                "5. DECISIONS & ACTION ITEMS: List all explicit/implied decisions and action items across the meeting.\n"
                "6. If no decisions or action items exist, write 'None' under those headings."
            )
            
            final_summary = _summarize_llama(llm, final_system_prompt, combined_summaries, is_chunk=False)
            return final_summary if final_summary else "[Error: Final summary synthesis failed]"
        
        # Standard single-pass summarization for smaller transcripts
        else:
            print("[Llama] Generating summary (single pass)...")
            summary = _summarize_llama(llm, system_prompt, transcript_text, is_chunk=False)
            return summary if summary else "[Error: Llama returned empty response]"

    except Exception as e:
        return f"[Error generating summary with Llama: {e}]"


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

    # 5. Summarize via local Llama model & save summary artifacts
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
    main()

