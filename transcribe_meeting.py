#!/usr/local/bigbluebutton/core/scripts/post_publish/lib/transcribe_meeting.py
import sys
import os
import shutil
import re
import json
import warnings
import datetime
import xml.etree.ElementTree as ET
import markdown
from faster_whisper import WhisperModel
from huggingface_hub import HfApi, hf_hub_download

# HARD LIMIT underlying C++ thread pools BEFORE importing llama_cpp
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

# Suppress harmless HuggingFace deprecation UserWarnings in logs
warnings.filterwarnings("ignore", category=UserWarning)

LLM_MODEL_FAMILY = os.environ.get("LLM_MODEL_FAMILY", "gemma-4").lower()

# Keep CPU usage bounded so transcription and summarization do not monopolize the server.
try:
    available_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
except Exception:
    available_cpus = os.cpu_count() or 1

# Cap default threads to 8 max for optimal llama.cpp SIMD execution
DEFAULT_CPU_THREADS = min(6, max(1, available_cpus - 4))

CPU_THREAD_LIMIT = int(os.environ.get("CPU_THREAD_LIMIT", str(DEFAULT_CPU_THREADS)))
LLM_NUM_THREADS = int(os.environ.get("LLM_NUM_THREADS", str(min(6, CPU_THREAD_LIMIT))))
WHISPER_CPU_THREADS = int(os.environ.get("WHISPER_CPU_THREADS", str(min(6, CPU_THREAD_LIMIT))))
LLM_N_BATCH = int(os.environ.get("LLM_N_BATCH", "1024"))
DEBUG_SUMMARY = os.environ.get("DEBUG_SUMMARY", "1").strip().lower() not in {"0", "false", "no", "off"}
CHUNK_SIZE_WORDS = int(os.environ.get("CHUNK_SIZE_WORDS", "7000"))
CHUNK_OVERLAP_WORDS = int(os.environ.get("CHUNK_OVERLAP_WORDS", "600"))
CHUNK_JSON_MAX_TOKENS = int(os.environ.get("CHUNK_JSON_MAX_TOKENS", "2200"))


def debug_log(message: str):
    if DEBUG_SUMMARY:
        print(f"[Debug] {message}")

def _safe_snippet(text: str, limit: int = 240) -> str:
    if not text:
        return "<empty>"
    return text[:limit].replace("\n", "\\n")


def _extract_first_chunk_preserve_lines(transcript_text: str, max_words: int) -> str:
    """Take the first chunk by word budget while preserving line structure and speaker labels."""
    if not transcript_text:
        return ""

    selected_lines = []
    used_words = 0
    for line in transcript_text.splitlines():
        line_words = len(line.split())
        if selected_lines and used_words + line_words > max_words:
            break
        selected_lines.append(line)
        used_words += line_words

    if not selected_lines:
        return transcript_text

    return "\n".join(selected_lines)


def _infer_group_source_label_counts(llm, transcript_text: str) -> dict[str, bool]:
    """Ask Gemma to identify which labels represent multi-speaker camera/room audio sources."""
    if not transcript_text.strip():
        return {}

    user_prompt = (
        "CRITICAL TASK: Identify whether each camera/audio label represents a SINGLE speaker or MULTIPLE speakers.\n\n"
        "KEY RULES:\n"
        "1. Camera or room labels (e.g., [SPOTS], [Room 101]) are audio SOURCES, not single individuals.\n"
        "2. Mark is_multi_speaker as TRUE if you hear dialogue or multiple distinct voices under that label.\n"
        "3. Mark is_multi_speaker as FALSE if it represents a single person's account or mic.\n\n"
        "Return STRICT JSON only in this shape:\n"
        "{\"labels\":[{\"label\":\"SPOTS\",\"is_multi_speaker\":true},{\"label\":\"Jason Finnie\",\"is_multi_speaker\":false}]}\n\n"
        "Transcript segment:\n"
        f"{transcript_text}"
    )

    formatted_prompt = _build_chat_prompt(
        "You classify camera and audio labels as either single-speaker or multi-speaker room sources.",
        user_prompt,
        LLM_MODEL_FAMILY,
    )
    response = llm(
        formatted_prompt,
        max_tokens=512,
        temperature=0.0,
        top_p=0.9,
        stop=["<end_of_turn>"],
    )

    raw_text = response.get("choices", [{}])[0].get("text", "").strip()
    debug_log(f"Group-count raw output snippet: {_safe_snippet(raw_text, 300)}")
    parsed = _safe_json_object(raw_text)
    
    multi_speaker_map = {}
    entries = parsed.get("labels", [])
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict):
                label = str(entry.get("label") or entry.get("name") or "").strip()
                is_multi = bool(entry.get("is_multi_speaker", False))
                if label:
                    multi_speaker_map[label] = is_multi

    debug_log(f"Group-count multi-speaker map: {multi_speaker_map}")
    return multi_speaker_map


_llama_import_error = None
try:
    from llama_cpp import Llama
except Exception as e:
    Llama = None
    _llama_import_error = str(e)


def generate_rtf_from_md(md_text: str, rtf_path: str, meeting_name: str = "", meeting_date: str = ""):
    """
    Parses Markdown block and inline formatting (headings, bold, italics, lists) 
    and converts it into a natively formatted RTF document with metadata headers.
    """
    def _escape_rtf(text: str) -> str:
        """Escapes RTF special characters and converts non-ASCII characters to RTF unicode sequences."""
        text = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
        res = []
        for char in text:
            codepoint = ord(char)
            if codepoint < 128:
                res.append(char)
            else:
                res.append(f"\\u{codepoint}?")
        return "".join(res)

    def _md_inline_to_rtf(text: str) -> str:
        """Converts inline Markdown (bold, italics, links) to native RTF formatting words."""
        # Strip markdown links [label](url) -> label
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        
        # Split by bold (**text** / __text__) and italics (*text* / _text_)
        parts = re.split(r"(\*\*.*?\*\*|\*.*?\*|__.*?__|_[^_]+_)", text)
        rtf_parts = []
        
        for part in parts:
            if (part.startswith("**") and part.endswith("**")) or (part.startswith("__") and part.endswith("__")):
                inner = _escape_rtf(part[2:-2])
                rtf_parts.append(f"\\b {inner}\\b0")
            elif (part.startswith("*") and part.endswith("*")) or (part.startswith("_") and part.endswith("_")):
                inner = _escape_rtf(part[1:-1])
                rtf_parts.append(f"\\i {inner}\\i0")
            else:
                rtf_parts.append(_escape_rtf(part))
                
        return "".join(rtf_parts)

    # RTF Header structure, font table, and color table
    rtf_lines = [
        "{\\rtf1\\ansi\\deff0",
        "{\\fonttbl{\\f0\\froman\\fcharset0 Gentium Book Plus;}{\\f1\\fnil\\fcharset0 Helvetica;}}",
        "{\\colortbl ;\\red43\\green108\\blue176;\\red113\\green128\\blue150;\\red26\\green32\\blue44;}",
        "\\viewkind4\\uc1\\paperw12240\\paperh15840\\margl1440\\margr1440\\margt1440\\margb1440",
        "\\f0\\fs24\\cf3"  # Default font size 12pt (fs24)
    ]

    # Add Document Title / Meeting Metadata Header
    if meeting_name:
        clean_name = _escape_rtf(meeting_name)
        rtf_lines.append(f"\\f1\\b\\fs36\\cf1 {clean_name}\\b0\\par")
    if meeting_date:
        clean_date = _escape_rtf(meeting_date)
        rtf_lines.append(f"\\f1\\fs20\\cf2 Date: {clean_date}\\fs24\\cf3\\par\\par")
    if meeting_name or meeting_date:
        rtf_lines.append("\\f0")  # Switch back to main body font

    # Parse Markdown blocks line-by-line
    for line in md_text.splitlines():
        clean_line = line.strip()

        if not clean_line:
            rtf_lines.append("\\par")
            continue

        # ## Heading 2
        if clean_line.startswith("## "):
            header_text = _md_inline_to_rtf(clean_line[3:])
            rtf_lines.append(f"\\sb200\\sa100\\b\\fs30\\cf1 {header_text}\\b0\\fs24\\cf3\\par")
        # ### Heading 3
        elif clean_line.startswith("### "):
            header_text = _md_inline_to_rtf(clean_line[4:])
            rtf_lines.append(f"\\sb150\\sa80\\b\\fs26\\cf1 {header_text}\\b0\\fs24\\cf3\\par")
        # Bullet Points (- item or * item)
        elif clean_line.startswith("- ") or clean_line.startswith("* "):
            item_text = _md_inline_to_rtf(clean_line[2:])
            rtf_lines.append(f"\\fi-360\\li720 \\bullet\\tab {item_text}\\par")
        # Regular Paragraph
        else:
            para_text = _md_inline_to_rtf(clean_line)
            rtf_lines.append(f"\\sa120 {para_text}\\par")

    rtf_lines.append("}")

    # Write generated RTF string to target file
    with open(rtf_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rtf_lines))

    print(f"[RTF] Successfully generated formatted RTF document at: {rtf_path}")


def generate_html_viewer(md_text: str, html_path: str, rtf_filename: str, meeting_name: str = "", meeting_date: str = ""):
    """
    Renders Markdown summary into a stylized HTML paper document with meeting metadata and downloadable RTF.
    """
    html_body = []
    
    if meeting_name or meeting_date:
        html_body.append("<div style='margin-bottom: 25px; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px;'>")
        if meeting_name:
            html_body.append(f"<h1 style='font-size: 28px; margin: 0 0 5px 0;'>{meeting_name}</h1>")
        if meeting_date:
            html_body.append(f"<p style='color: #718096; font-size: 16px; margin: 0;'><strong>Date:</strong> {meeting_date}</p>")
        html_body.append("</div>")

    in_list = False

    for line in md_text.splitlines():
        clean_line = line.strip()
        
        if clean_line.startswith("- ") or clean_line.startswith("* "):
            if not in_list:
                html_body.append("<ul>")
                in_list = True
            item_text = clean_line[2:]
            item_text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", item_text)
            html_body.append(f"  <li>{item_text}</li>")
            continue
        else:
            if in_list:
                html_body.append("</ul>")
                in_list = False

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
    <title>Meeting Summary - {meeting_name or "BBB Session"}</title>
    <style>
        @import url('https://fonts.bunny.net/css?family=gentium-book-plus:400,400i,700,700i');

        * {{ box-sizing: border-box; }}

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

        .btn-download:hover {{ background-color: #2c5282; }}

        h2 {{
            font-size: 26px;
            color: #1a202c;
            margin-top: 35px;
            margin-bottom: 15px;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 6px;
        }}

        h3 {{ font-size: 21px; color: #2d3748; margin-top: 25px; margin-bottom: 10px; }}
        p {{ margin-bottom: 18px; }}
        ul {{ margin-top: 5px; margin-bottom: 20px; padding-left: 25px; }}
        li {{ margin-bottom: 8px; }}
        strong {{ color: #1a202c; }}
    </style>
</head>
<body>
    <div class="paper-container">
        <div class="toolbar">
            <span class="meeting-title">Meeting Summary</span>
            <a href="{rtf_filename}" download class="btn-download">
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


def get_meeting_metadata(pres_dir: str) -> tuple[str, str]:
    """Reads BBB presentation metadata.xml to extract Room Name and Meeting Date."""
    pres_meta_path = os.path.join(pres_dir, "metadata.xml")
    room_name = "Meeting Session"
    meeting_date = datetime.date.today().strftime("%Y-%m-%d")

    if os.path.exists(pres_meta_path) and os.path.getsize(pres_meta_path) > 0:
        try:
            tree = ET.parse(pres_meta_path)
            root = tree.getroot()
            meeting_name_node = root.find(".//meetingName") or root.find(".//meta/meetingName")
            if meeting_name_node is not None and meeting_name_node.text:
                room_name = meeting_name_node.text.strip()

            start_time_node = root.find("start_time")
            if start_time_node is not None and start_time_node.text:
                ms = float(start_time_node.text.strip())
                dt = datetime.datetime.fromtimestamp(ms / 1000.0)
                meeting_date = dt.strftime("%Y-%m-%d")
        except Exception as e:
            print(f"[Metadata Warning] Could not parse metadata details: {e}")

    return room_name, meeting_date


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

    start_rec_event = root.find(".//event[@module='VOICE'][@eventname='StartRecordingEvent']")
    if start_rec_event is None or start_rec_event.find("timestampUTC") is None:
        print("[Speaker Map] Warning: StartRecordingEvent missing in XML. Skipping speaker attribution.")
        return []

    t0_ms = float(start_rec_event.findtext("timestampUTC"))

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


def _build_chat_prompt(system_prompt: str, user_prompt: str, model_family: str | None = None) -> str:
    """Build a model-specific chat prompt for the configured model family."""
    family = (model_family or LLM_MODEL_FAMILY or "gemma-4").lower()
    if family in {"gemma", "gemma-4"}:
        return (
            "<start_of_turn>system\n"
            f"{system_prompt}\n"
            "<end_of_turn>\n"
            "<start_of_turn>user\n"
            f"{user_prompt}\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

    return f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"


def _resolve_hf_model_file(cache_dir: str, filename_hint: str | None = None) -> str:
    """Resolve a downloadable GGUF/GGML file from Hugging Face for the configured family only."""
    family = (LLM_MODEL_FAMILY or "gemma-4").lower()
    
    if family in {"gemma", "gemma-4"}:
        search_query = "gemma-4-E4B-it gguf"
        default_candidates = [
            "gemma-4-E4B-it-Q4_K_M.gguf",
            "gemma-4-E4B-it-Q5_K_M.gguf",
            "gemma-4-E4B-it-Q8_0.gguf",
        ]
    else:
        raise RuntimeError(f"Unsupported LLM model family: {family}")

    candidate_filenames = []
    if filename_hint:
        candidate_filenames.append(filename_hint)
    for name in default_candidates:
        if name not in candidate_filenames:
            candidate_filenames.append(name)

    try:
        api = HfApi()
        models = list(api.list_models(search=search_query, limit=8))
    except Exception as exc:
        raise RuntimeError(f"Unable to search Hugging Face for {family} models: {exc}") from exc

    for model in models:
        repo_id = getattr(model, "id", None)
        if not repo_id:
            continue

        print(f"[LLM] Trying Hugging Face repo '{repo_id}'...")
        debug_log(f"Model filename candidates: {candidate_filenames}")
        for filename in candidate_filenames:
            try:
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=cache_dir,
                    local_dir_use_symlinks=False,
                    repo_type="model",
                )
                print(f"[LLM] Successfully downloaded/verified '{filename}' from '{repo_id}'.")
                return downloaded_path
            except Exception as exc:
                print(f"[LLM] File '{filename}' not found in '{repo_id}': {exc}")

    raise RuntimeError(
        f"Unable to resolve a downloadable GGUF/GGML model for family '{family}' using Hugging Face search."
    )

def _safe_json_object(text: str) -> dict:
    """Parse a JSON object from model output, tolerating code fences or leading prose."""
    if not text:
        return {}

    compact_keys = ("t", "d", "a", "o", "s", "x", "u")

    def _try_json_loads(blob: str) -> dict:
        try:
            parsed_blob = json.loads(blob)
            return parsed_blob if isinstance(parsed_blob, dict) else {}
        except Exception:
            pass

        fixed_blob = re.sub(r",\s*([}\]])", r"\1", blob)
        try:
            parsed_blob = json.loads(fixed_blob)
            return parsed_blob if isinstance(parsed_blob, dict) else {}
        except Exception:
            return {}

    def _extract_balanced_json_object(blob: str) -> dict:
        start = blob.find("{")
        while start != -1:
            depth = 0
            in_string = False
            escaping = False
            for idx in range(start, len(blob)):
                ch = blob[idx]
                if in_string:
                    if escaping:
                        escaping = False
                    elif ch == "\\":
                        escaping = True
                    elif ch == '"':
                        in_string = False
                    continue

                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = blob[start : idx + 1]
                        parsed_candidate = _try_json_loads(candidate)
                        if parsed_candidate:
                            return parsed_candidate
                        break
            start = blob.find("{", start + 1)
        return {}

    def _extract_balanced_json_array(blob: str, start_idx: int) -> str:
        depth = 0
        in_string = False
        escaping = False
        for idx in range(start_idx, len(blob)):
            ch = blob[idx]
            if in_string:
                if escaping:
                    escaping = False
                elif ch == "\\":
                    escaping = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return blob[start_idx : idx + 1]

        if depth > 0:
            return blob[start_idx:] + ("]" * depth)
        return ""

    def _extract_compact_key_arrays(blob: str) -> dict:
        extracted = {}
        for key in compact_keys:
            match = re.search(rf'"{key}"\s*:\s*\[', blob)
            if not match:
                continue

            array_start = match.end() - 1
            array_text = _extract_balanced_json_array(blob, array_start)
            if not array_text:
                continue

            parsed_array = None
            for candidate in (
                array_text,
                re.sub(r",\s*(\]|\})", r"\1", array_text),
            ):
                try:
                    maybe_array = json.loads(candidate)
                except Exception:
                    continue
                if isinstance(maybe_array, list):
                    parsed_array = maybe_array
                    break

            if parsed_array is not None:
                extracted[key] = parsed_array

        return extracted

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    parsed = _try_json_loads(cleaned)
    if parsed:
        compact_extracted = _extract_compact_key_arrays(cleaned)
        if compact_extracted:
            for key, value in compact_extracted.items():
                current = parsed.get(key)
                if not isinstance(current, list) or not current:
                    parsed[key] = value
        return parsed

    parsed = _extract_balanced_json_object(cleaned)
    if parsed:
        compact_extracted = _extract_compact_key_arrays(cleaned)
        if compact_extracted:
            for key, value in compact_extracted.items():
                current = parsed.get(key)
                if not isinstance(current, list) or not current:
                    parsed[key] = value
        return parsed

    parsed = _extract_balanced_json_object(text)
    if parsed:
        compact_extracted = _extract_compact_key_arrays(text)
        if compact_extracted:
            for key, value in compact_extracted.items():
                current = parsed.get(key)
                if not isinstance(current, list) or not current:
                    parsed[key] = value
        return parsed

    compact_extracted = _extract_compact_key_arrays(cleaned)
    if compact_extracted:
        return compact_extracted

    compact_extracted = _extract_compact_key_arrays(text)
    if compact_extracted:
        return compact_extracted

    return {}


def _normalize_compact_rows(rows: object) -> list:
    """Normalize compact schema rows across inconsistent wrapper nesting patterns."""
    if not isinstance(rows, list):
        return []

    def _unwrap_singletons(value: object) -> object:
        while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
            value = value[0]
        return value

    def _collect_row_lists(value: object) -> list:
        value = _unwrap_singletons(value)
        if not isinstance(value, list) or not value:
            return []

        if all(not isinstance(item, list) for item in value):
            return [value]

        collected = []
        for item in value:
            if isinstance(item, list):
                collected.extend(_collect_row_lists(item))
        return collected

    return _collect_row_lists(rows)


def build_context_register(meeting_state: dict) -> dict:
    """Keep recent state available for context during chunk updates."""
    limits = {
        "topics": 12,
        "decisions": 8,
        "action_items": 8,
        "open_loops": 12,
        "speaker_notes": 12,
        "cross_references": 8,
        "uncertain_inferences": 6,
    }
    return {
        field: list(meeting_state.get(field, [])[-limit:])
        for field, limit in limits.items()
    }


def summarize_chunk_to_state_delta(
    llm,
    system_prompt: str,
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
    context_register: dict,
    compact_mode: bool = False,
) -> dict:
    """Ask the model to emit a compact structured JSON delta for one transcript chunk."""
    register_sizes = {
        key: len(value or []) for key, value in context_register.items()
    }
    debug_log(
        f"Chunk {chunk_index}/{total_chunks}: words={len(chunk_text.split())}, chars={len(chunk_text)}, context_sizes={register_sizes}"
    )

    user_prompt = (
        f"You are updating a compact meeting-state memory for chunk {chunk_index}/{total_chunks} of a longer meeting transcript.\n\n"
        "Return STRICT JSON only. No prose, no markdown, no commentary.\n"
        "Return JSON on a single line (minified).\n\n"
        "Use only the keys that appear in the schema. If a section has nothing useful, return an empty list for that key.\n\n"
        "Output constraints (MANDATORY):\n"
        "- Keep output compact and under ~1800 tokens.\n"
        "- Each text should be concise and factual.\n\n"
        + (
            "Use compact schema with short keys:\n"
            "{\"t\":[[topic,speaker]],\"d\":[[decision,speaker]],\"a\":[[action,speaker,owner]],\"o\":[[open_loop,speaker]],\"s\":[[speaker,note]],\"x\":[[reference,speaker,uncertain]],\"u\":[[uncertain_text,reason]]}\n\n"
            if compact_mode
            else
            "Preferred compact schema:\n"
            "{\"t\":[[topic,speaker]],\"d\":[[decision,speaker]],\"a\":[[action,speaker,owner]],\"o\":[[open_loop,speaker]],\"s\":[[speaker,note]],\"x\":[[reference,speaker,uncertain]],\"u\":[[uncertain_text,reason]]}\n"
            "Legacy schema also accepted:\n"
            "{\"topics\":[],\"decisions\":[],\"action_items\":[],\"open_loops\":[],\"speaker_notes\":[],\"cross_references\":[],\"uncertain_inferences\":[]}\n\n"
        )
        +
        "Rules:\n"
        "- Extract only content clearly supported by this chunk.\n"
        "- Do not invent links between chunks unless explicitly supported by the chunk.\n"
        "- Record unresolved questions, deferred tasks, or pending reviews in open_loops.\n"
        "- If a reference is unclear, put it in uncertain_inferences.\n"
        "- The prior context register is only for resolving explicit references; do not repeat it as a new fact.\n"
        "- Prefer speaker-attributed and topic-attributed notes.\n"
        "- Preserve names, technical terms, and late references.\n"
        "- Keep entries concise and factual.\n\n"
        "Prior context register:\n"
        f"{json.dumps(context_register, ensure_ascii=False)}\n\n"
        "Chunk text:\n"
        f"{chunk_text}"
    )

    formatted_prompt = _build_chat_prompt(system_prompt, user_prompt, LLM_MODEL_FAMILY)
    response = llm(
        formatted_prompt,
        max_tokens=CHUNK_JSON_MAX_TOKENS,
        temperature=0.1,
        top_p=0.9,
        stop=["<end_of_turn>"],
    )

    choice = response.get("choices", [{}])[0]
    raw_text = choice.get("text", "").strip()
    finish_reason = str(choice.get("finish_reason", "")).lower()
    debug_log(f"Chunk {chunk_index}/{total_chunks} raw output snippet: {_safe_snippet(raw_text, 320)}")
    debug_log(f"Chunk {chunk_index}/{total_chunks} finish_reason={finish_reason}")
    parsed = _safe_json_object(raw_text)

    expected_fields = [
        "topics",
        "decisions",
        "action_items",
        "open_loops",
        "speaker_notes",
        "cross_references",
        "uncertain_inferences",
    ]

    if isinstance(parsed, dict) and any(key in parsed for key in ("t", "d", "a", "o", "s", "x", "u")):
        normalized = {
            "topics": [],
            "decisions": [],
            "action_items": [],
            "open_loops": [],
            "speaker_notes": [],
            "cross_references": [],
            "uncertain_inferences": [],
        }

        for item in _normalize_compact_rows(parsed.get("t", [])):
            normalized["topics"].append({"text": str(item[0]), "speaker": str(item[1]) if len(item) > 1 else ""})
        for item in _normalize_compact_rows(parsed.get("d", [])):
            normalized["decisions"].append({"text": str(item[0]), "speaker": str(item[1]) if len(item) > 1 else ""})
        for item in _normalize_compact_rows(parsed.get("a", [])):
            normalized["action_items"].append({
                "text": str(item[0]),
                "speaker": str(item[1]) if len(item) > 1 else "",
                "owner": str(item[2]) if len(item) > 2 else "",
            })
        for item in _normalize_compact_rows(parsed.get("o", [])):
            normalized["open_loops"].append({"text": str(item[0]), "speaker": str(item[1]) if len(item) > 1 else ""})
        for item in _normalize_compact_rows(parsed.get("s", [])):
            normalized["speaker_notes"].append({"speaker": str(item[0]), "note": str(item[1]) if len(item) > 1 else ""})
        for item in _normalize_compact_rows(parsed.get("x", [])):
            normalized["cross_references"].append({
                "text": str(item[0]),
                "speaker": str(item[1]) if len(item) > 1 else "",
                "uncertain": bool(item[2]) if len(item) > 2 else False,
            })
        for item in _normalize_compact_rows(parsed.get("u", [])):
            normalized["uncertain_inferences"].append({"text": str(item[0]), "reason": str(item[1]) if len(item) > 1 else "unclear"})

        debug_log(
            "Normalized compact counts="
            + str({k: len(v) for k, v in normalized.items()})
        )

        parsed = normalized

    delta = {field: [] for field in expected_fields}
    parse_ok = any(field in parsed and isinstance(parsed.get(field), list) for field in expected_fields)
    if finish_reason == "length" and not parse_ok:
        debug_log(f"Chunk {chunk_index}/{total_chunks} likely truncated by token limit.")
    for field in expected_fields:
        value = parsed.get(field, [])
        if isinstance(value, list):
            delta[field] = value

    delta["_parse_ok"] = parse_ok
    debug_log(
        "Chunk "
        f"{chunk_index}/{total_chunks} parse_ok={parse_ok}, "
        f"counts={{" + ", ".join(f"{k}:{len(delta[k])}" for k in expected_fields) + "}}"
    )

    if not parse_ok:
        snippet = _safe_snippet(raw_text, 240)
        print(f"[Gemma] JSON parse failed for chunk {chunk_index}/{total_chunks}. Output snippet: {snippet}")
        delta["uncertain_inferences"] = [{
            "text": "Chunk could not be parsed into structured state.",
            "reason": "malformed or truncated model output"
        }]

    return delta


def merge_meeting_state(meeting_state: dict, delta: dict) -> dict:
    """Merge a chunk-level JSON delta into a unified meeting state without string-filtering heuristics."""
    merged = {
        key: list(value or []) for key, value in meeting_state.items()
    }
    before_sizes = {key: len(value or []) for key, value in merged.items()}

    for field in [
        "topics",
        "decisions",
        "action_items",
        "open_loops",
        "speaker_notes",
        "cross_references",
        "uncertain_inferences",
    ]:
        entries = delta.get(field, []) or []
        if not isinstance(entries, list):
            continue

        existing_signatures = {
            json.dumps(existing, sort_keys=True)
            for existing in merged.get(field, [])
        }
        for entry in entries:
            if not isinstance(entry, dict):
                entry = {"text": str(entry)}
            if field == "cross_references" and "uncertain" not in entry:
                entry["uncertain"] = False
            if field == "uncertain_inferences" and "reason" not in entry:
                entry["reason"] = "unclear"
            if field == "speaker_notes" and "note" not in entry and "text" in entry:
                entry["note"] = entry["text"]
            signature = json.dumps(entry, sort_keys=True)
            if signature not in existing_signatures:
                merged[field].append(entry)
                existing_signatures.add(signature)

    after_sizes = {key: len(value or []) for key, value in merged.items()}
    debug_log(f"State merge sizes before={before_sizes} after={after_sizes}")

    return merged


def prioritize_meeting_state(llm, full_meeting_state: dict, max_items_per_category: int = 12) -> dict:
    """Uses Gemma to evaluate, rank, and prioritize accumulated state entries across all chunks."""
    total_items = sum(len(v) for v in full_meeting_state.values() if isinstance(v, list))
    if total_items <= 15:
        return full_meeting_state

    user_prompt = (
        "You are an executive assistant distilling raw accumulated notes from a meeting.\n"
        "Your task is to RANK and DISTILL the following JSON meeting state down to the MOST IMPORTANT items across the entire meeting.\n\n"
        "CRITERIA FOR IMPORTANCE:\n"
        "- High Conversational Weight: Major administrative reports, core policy discussions, high-impact decisions.\n"
        "- Actionability: Explicitly assigned tasks, official follow-ups, or pending administrative loops.\n"
        "- Contextual & Domain Value: Preserve specific accreditation, system logistics, or organizational updates regardless of which chunk they occurred in.\n"
        "- Omit minor off-hand chatter, duplicates, or trivial comments.\n\n"
        "OUTPUT INSTRUCTIONS:\n"
        "- Return STRICT JSON matching the original keys (topics, decisions, action_items, open_loops, speaker_notes, cross_references, uncertain_inferences).\n"
        f"- Retain up to {max_items_per_category} top-priority items per category array.\n"
        "- Do not alter exact names, assigned roles, or facts.\n\n"
        "Raw Accumulated Meeting State:\n"
        f"{json.dumps(full_meeting_state, ensure_ascii=False)}"
    )

    formatted_prompt = _build_chat_prompt(
        "You distill and prioritize meeting state JSON based on importance and conversational weight.",
        user_prompt,
        LLM_MODEL_FAMILY,
    )

    response = llm(
        formatted_prompt,
        max_tokens=1800,
        temperature=0.1,
        top_p=0.9,
        stop=["<end_of_turn>"],
    )

    raw_text = response.get("choices", [{}])[0].get("text", "").strip()
    debug_log(f"Prioritize raw output snippet: {_safe_snippet(raw_text, 300)}")
    prioritized_state = _safe_json_object(raw_text)

    if isinstance(prioritized_state, dict) and any(k in prioritized_state for k in full_meeting_state.keys()):
        for key in full_meeting_state.keys():
            if key not in prioritized_state or not isinstance(prioritized_state[key], list):
                prioritized_state[key] = full_meeting_state[key][:max_items_per_category]
        return prioritized_state

    return full_meeting_state


def _state_texts(items: list) -> list:
    """Recursively extracts all meaningful text values from arbitrary state dictionaries."""
    texts = []
    for item in items or []:
        if isinstance(item, dict):
            texts.extend([str(v).strip() for v in item.values() if v and not isinstance(v, (dict, list, bool))])
        elif item:
            texts.append(str(item).strip())
    return [t for t in texts if t]


def synthesize_final_summary(llm, system_prompt: str, meeting_state: dict) -> str:
    """Turn the intelligently prioritized meeting state into the final markdown summary."""
    compact_sizes = {key: len(value or []) for key, value in meeting_state.items()}
    debug_log(f"Final synthesis state sizes: {compact_sizes}")

    state_json = json.dumps(meeting_state, ensure_ascii=False, indent=2)
    user_prompt = (
        "Convert the following structured meeting state into a concise Markdown summary.\n\n"
        "Return only Markdown with these headings:\n"
        "## Executive Summary\n"
        "## Key Discussion Points\n"
        "## Decisions Made\n"
        "## Action Items & Next Steps\n\n"
        "Rules:\n"
        "- Base the summary strictly on the provided structured state.\n"
        "- Do not invent facts. If details are missing, use 'None' or note that details are uncertain.\n"
        "- Do not use the phrase 'Open Loops'; describe unresolved items as follow-ups.\n"
        "- Preserve names, technical terms, and early/late references accurately.\n"
        "- Keep the summary factual, structured, and comprehensive.\n\n"
        "Structured meeting state:\n"
        f"{state_json}"
    )

    formatted_prompt = _build_chat_prompt(system_prompt, user_prompt, LLM_MODEL_FAMILY)
    response = llm(
        formatted_prompt,
        max_tokens=1400,
        temperature=0.3,
        top_p=0.9,
        stop=["<end_of_turn>"],
    )

    raw_summary = response.get("choices", [{}])[0].get("text", "").strip()
    debug_log(f"Final synthesis output length={len(raw_summary)} snippet={_safe_snippet(raw_summary, 320)}")
    if raw_summary:
        return raw_summary

    topics_list = _state_texts(meeting_state.get("topics", []))
    decisions_list = _state_texts(meeting_state.get("decisions", []))
    actions_list = _state_texts(meeting_state.get("action_items", []))
    open_loops_list = _state_texts(meeting_state.get("open_loops", []))
    speaker_notes_list = _state_texts(meeting_state.get("speaker_notes", []))
    uncertain_list = _state_texts(meeting_state.get("uncertain_inferences", []))

    topics = "; ".join(topics_list) or "None"

    key_points = []
    key_points.extend(f"- {topic}" for topic in topics_list)
    key_points.extend(f"- {note}" for note in speaker_notes_list)
    if not key_points:
        key_points.append("- None")

    decision_lines = [f"- {item}" for item in decisions_list] or ["- None"]
    action_lines = [f"- {item}" for item in actions_list] or ["- None"]
    if open_loops_list:
        action_lines.append("- Follow-ups: " + "; ".join(open_loops_list))
    if uncertain_list:
        action_lines.append("- Uncertain references: " + "; ".join(uncertain_list))

    return (
        "## Executive Summary\n"
        f"The meeting focused on {topics}.\n\n"
        "## Key Discussion Points\n"
        + "\n".join(key_points)
        + "\n\n"
        "## Decisions Made\n"
        + "\n".join(decision_lines)
        + "\n\n"
        "## Action Items & Next Steps\n"
        + "\n".join(action_lines)
    )


def generate_summary(transcript_text: str) -> str:
    """
    Generates a meeting summary using a local Llama model.
    Automatically downloads the GGUF model from Hugging Face if not found locally.
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

    if Llama is None:
        err_detail = f" ({_llama_import_error})" if _llama_import_error else ""
        return f"[Error: llama_cpp library not available{err_detail}. Install in venv: /opt/speech_env/bin/pip install llama-cpp-python]"

    transcript_tokens = int(len(words) * 1.5 * 1.25)
    system_prompt_tokens = 800
    response_buffer_tokens = 1200
    format_overhead = 200
    total_needed = transcript_tokens + system_prompt_tokens + response_buffer_tokens + format_overhead
    debug_log(f"Transcript words={len(words)} token_estimate={transcript_tokens} total_needed={total_needed}")

    MAX_CONTEXT = 32768
    SAFE_USABLE_CONTEXT = max(1024, int(MAX_CONTEXT * 0.85))
    needs_chunking = total_needed > SAFE_USABLE_CONTEXT
    n_ctx = min(MAX_CONTEXT, max(4096, total_needed))

    print(f"[Gemma] Transcript estimate: {transcript_tokens} tokens → Using n_ctx={n_ctx} (max context {MAX_CONTEXT})")
    if needs_chunking:
        print(f"[Gemma] Transcript exceeds safe limit of {SAFE_USABLE_CONTEXT} tokens; will chunk and summarize sections.")

    model_path = os.environ.get("LLAMA_MODEL_PATH")
    hf_cache = os.environ.get("HF_HOME", "/var/bigbluebutton/hf_cache")

    try:
        if model_path and os.path.exists(model_path):
            print(f"[LLM] Loading model from path: {model_path}...")
            model_file = model_path
        else:
            filename = os.environ.get("LLM_MODEL_FILENAME") or os.environ.get("LLAMA_FILENAME")
            print(f"[LLM] Resolving model from Hugging Face search...")
            model_file = _resolve_hf_model_file(hf_cache, filename)
            print(f"[LLM] Loading from cache: {model_file}...")

        llm = Llama(
            model_path=model_file,
            n_ctx=n_ctx,
            n_batch=LLM_N_BATCH,
            n_threads=LLM_NUM_THREADS,
            n_threads_batch=LLM_NUM_THREADS,
            flash_attn=False,
            verbose=False,
        )
        debug_log(f"Llama init params: n_ctx={n_ctx}, n_batch={LLM_N_BATCH}, n_threads={LLM_NUM_THREADS}, n_threads_batch={LLM_NUM_THREADS}")

        chunk_size_words = max(1000, CHUNK_SIZE_WORDS)
        first_chunk_text = _extract_first_chunk_preserve_lines(transcript_text, chunk_size_words)
        debug_log(
            f"Group-count chunk1 preserved shape: lines={len(first_chunk_text.splitlines())}, words={len(first_chunk_text.split())}"
        )
        group_label_counts = _infer_group_source_label_counts(llm, first_chunk_text)

        # Dynamic room/source guide without mutating the transcript text
        multi_speaker_labels = [k for k, is_multi in group_label_counts.items() if is_multi]
        if multi_speaker_labels:
            room_attribution_guide = (
                f"\nAUDIO SOURCE CONTEXT:\n"
                f"- MULTI-SPEAKER / ROOM SOURCES: {', '.join(f'[{label}]' for label in multi_speaker_labels)}\n"
                f"  These labels represent shared room setups or group microphones with multiple attendees.\n"
                f"  Treat dialogue under these labels as in-room/group discussions rather than attributing statements to a single person named after the label.\n"
                f"- SINGLE-SPEAKER SOURCES: All other user tags represent individual participants.\n"
            )
        else:
            room_attribution_guide = ""

        system_prompt = (
            "You are an executive assistant analyzing speaker-attributed transcripts from an Orthodox Christian seminary.\n\n"
            f"{room_attribution_guide}\n"
            "GOAL: Produce a faithful, comprehensive summary that captures both main topics and subtle, critical side remarks, technical details, or quick decisions.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Do NOT include conversational intros or preambles. Start directly with '## Executive Summary'.\n"
            "   - Do NOT use the terms 'the transcript' or 'the meeting transcript'—refer to it as the 'meeting', 'class', or 'session'.\n"
            "2. COMPREHENSION & DETAIL:\n"
            "   - Treat all spoken content as potentially significant, including side comments, quick clarifications, or niche/technical points.\n"
            "   - Do NOT omit specialized theological terminology, specific proper nouns, or short off-hand statements—preserve them as stated.\n"
            "3. EXECUTIVE SUMMARY:\n"
            "   - Provide a high-level overview (2-4 sentences) summarizing the main purpose or core themes of the discussion.\n"
            "   - If the meeting consists purely of casual banter, summarize the general nature of the conversation instead of writing 'None'.\n"
            "4. KEY DISCUSSION POINTS:\n"
            "   - Detail specific topics, claims, observations, or brief tangential points made during the session.\n"
            "   - Focus on the substance of what was discussed, attributing observations to named individuals when clear.\n"
            "5. DECISIONS & ACTION ITEMS:\n"
            "   - List any explicit or implied decisions, agreements, assignments, or follow-ups.\n"
            "   - If truly no decisions or action items were stated, write 'None' under those specific headings."
        )

        chunk_overlap_words = max(0, min(chunk_size_words - 1, CHUNK_OVERLAP_WORDS))
        chunk_stride_words = max(1, chunk_size_words - chunk_overlap_words)
        transcript_words = transcript_text.split()
        chunks = []
        for i in range(0, len(transcript_words), chunk_stride_words):
            chunk_words = transcript_words[i:i + chunk_size_words]
            if not chunk_words:
                continue
            chunks.append(" ".join(chunk_words))
            if i + chunk_size_words >= len(transcript_words):
                break

        if not chunks:
            chunks = [transcript_text]

        if needs_chunking:
            print(f"[Gemma] Processing {len(chunks)} chunks into structured meeting state...")
        else:
            print("[Gemma] Processing transcript into structured meeting state...")
        debug_log(
            f"Chunk plan: chunk_size_words={chunk_size_words}, overlap_words={chunk_overlap_words}, stride_words={chunk_stride_words}, chunk_count={len(chunks)}, chunk_word_counts={[len(c.split()) for c in chunks[:6]]}"
        )

        meeting_state = {
            "topics": [],
            "decisions": [],
            "action_items": [],
            "open_loops": [],
            "speaker_notes": [],
            "cross_references": [],
            "uncertain_inferences": [],
        }

        for idx, chunk in enumerate(chunks, 1):
            print(f"[Gemma] Updating state from chunk {idx}/{len(chunks)}...")
            try:
                context_register = build_context_register(meeting_state)
                delta = summarize_chunk_to_state_delta(
                    llm,
                    system_prompt,
                    chunk,
                    idx,
                    len(chunks),
                    context_register,
                )
                if not delta.get("_parse_ok", False):
                    debug_log(f"Chunk {idx} parse failed; retrying same chunk in compact mode before splitting.")
                    compact_retry = summarize_chunk_to_state_delta(
                        llm,
                        system_prompt,
                        chunk,
                        idx,
                        len(chunks),
                        context_register,
                        compact_mode=True,
                    )
                    if compact_retry.get("_parse_ok", False):
                        meeting_state = merge_meeting_state(meeting_state, compact_retry)
                        continue

                    print(f"[Gemma] Chunk {idx} returned no usable JSON; retrying with a smaller slice.")
                    sub_words = chunk.split()
                    if len(sub_words) > 2500:
                        quarter = max(800, len(sub_words) // 4)
                        retry_chunks = [
                            " ".join(sub_words[0 : 2 * quarter]),
                            " ".join(sub_words[2 * quarter :]),
                        ]
                    elif len(sub_words) > 1400:
                        retry_chunks = [
                            " ".join(sub_words[: len(sub_words) // 2]),
                            " ".join(sub_words[len(sub_words) // 2 :]),
                        ]
                    else:
                        retry_chunks = [chunk]

                    for sub_idx, sub_chunk in enumerate(retry_chunks, 1):
                        try:
                            debug_log(f"Retrying chunk {idx}.{sub_idx} with words={len(sub_chunk.split())}")
                            sub_delta = summarize_chunk_to_state_delta(
                                llm,
                                system_prompt,
                                sub_chunk,
                                idx,
                                len(chunks),
                                context_register,
                                compact_mode=True,
                            )
                            meeting_state = merge_meeting_state(meeting_state, sub_delta)
                        except Exception as sub_exc:
                            print(f"[Gemma] Retry chunk {idx}.{sub_idx} failed: {sub_exc}")
                    continue
                meeting_state = merge_meeting_state(meeting_state, delta)
            except Exception as exc:
                print(f"[Gemma] Error processing chunk {idx}: {exc}")
                meeting_state["uncertain_inferences"].append({
                    "text": "Chunk processing failed.",
                    "reason": str(exc)
                })

        debug_log(f"Final merged meeting_state sizes before ranking: {{k: len(v) for k, v in meeting_state.items()}}")

        ranked_meeting_state = prioritize_meeting_state(llm, meeting_state, max_items_per_category=12)
        debug_log(f"Ranked state sizes after LLM prioritization: {{k: len(v) for k, v in ranked_meeting_state.items()}}")

        final_system_prompt = (
            "You are an executive assistant synthesizing a structured meeting state for an Orthodox Christian seminary meeting or class.\n\n"
            f"{room_attribution_guide}\n"
            "GOAL: Turn the compacted meeting memory into one coherent, executive-level summary.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Do NOT include conversational intros or preambles. Start directly with '## Executive Summary'.\n"
            "   - Do NOT use the terms 'the transcript' or 'the meeting transcript'—refer to it as the 'meeting', 'class', or 'session'.\n"
            "2. Preserve the most important theological, academic, and administrative topics, decisions, and action items from the state.\n"
            "   - Do not use the phrase 'Open Loops'; describe unresolved items as follow-ups.\n"
            "3. EXECUTIVE SUMMARY: Provide 2-4 sentences covering the main themes.\n"
            "4. KEY DISCUSSION POINTS: Consolidate the core topics.\n"
            "5. DECISIONS & ACTION ITEMS: List all explicit/implied decisions and action items.\n"
            "6. If no decisions or action items exist, write 'None' under those headings."
        )

        final_summary = synthesize_final_summary(llm, final_system_prompt, ranked_meeting_state)
        debug_log(f"Final summary length={len(final_summary)}")
        return final_summary if final_summary else "[Error: Final summary synthesis failed]"

    except Exception as e:
        return f"[Error generating summary with Gemma: {e}]"


def register_custom_formats_native(
    meeting_id: str,
    base_published_dir: str = "/var/bigbluebutton/published"
):
    """
    Creates native published format directories for 'summary' and 'transcript'
    by reading (and NOT modifying) the primary presentation metadata.xml.
    """
    domain = get_server_domain()
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
        tree = ET.parse(pres_meta_path)
        root = tree.getroot()

        meeting_elem = root.find("meeting")
        meta_elem = root.find("meta")
        start_time = root.findtext("start_time", default="")
        end_time = root.findtext("end_time", default="")
        participants = root.findtext("participants", default="0")
        
        playback = root.find("playback")
        duration = playback.findtext("duration", default="0") if playback is not None else "0"

        formats = {
            "summary": f"{domain}/presentation/{meeting_id}/summary/summary.html",
            "transcript": f"{domain}/presentation/{meeting_id}/transcript/transcript.txt"
        }

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
    argv = sys.argv[1:]
    if not argv:
        print("Usage: python3 transcribe_meeting.py <meeting_id> [--summary-only] [--transcript-only] [--transcript-file PATH]")
        sys.exit(1)

    meeting_id = None
    summary_only = False
    transcript_only = False
    transcript_input_path = None

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--summary-only":
            summary_only = True
        elif arg == "--transcript-only":
            transcript_only = True
        elif arg == "--transcript-file":
            if i + 1 >= len(argv):
                print("[Error] --transcript-file requires a file path argument.")
                sys.exit(1)
            transcript_input_path = argv[i + 1]
            i += 1
        elif arg.startswith("--"):
            print(f"[Error] Unknown argument: {arg}")
            sys.exit(1)
        elif meeting_id is None:
            meeting_id = arg
        else:
            print(f"[Error] Unexpected positional argument: {arg}")
            sys.exit(1)
        i += 1

    if not meeting_id:
        print("Usage: python3 transcribe_meeting.py <meeting_id> [--summary-only] [--transcript-only] [--transcript-file PATH]")
        sys.exit(1)

    base_published_dir = "/var/bigbluebutton/published"
    
    pres_dir = os.path.join(base_published_dir, "presentation", meeting_id)
    events_xml_path = f"/var/bigbluebutton/recording/raw/{meeting_id}/events.xml"
    
    transcript_dir = os.path.join(pres_dir, "transcript")
    summary_dir = os.path.join(pres_dir, "summary")
    
    os.makedirs(transcript_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    print(f"--- Processing AI Pipeline for Recording: {meeting_id} ---")

    full_text = ""

    if summary_only:
        print("[Summary] Summary-only mode enabled; skipping audio transcription.")
        if transcript_input_path:
            if not os.path.exists(transcript_input_path):
                print(f"[Error] Transcript file not found: {transcript_input_path}")
                sys.exit(1)
            with open(transcript_input_path, "r", encoding="utf-8") as handle:
                full_text = handle.read().strip()
            print(f"[Summary] Loaded transcript from {transcript_input_path}")
        else:
            default_transcript_path = os.path.join(transcript_dir, "transcript.txt")
            if os.path.exists(default_transcript_path):
                with open(default_transcript_path, "r", encoding="utf-8") as handle:
                    full_text = handle.read().strip()
                print(f"[Summary] Loaded existing transcript from {default_transcript_path}")
            else:
                print(f"[Error] No existing transcript found at {default_transcript_path}. Pass --transcript-file PATH to provide one.")
                sys.exit(1)
    else:
        webm_path = os.path.join(pres_dir, "video", "webcams.webm")
        if not os.path.exists(webm_path):
            fallback_audio = os.path.join(pres_dir, "audio", "audio.webm")
            if os.path.exists(fallback_audio):
                webm_path = fallback_audio
            else:
                print(f"[Error] No audio/video WebM file found for meeting {meeting_id} under {pres_dir}")
                sys.exit(1)

        timeline = parse_bbb_speaker_timeline(events_xml_path) if os.path.exists(events_xml_path) else []

        print(f"[Whisper] Loading model 'large-v3-turbo'...")
        model = WhisperModel(
            "large-v3-turbo",
            device="cpu",
            compute_type="int8",
            cpu_threads=WHISPER_CPU_THREADS,
            download_root="/var/bigbluebutton/hf_cache",
        )

        print(f"[Whisper] Transcribing WebM: {webm_path}...")
        segments, _ = model.transcribe(
            webm_path, 
            vad_filter=True, 
            language="en",
            initial_prompt=(
                "Orthodox Christian seminary academic lecture, class, or faculty meeting. "
                "Discussion covers theological studies, coursework, liturgy, faculty, "
                "and administrative policies."
            )
        )

        full_transcript = []
        vtt_lines = ["WEBVTT\n"]
        current_speaker = None

        for segment in segments:
            raw_text = segment.text.strip()

            if "you are currently the only person" in raw_text.lower():
                continue

            speaker = get_speaker_for_timestamp(segment.start, segment.end, timeline)

            hrs = int(segment.start // 3600)
            mins = int((segment.start % 3600) // 60)
            secs = int(segment.start % 60)
            time_str = f"[{hrs:02d}:{mins:02d}:{secs:02d}]" if hrs > 0 else f"[{mins:02d}:{secs:02d}]"

            if speaker and speaker != current_speaker:
                if full_transcript:
                    full_transcript.append("")
                full_transcript.append(f"[{speaker}]")
                current_speaker = speaker

            full_transcript.append(f"{time_str}: {raw_text}")

            vtt_prefix = f"[{speaker}]: " if speaker else ""
            vtt_text = f"{vtt_prefix}{raw_text}"

            start_fmt = f"{int(segment.start//3600):02d}:{int((segment.start%3600)//60):02d}:{segment.start%60:06.3f}"
            end_fmt = f"{int(segment.end//3600):02d}:{int((segment.end%3600)//60):02d}:{segment.end%60:06.3f}"

            vtt_lines.append(f"{start_fmt} --> {end_fmt}\n{vtt_text}\n")

        full_text = "\n".join(full_transcript)

        txt_path = os.path.join(transcript_dir, "transcript.txt")
        vtt_path = os.path.join(transcript_dir, "transcript.vtt")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text + "\n")

        with open(vtt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(vtt_lines) + "\n")

        for path in [transcript_dir, txt_path, vtt_path]:
            shutil.chown(path, user="bigbluebutton", group="bigbluebutton")
            os.chmod(path, 0o755 if os.path.isdir(path) else 0o644)

    if transcript_only:
        print("[Pipeline] Transcript-only mode enabled; skipping summary generation.")
        return

    # 5. Extract metadata & generate summary artifacts
    print("[Summary] Generating AI meeting summary...")
    summary = generate_summary(full_text)

    room_name, meeting_date = get_meeting_metadata(pres_dir)

    # Sanitize room name for filesystem safety in filename
    safe_room_name = re.sub(r'[\\/*?:"<>|]', "", room_name).replace(" ", "_")
    rtf_filename = f"Summary_{safe_room_name}_{meeting_date}.rtf"

    summary_md_path = os.path.join(summary_dir, "summary.md")
    summary_html_path = os.path.join(summary_dir, "summary.html")
    summary_rtf_path = os.path.join(summary_dir, rtf_filename)

    # Write Markdown
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    # Generate HTML & RTF with room metadata
    generate_rtf_from_md(summary, summary_rtf_path, meeting_name=room_name, meeting_date=meeting_date)
    generate_html_viewer(summary, summary_html_path, rtf_filename=rtf_filename, meeting_name=room_name, meeting_date=meeting_date)

    # Apply permissions across all summary artifacts
    for path in [summary_dir, summary_md_path, summary_html_path, summary_rtf_path]:
        if os.path.exists(path):
            shutil.chown(path, user="bigbluebutton", group="bigbluebutton")
            os.chmod(path, 0o755 if os.path.isdir(path) else 0o644)

    print("[Pipeline] Registering native format directories...")
    register_custom_formats_native(
        meeting_id=meeting_id,
        base_published_dir=base_published_dir
    )

    print(f"[Pipeline] Finished processing {meeting_id} successfully.")


if __name__ == "__main__":
    main()