#!/usr/bin/env python3
"""Publish transcript and summary files into a BigBlueButton recording.

Stdlib only, so the BigBlueButton host can render HTML, RTF, and playback
metadata without Whisper or the local LLM.
"""

import datetime
import os
import re
import shutil
import xml.etree.ElementTree as ET
from html import escape


def own_published(path: str) -> None:
    """Make a published file readable by the BigBlueButton web server."""
    if not os.path.exists(path):
        return
    try:
        shutil.chown(path, user="bigbluebutton", group="bigbluebutton")
    except (LookupError, PermissionError, OSError):
        pass
    os.chmod(path, 0o755 if os.path.isdir(path) else 0o644)


def summary_rtf_filename(meeting_name: str, meeting_date: str) -> str:
    safe = re.sub(r'[\\/*?:"<>|]', "", meeting_name).replace(" ", "_")
    safe = safe.strip("._") or "Meeting"
    return f"Summary_{safe}_{meeting_date}.rtf"


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
    meeting_name = escape(meeting_name)
    meeting_date = escape(meeting_date)
    rtf_filename = escape(rtf_filename, quote=True)
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
            meeting_name_node = root.find(".//meetingName")
            if meeting_name_node is None:
                meeting_name_node = root.find(".//meta/meetingName")
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

def register_custom_formats_native(
    meeting_id: str,
    base_published_dir: str = "/var/bigbluebutton/published",
    properties_path: str | None = None,
):
    """
    Creates native published format directories for 'summary' and 'transcript'
    by reading (and NOT modifying) the primary presentation metadata.xml.
    """
    domain = get_server_domain(properties_path)
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

            own_published(fmt_dir)
            own_published(meta_file_path)

            print(f"[Success] Created native format for '{fmt_name}' at {meta_file_path}")

    except ET.ParseError as pe:
        print(f"[Metadata Error] Malformed XML encountered in primary metadata: {pe}")
    except Exception as e:
        print(f"[Metadata Error] Failed to process {meeting_id}: {e}")

def get_server_domain(prop_path: str | None = None) -> str:
    """
    Reads the BigBlueButton server URL from bbb-web.properties.
    Raises RuntimeError if the file or key is missing.
    """
    path = prop_path or os.environ.get("BBB_WEB_PROPERTIES", "/etc/bigbluebutton/bbb-web.properties")

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip().startswith("bigbluebutton.web.serverURL="):
                        url = line.split("=", 1)[1].strip()
                        if url:
                            return url.rstrip("/")
        except OSError as exc:
            raise RuntimeError(f"Failed to read {path}: {exc}") from exc

    raise RuntimeError(
        f"Could not resolve server URL. Key 'bigbluebutton.web.serverURL' missing or file not found at {path}"
    )
