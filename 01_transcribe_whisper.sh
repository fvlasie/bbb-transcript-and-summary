#!/bin/bash
# /usr/local/bigbluebutton/core/scripts/post_publish/01_transcribe_whisper.sh

MEETING_ID=$1
RAW_DIR="/var/bigbluebutton/recording/raw/${MEETING_ID}"
PUBLISHED_DIR="/var/bigbluebutton/published/presentation/${MEETING_ID}"
LOG_FILE="/var/log/bigbluebutton/post_publish_transcribe.log"

# Localized cache directory for Whisper model weights
export HF_HOME="/var/bigbluebutton/hf_cache"

# Cloudflare Workers AI Credentials
export CF_ACCOUNT_ID="your_actual_account_id_here"
export CF_API_TOKEN="your_actual_api_token_here"

# Locate the published WebM artifact directly (No raw audio dependency)
WEBM_FILE=""
if [ -f "${PUBLISHED_DIR}/video/webcams.webm" ]; then
    WEBM_FILE="${PUBLISHED_DIR}/video/webcams.webm"
elif [ -f "${PUBLISHED_DIR}/deskshare/deskshare.webm" ]; then
    WEBM_FILE="${PUBLISHED_DIR}/deskshare/deskshare.webm"
fi

EVENTS_XML="${RAW_DIR}/events.xml"

echo "[$(date)] Starting published WebM transcription for ${MEETING_ID}..." >> "$LOG_FILE"

if [ -n "$WEBM_FILE" ] && [ -f "$WEBM_FILE" ]; then
    nice -n 15 ionice -c 3 /opt/speech_env/bin/python3 \
        /usr/local/bigbluebutton/core/scripts/post_publish/lib/transcribe_meeting.py \
        "$WEBM_FILE" \
        "$PUBLISHED_DIR" \
        "$EVENTS_XML" >> "$LOG_FILE" 2>&1
else
    echo "[$(date)] Error: Published WebM file not found in ${PUBLISHED_DIR}" >> "$LOG_FILE"
fi
