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

echo "[$(date)] Starting speaker-tagged transcription for ${MEETING_ID}..." >> "$LOG_FILE"

AUDIO_FILE="${RAW_DIR}/audio/audio.webm"
EVENTS_XML="${RAW_DIR}/events.xml"

if [ -f "$AUDIO_FILE" ]; then
    python3 /usr/local/bigbluebutton/core/scripts/transcribe_meeting.py \
        "$AUDIO_FILE" \
        "$PUBLISHED_DIR" \
        "$EVENTS_XML" >> "$LOG_FILE" 2>&1
else
    echo "[$(date)] Error: Audio file $AUDIO_FILE not found." >> "$LOG_FILE"
fi
