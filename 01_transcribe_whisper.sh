#!/bin/bash
# /usr/local/bigbluebutton/core/scripts/post_publish/01_transcribe_whisper.sh

MEETING_ID=$1
LOG_FILE="/var/log/bigbluebutton/post_publish_transcribe.log"

# Localized cache directory for Whisper model weights
export HF_HOME="/var/bigbluebutton/hf_cache"

# Cloudflare Workers AI Credentials
export CF_ACCOUNT_ID="your_actual_account_id_here"
export CF_API_TOKEN="your_actual_api_token_here"

echo "[$(date)] Starting published WebM transcription for ${MEETING_ID}..." >> "$LOG_FILE"

# Check that MEETING_ID parameter was provided
if [ -n "$MEETING_ID" ]; then
    nice -n 15 ionice -c 3 /opt/speech_env/bin/python3 \
        /usr/local/bigbluebutton/core/scripts/post_publish/lib/transcribe_meeting.py \
        "$MEETING_ID" >> "$LOG_FILE" 2>&1
else
    echo "[$(date)] Error: No MEETING_ID passed to post_publish hook." >> "$LOG_FILE"
fi
