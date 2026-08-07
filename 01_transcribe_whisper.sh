#!/bin/bash
# /usr/local/bigbluebutton/core/scripts/post_publish/01_transcribe_whisper.sh

MEETING_ID=$1
LOG_FILE="/var/log/bigbluebutton/post_publish_transcribe.log"

# Localized cache directory for both Whisper and Llama models
export HF_HOME="/var/bigbluebutton/hf_cache"

# Gemma 4 (best quality for meeting context/nuance)
export LLM_MODEL_FAMILY="gemma-4"

echo "[$(date)] Starting published WebM transcription for ${MEETING_ID}..." >> "$LOG_FILE"

# Check that MEETING_ID parameter was provided
if [ -n "$MEETING_ID" ]; then
    nice -n 15 ionice -c 3 /opt/speech_env/bin/python3 \
        /usr/local/bigbluebutton/core/scripts/post_publish/lib/transcribe_meeting.py \
        "$MEETING_ID" >> "$LOG_FILE" 2>&1
else
    echo "[$(date)] Error: No MEETING_ID passed to post_publish hook." >> "$LOG_FILE"
fi
