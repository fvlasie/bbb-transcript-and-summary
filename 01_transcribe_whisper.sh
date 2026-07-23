#!/bin/bash

MEETING_ID=$1
LOG_FILE="/var/log/bigbluebutton/post_publish_transcribe.log"

export HF_HOME="/var/bigbluebutton/hf_cache"
export CF_ACCOUNT_ID="your_cloudflare_account_id_here"
export CF_API_TOKEN="your_cloudflare_api_token_here"

(
  echo "=== [$(date)] Starting Whisper & CF Summary pipeline for $MEETING_ID ===" >> "$LOG_FILE"
  nice -n 15 ionice -c 3 /opt/speech_env/bin/python3 /usr/local/bigbluebutton/core/scripts/transcribe_meeting.py "$MEETING_ID" >> "$LOG_FILE" 2>&1
  echo "=== [$(date)] Finished Whisper & CF Summary pipeline for $MEETING_ID ===" >> "$LOG_FILE"
) &

exit 0
