#!/bin/bash

MEETING_ID=$1
LOG_FILE="/var/log/bigbluebutton/post_publish_transcribe.log"

# Force ALL HuggingFace operations (models, logs, locks) to the large drive
export HF_HOME="/var/bigbluebutton/hf_cache"

echo "=== [$(date)] Starting Whisper transcription for $MEETING_ID ===" >> $LOG_FILE

nice -n 15 ionice -c 3 /opt/speech_env/bin/python3 /usr/local/bigbluebutton/core/scripts/transcribe_meeting.py "$MEETING_ID" >> $LOG_FILE 2>&1

echo "=== [$(date)] Finished Whisper transcription for $MEETING_ID ===" >> $LOG_FILE
