#!/bin/bash

MEETING_ID=$1
LOG_FILE="/var/log/bigbluebutton/post_publish_transcribe.log"

echo "=== [$(date)] Starting Whisper transcription for $MEETING_ID ===" >> $LOG_FILE

# Run with nice/ionice so CPU priority stays low for live BBB WebRTC audio
nice -n 15 ionice -c 3 /opt/speech_env/bin/python3 /usr/local/bigbluebutton/core/scripts/transcribe_meeting.py "$MEETING_ID" >> $LOG_FILE 2>&1

echo "=== [$(date)] Finished Whisper transcription for $MEETING_ID ===" >> $LOG_FILE
