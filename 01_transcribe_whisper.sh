#!/bin/bash
# Submit a published recording to the off-host transcript worker.
# The hook returns as soon as the upload is accepted or recorded for retry.
# bbb-transcript-retrieve.timer pulls the transcript and summary later.

MEETING_ID=$1
LOG_FILE="${TRANSCRIPT_LOG_FILE:-/var/log/bigbluebutton/post_publish_transcribe.log}"
ENV_FILE="/etc/bigbluebutton/transcript-worker.env"

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
if [ -f "$SCRIPT_DIR/lib/bbb_submit.py" ]; then
    SUBMIT_PY="$SCRIPT_DIR/lib/bbb_submit.py"
else
    SUBMIT_PY="$SCRIPT_DIR/bbb_submit.py"
fi

mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true

log() {
    echo "[$(date)] $*" >> "$LOG_FILE" 2>/dev/null || echo "[$(date)] $*"
}

if [ -z "${MEETING_ID}" ]; then
    log "Error: No MEETING_ID passed to post_publish hook."
    exit 0
fi

if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set +a
else
    log "Error: Missing ${ENV_FILE}. Left ${MEETING_ID} unsubmitted."
    exit 0
fi

PYTHON=$(command -v python3 || true)
if [ -z "$PYTHON" ]; then
    log "Error: python3 not found. Left ${MEETING_ID} unsubmitted."
    exit 0
fi

log "Submitting transcript job for ${MEETING_ID}..."
"$PYTHON" "$SUBMIT_PY" "$MEETING_ID" >> "$LOG_FILE" 2>&1 || log "Submit script exited $? for ${MEETING_ID}."
exit 0
