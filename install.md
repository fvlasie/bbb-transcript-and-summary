# 1. Ensure system audio tools & FFmpeg are present
sudo apt update && sudo apt install -y python3.10-venv python3-pip ffmpeg portaudio19-dev

# 2. Create and activate a dedicated virtual environment
python3 -m venv speech_env
source speech_env/bin/activate

# 3. Upgrade pip and install faster-whisper + audio tools
pip install --upgrade pip
pip install faster-whisper sounddevice numpy

# Move virtual environment out of /root
sudo mv /root/speech_env /opt/speech_env

# Ensure the cache folder exists on your big /var/bigbluebutton partition
sudo mkdir -p /var/bigbluebutton/hf_cache

# Transfer ownership to the bigbluebutton user
sudo chown -R bigbluebutton:bigbluebutton /opt/speech_env /var/bigbluebutton/hf_cache

# Copy file to /usr/local/bigbluebutton/core/scripts/transcribe_meeting.py 
sudo chmod +x /usr/local/bigbluebutton/core/scripts/transcribe_meeting.py
sudo chown bigbluebutton:bigbluebutton /usr/local/bigbluebutton/core/scripts/transcribe_meeting.py

# Create sh script
nano /usr/local/bigbluebutton/core/scripts/post_publish/01_transcribe_whisper.sh
sudo chmod +x /usr/local/bigbluebutton/core/scripts/post_publish/01_transcribe_whisper.sh
sudo chown bigbluebutton:bigbluebutton /usr/local/bigbluebutton/core/scripts/post_publish/01_transcribe_whisper.sh
