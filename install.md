# 1. Ensure system audio tools, build tools, & FFmpeg are present
sudo apt update && sudo apt install -y python3.10-venv python3-pip ffmpeg portaudio19-dev build-essential cmake

<<<<<<< HEAD
# 2. Create the virtual environment in /opt/speech_env
# IMPORTANT: Create it directly at /opt/speech_env (do NOT create in /root or ~ and move it, venvs have hardcoded path references!)
sudo python3 -m venv /opt/speech_env  

# 3. Install dependencies directly using the venv's pip executable
# (Using explicit binary paths avoids path activation confusion)
sudo /opt/speech_env/bin/pip install --upgrade pip  
sudo /opt/speech_env/bin/pip install faster-whisper sounddevice numpy markdown llama-cpp-python huggingface-hub
=======
# 2. Create and activate a dedicated virtual environment
python3 -m venv speech_env  
source speech_env/bin/activate

# 3. Upgrade pip and install faster-whisper + audio tools
pip install --upgrade pip  
pip install faster-whisper sounddevice numpy markdown
>>>>>>> de72fb8f8462d406665cc4e7b9d607b09b23aded

# 4. Verify imports in the venv
sudo /opt/speech_env/bin/python3 -c 'from llama_cpp import Llama; from faster_whisper import WhisperModel; print("Venv setup successful!")'  
sudo mkdir -p /var/bigbluebutton/hf_cache  
sudo chown -R bigbluebutton:bigbluebutton /var/bigbluebutton/hf_cache /opt/speech_env

# Copy Python script to BBB location
sudo chmod +x /usr/local/bigbluebutton/core/scripts/post_publish/lib/transcribe_meeting.py  
sudo chown bigbluebutton:bigbluebutton /usr/local/bigbluebutton/core/scripts/post_publish/lib/transcribe_meeting.py

<<<<<<< HEAD
# Create shell script wrapper that sets the model path
=======
# Transfer ownership to the bigbluebutton user
sudo chown -R bigbluebutton:bigbluebutton /opt/speech_env /var/bigbluebutton/hf_cache

# Copy file to /usr/local/bigbluebutton/core/scripts/transcribe_meeting.py 
sudo chmod +x /usr/local/bigbluebutton/core/scripts/transcribe_meeting.py  
sudo chown bigbluebutton:bigbluebutton /usr/local/bigbluebutton/core/scripts/transcribe_meeting.py

# Create sh script
>>>>>>> de72fb8f8462d406665cc4e7b9d607b09b23aded
nano /usr/local/bigbluebutton/core/scripts/post_publish/01_transcribe_whisper.sh  
sudo chmod +x /usr/local/bigbluebutton/core/scripts/post_publish/01_transcribe_whisper.sh  
sudo chown bigbluebutton:bigbluebutton /usr/local/bigbluebutton/core/scripts/post_publish/01_transcribe_whisper.sh

# Example 01_transcribe_whisper.sh content:
# #!/bin/bash
# export HF_HOME="/var/bigbluebutton/hf_cache"
# /opt/speech_env/bin/python3 /usr/local/bigbluebutton/core/scripts/post_publish/lib/transcribe_meeting.py "$1"

# Give BBB user ownership of its own home cache
sudo chown -R bigbluebutton:bigbluebutton /home/bigbluebutton/.cache

# Create new format directories
sudo mkdir -p /var/bigbluebutton/published/summary  
sudo mkdir -p /var/bigbluebutton/published/transcript  

sudo chown -R bigbluebutton:bigbluebutton /var/bigbluebutton/published/summary  
sudo chown -R bigbluebutton:bigbluebutton /var/bigbluebutton/published/transcript  

sudo chmod -R 775 /var/bigbluebutton/published/summary  
<<<<<<< HEAD
sudo chmod -R 775 /var/bigbluebutton/published/transcript

=======
sudo chmod -R 775 /var/bigbluebutton/published/transcript  

# Re-verify the large cache folder is owned by BBB
sudo mkdir -p /home/bigbluebutton/.cache  
sudo chown -R bigbluebutton:bigbluebutton /var/bigbluebutton/hf_cache

>>>>>>> de72fb8f8462d406665cc4e7b9d607b09b23aded
# Execute from the ruby trigger (add this to /usr/local/bigbluebutton/core/scripts/post_publish/post_publish.rb)
logger.info("Executing Whisper transcription for meeting: #{meeting_id}")  
system("/usr/local/bigbluebutton/core/scripts/post_publish/01_transcribe_whisper.sh #{meeting_id}")
