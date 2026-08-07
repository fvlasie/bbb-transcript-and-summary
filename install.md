# 1. Ensure system audio tools, build tools, & FFmpeg are present
sudo apt update && sudo apt install -y python3.10-venv python3-pip ffmpeg portaudio19-dev build-essential cmake

# 2. Create the virtual environment in /opt/speech_env
sudo python3 -m venv /opt/speech_env  

# 3. Install dependencies directly using the venv's pip executable
sudo /opt/speech_env/bin/pip install --upgrade pip  
sudo /opt/speech_env/bin/pip install faster-whisper sounddevice numpy markdown llama-cpp-python huggingface-hub

# 4. Verify imports in the venv
sudo /opt/speech_env/bin/python3 -c 'from llama_cpp import Llama; from faster_whisper import WhisperModel; print("Venv setup successful!")'  
sudo mkdir -p /var/bigbluebutton/hf_cache  
sudo chown -R bigbluebutton:bigbluebutton /var/bigbluebutton/hf_cache /opt/speech_env

# Copy Python script to BBB location
sudo chmod +x /usr/local/bigbluebutton/core/scripts/post_publish/lib/transcribe_meeting.py  
sudo chown bigbluebutton:bigbluebutton /usr/local/bigbluebutton/core/scripts/post_publish/lib/transcribe_meeting.py

# Create shell script wrapper that sets the model path
nano /usr/local/bigbluebutton/core/scripts/post_publish/01_transcribe_whisper.sh  
sudo chmod +x /usr/local/bigbluebutton/core/scripts/post_publish/01_transcribe_whisper.sh  
sudo chown bigbluebutton:bigbluebutton /usr/local/bigbluebutton/core/scripts/post_publish/01_transcribe_whisper.sh

# Give BBB user ownership of its own home cache
sudo chown -R bigbluebutton:bigbluebutton /home/bigbluebutton/.cache

# Create new format directories
sudo mkdir -p /var/bigbluebutton/published/summary  
sudo mkdir -p /var/bigbluebutton/published/transcript  

sudo chown -R bigbluebutton:bigbluebutton /var/bigbluebutton/published/summary  
sudo chown -R bigbluebutton:bigbluebutton /var/bigbluebutton/published/transcript  

sudo chmod -R 775 /var/bigbluebutton/published/summary  
sudo chmod -R 775 /var/bigbluebutton/published/transcript

# Execute from the ruby trigger (add this to /usr/local/bigbluebutton/core/scripts/post_publish/post_publish.rb)
logger.info("Executing Whisper transcription for meeting: #{meeting_id}")  
system("/usr/local/bigbluebutton/core/scripts/post_publish/01_transcribe_whisper.sh #{meeting_id}")

# Customise the prompts
transcribe_meeting.py: 495, system_prompt
transcribe_meeting.py: 1286, final_system_prompt
transcribe_meeting.py: 1160, initial_prompt