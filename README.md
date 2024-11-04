**1. Install dependencies:**

   ffmpeg:
   
   linux
   
   ```
   sudo apt install ffmpeg
   ```
   
   windows:
   
   a. Download ffmpeg from https://www.ffmpeg.org/download.html
   
   b. Extract and move to target location
   
   c. set environment variable with the full path of ffmpeg/bin folder
   
   portaudio:
   ```
   sudo apt-get install portaudio19-dev
   ```
   pip:
   ```
   pip install -r requirements.txt
   ```
   
**3. Run the server:**

   In default Config:
   ```
   python server.py
   ```
     
   In custom host:
   ```
   uvicorn server:app --host {host_ip} --port {port_number}
   ```
