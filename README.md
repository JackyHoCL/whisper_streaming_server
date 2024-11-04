1. Install dependencies:
   ```
   sudo apt-get install portaudio19-dev
   ```
   ```
   pip install -r requirements.txt
   ```
   
3. Run the server:
   In default Config:
   ```
   python server.py
   ```
     
   In custom host:
   ```
   uvicorn server:app --host {host_ip} --port {port_number}
   ```
