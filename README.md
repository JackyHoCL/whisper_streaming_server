1. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
   
2. Run the server:
   In default Config:
     ```
     python server.py
     ```
     
   In custom host:
    ```
    uvicorn server:app --host {host_ip} --port {port_number}
    ```
