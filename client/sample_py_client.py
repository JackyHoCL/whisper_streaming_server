import asyncio
import websockets
import numpy as np
import whisper
import pyaudio

SERVER_URL = "ws://localhost:8000/transcribe/stream"

def capture_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024)
    return stream, p

async def send_audio(websocket, stream):
    print("WebSocket connection established.")
    
    # Capture audio from the microphone in chunks and send it to the server
    chunk_size = 1024  # Adjust chunk size as needed
    while True:
        audio_chunk = stream.read(chunk_size, exception_on_overflow=False)
        audio_samples = np.frombuffer(audio_chunk, dtype=np.float32)
        
        # Send the audio data in chunks
        await websocket.send(audio_samples.tobytes())
        await asyncio.sleep(0.01)  # Small delay to prevent overwhelming the server

async def receive_transcript(websocket):
    try:
        while True:
            message = await websocket.recv()
            if isinstance(message, str):
                print("Received transcript:", message)
            else:
                print("Received unexpected message type.")
    except websockets.ConnectionClosed:
        print("WebSocket connection closed.")

async def main():
    stream, p = capture_audio()
    async with websockets.connect(SERVER_URL) as websocket:
        await asyncio.gather(send_audio(websocket, stream), receive_transcript(websocket))
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    asyncio.run(main())