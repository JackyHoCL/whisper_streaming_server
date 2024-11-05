from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import os
from faster_whisper import WhisperModel
import numpy as np

from pydub import AudioSegment

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

app = FastAPI()

# Initialize the WhisperModel once during startup
model_size = "deepdml/faster-whisper-large-v3-turbo-ct2"
# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")
# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

class TranscriptionResponse(BaseModel):
    language: str
    language_probability: float
    segments: list

@app.post("/transcribe/file")
async def transcribe_audio(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_file_path = f"tmp/{file.filename}"
    with open(temp_file_path, "wb") as f:
        f.write(await file.read())
    try:
        segments, info = model.transcribe(temp_file_path, beam_size=5, vad_filter=True)
        result_segments = [{"start": segment.start, "end": segment.end, "text": segment.text} for segment in segments]
        response = {
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": result_segments
        }
        return TranscriptionResponse(**response)
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)

@app.websocket("/transcribe/stream")
async def transcribe_stream(ws: WebSocket):
    await ws.accept()
    audio_data = bytearray()
    
    try:
        while True:
            data = await ws.receive_bytes()
            audio_data.extend(data)

            if len(audio_data) > 150000:  # Threshold for triggering transcription
                await ws.send_json({"transcribing": True})
                data_input = np.frombuffer(audio_data, dtype=np.float32).astype(np.float32)
                segments, info = model.transcribe(audio=data_input, beam_size=5, vad_filter=True)
                
                transcript = " ".join([segment.text for segment in segments])
                
                # Send transcript back to client
                await ws.send_json({"transcript": transcript})
                
                # Reset buffer after transcription
                audio_data.clear()

    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run('server:app', host="0.0.0.0", port=10928, reload=True)