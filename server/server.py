from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import os
from faster_whisper import WhisperModel
import numpy as np
import argparse
from pydub import AudioSegment
import whisper
import asyncio

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

parser = argparse.ArgumentParser(description="Use a specified GPU")
parser.add_argument('--gpu', type=int, default=0, help='GPU index to use')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

app = FastAPI()

# Initialize the WhisperModel once during startup
model_size = "JackyHoCL/whisper-small-cantonese-yue-english-ct2"
# Run on GPU with FP32
model = WhisperModel(model_size, device="cuda", compute_type="float32")
# Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")
# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# model = WhisperModel(model_size, device="cuda", compute_type="int8")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

warm_up_audio_file_name = 'warmup.mp3'

def get_warmup_audio():
    current_directory =  os.path.dirname(os.path.abspath(__file__))
    warm_up_audio = current_directory + '/' + warm_up_audio_file_name
    audio = whisper.load_audio(warm_up_audio)
    return audio


warm_up_audio = get_warmup_audio()

class TranscriptionResponse(BaseModel):
    language: str
    language_probability: float
    segments: list

@app.post("/transcribe/file")
async def transcribe_audio(file: UploadFile = File(...)):
    print('ok')
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

            if len(audio_data) > 200000:  # Threshold for triggering transcription
                data_input_from_source = np.frombuffer(audio_data, dtype=np.float32).astype(np.float32)
                # data_input = np.concatenate((warm_up_audio, data_input_from_source), axis=0)
                asyncio.create_task(transcribe_audio(data_input_from_source, ws))
                # Reset buffer after transcription
                audio_data.clear()

    except WebSocketDisconnect:
        print("Client disconnected")

async def transcribe_audio(data, ws: WebSocket):
    segments, info = model.transcribe(audio=data, beam_size=5, vad_filter=True)
    transcript = " ".join([segment.text for segment in segments])
    
    # Send transcript back to client
    await ws.send_json({"transcript": transcript})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run('server:app', host="0.0.0.0", port=10928, reload=True)