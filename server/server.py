from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import os
from faster_whisper import WhisperModel
import numpy as np
import argparse
from pydub import AudioSegment
import whisper
import asyncio
import webrtcvad

vad = webrtcvad.Vad()
vad.set_mode(2)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

parser = argparse.ArgumentParser(description="Use a specified GPU")
parser.add_argument('--gpu', type=int, default=0, help='GPU index to use')
args = parser.parse_args()
sample_rate = 16000
stream_timeout = 30
start_window_size = 0.03 #second
stop_window_size = 2

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

app = FastAPI()

# Initialize the WhisperModel once during startup
model_size = "JackyHoCL/whisper-large-v3-turbo-cantonese-yue-english-ct2"
# Run on GPU with FP32
# model = WhisperModel(model_size, device="cuda", compute_type="float32")
# Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")
# or run on GPU with INT8
model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
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
    sliding_window = bytearray()
    transcrbe_data = bytearray()
    temp = bytearray()
    transcribing = False
    vad_result = False
    false_count = 0

    try:
        while True:
            data = await ws.receive_bytes()
            audio_data.extend(data)
            data_input_from_source = np.frombuffer(audio_data, dtype=np.float32).astype(np.float32)
            # print(data_input_from_source)
            window_size = start_window_size
            print(len(transcrbe_data))
            if len(audio_data) > int(sample_rate * window_size): 
                sliding_window = data_input_from_source.tobytes()[(-1 * int(sample_rate*window_size*2)):]
                # print(sliding_window)
                vad_result = vad.is_speech(sliding_window, sample_rate)
                print(vad_result)
            if vad_result and len(transcrbe_data) < stream_timeout * sample_rate:
                transcribing = True
                false_count = 0
                # Reset buffer after transcription
                transcrbe_data.extend(data)
                # audio_data.clear()
                # sliding_window.clear()
            else:
                if (transcribing):
                    false_count += 1
                if (false_count > 100): 
                    transcribing = False
                    #transcribe recorded data after silent more than 2s or reach the speech threshold
                    if len(transcrbe_data) > 0:
                        asyncio.create_task(transcribe_audio(np.frombuffer(transcrbe_data, dtype=np.float32).astype(np.float32), ws))
                        transcrbe_data.clear()
                    false_count = 0

            if len(audio_data) > (stream_timeout * sample_rate * 2):
                audio_data = audio_data[-sample_rate*stream_timeout:]

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