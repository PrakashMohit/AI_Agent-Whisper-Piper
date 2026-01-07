from email.mime import audio
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import os
import subprocess
import torch
from transformers import pipeline
import queue
import sounddevice as sd
import time

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
DURATION = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD MODELS ONCE ----------------
print("Loading Whisper...")
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device=0 if DEVICE == "cuda" else -1,
    generate_kwargs={
        "language": "en",
        "task": "transcribe",
        "temperature": 0.0,
    }
)

PIPER_EXE = r"D:\AI assistant\piper\piper.exe"
MODEL = r"D:\AI assistant\piper\models\en_US-amy-medium.onnx"

def piper_tts(text, out_path="reply.wav"):
     # send text via stdin (avoid temporary txt file)
    subprocess.run(
        [PIPER_EXE, "--model", MODEL, "--output_file", out_path],
        input=text,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )


audio_level_queue = queue.Queue()

def record_audio():
    frames = []
    start_time = time.time()

    def callback(indata, frames_count, time_info, status):
        if status:
            print(status)
        frames.append(indata.copy())
        audio_level_queue.put(np.abs(indata).mean())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=callback
    ):
        while time.time() - start_time < DURATION:
            time.sleep(0.01)

    audio = np.concatenate(frames, axis=0).squeeze()
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return audio  # return numpy array instead of temp wav file




def ollama_reason(user_text):
    prompt = (
        "You are a calm, concise personal assistant.\n"
        "Your name is Friday.\n"
        "Your are a Female.\n"
        "Reply briefly and naturally.\n\n"
        f"User: {user_text}\nAssistant:"
    )

    return subprocess.check_output(
        ["ollama", "run", "gemma3:4b", prompt],
        encoding="utf-8",
        errors="ignore"
    ).strip()

def play_audio(path):
    # use sounddevice to play WAV (faster than launching ffplay)
    sr, data = wav.read(path)
    if data.dtype == np.int16:
        data = (data.astype("float32") / 32768.0)
    if data.ndim == 1:
        sd.play(data, sr)
    else:
        sd.play(data, sr, blocking=False)
    sd.wait()

# UI calling main loop funcion
def run_agent_once():
    audio= record_audio()
    # call ASR: pass a dict so sampling_rate is treated as input (not forwarded to model.generate)
    try:
        result = asr({"array": audio, "sampling_rate": SAMPLE_RATE})
    except Exception:
        # fallback for pipeline implementations that expect a raw array
        result = asr(audio)
    user_text = result.get("text", "").strip()
    
    if not user_text:
        return None, None

    reply = ollama_reason(user_text)

    piper_tts(reply)
    play_audio("reply.wav")
    try:
        os.remove("reply.wav")
    except:
        pass

    return user_text, reply
