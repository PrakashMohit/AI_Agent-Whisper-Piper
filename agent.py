import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
import subprocess
import torch
from transformers import pipeline
import queue
import time
import json
from pathlib import Path

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).parent
MEMORY_DIR = BASE_DIR / "memory"
PROFILE_PATH = MEMORY_DIR / "profile.json"
CHAT_PATH = MEMORY_DIR / "chat_history.json"

MEMORY_DIR.mkdir(exist_ok=True)

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
DURATION = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CHAT_TURNS = 20  # keep memory small & fast

# ---------------- LOAD MEMORY ----------------

def load_profile():
    if PROFILE_PATH.exists():
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "user_name": "User",
        "assistant_name": "Friday",
        "personality": "calm, emotionally available, supportive"
    }

def load_chat():
    if CHAT_PATH.exists():
        with open(CHAT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_chat(chat):
    chat = chat[-MAX_CHAT_TURNS:]
    with open(CHAT_PATH, "w", encoding="utf-8") as f:
        json.dump(chat, f, indent=2)

profile = load_profile()
chat_history = load_chat()

# ---------------- LOAD MODELS ----------------
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
    subprocess.run(
        [PIPER_EXE, "--model", MODEL, "--output_file", out_path],
        input=text,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

# ---------------- AUDIO INPUT ----------------

audio_level_queue = queue.Queue()

def record_audio():
    frames = []
    start_time = time.time()

    def callback(indata, frames_count, time_info, status):
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
    return audio

# ---------------- LLM WITH MEMORY ----------------

def build_prompt(user_text):
    system = (
        f"You are {profile['assistant_name']}.\n"
        f"The user's name is {profile['user_name']}.\n"
        f"Personality: {profile['personality']}.\n"
        "You remember previous conversations.\n"
        "Respond naturally and emotionally.\n\n"
    )

    history = ""
    for turn in chat_history:
        history += f"{turn['role'].capitalize()}: {turn['content']}\n"

    return system + history + f"User: {user_text}\nAssistant:"

def ollama_reason(user_text):
    prompt = build_prompt(user_text)

    reply = subprocess.check_output(
        ["ollama", "run", "gemma3:4b", prompt],
        encoding="utf-8",
        errors="ignore"
    ).strip()

    return reply

# ---------------- AUDIO OUTPUT ----------------

def play_audio(path):
    sr, data = wav.read(path)
    if data.dtype == np.int16:
        data = data.astype("float32") / 32768.0
    sd.play(data, sr)
    sd.wait()

# ---------------- MAIN AGENT CALL ----------------

def run_agent_once():
    global chat_history

    audio = record_audio()

    try:
        result = asr({"array": audio, "sampling_rate": SAMPLE_RATE})
    except:
        result = asr(audio)

    user_text = result.get("text", "").strip()
    if not user_text:
        return None, None

    reply = ollama_reason(user_text)

    # update memory
    chat_history.append({"role": "user", "content": user_text})
    chat_history.append({"role": "assistant", "content": reply})
    save_chat(chat_history)

    piper_tts(reply)
    play_audio("reply.wav")
    os.remove("reply.wav")

    return user_text, reply
