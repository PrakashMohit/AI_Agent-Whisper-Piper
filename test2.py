import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import os
import subprocess
import torch
import soundfile as sf

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from snac import SNAC


# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
DURATION = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- LOAD MODELS ONCE ----------------

print("Loading Whisper...")
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    device=0 if DEVICE == "cuda" else -1,
    generate_kwargs={
        "language": "en",
        "task": "transcribe",
        "temperature": 0.0,
    }
)

print("Loading Maya-1...")
tokenizer = AutoTokenizer.from_pretrained("maya-research/maya1")
maya_model = AutoModelForCausalLM.from_pretrained(
    "maya-research/maya1",
    torch_dtype=torch.float16,
    device_map="auto"
)
snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(DEVICE)


# ---------------- FUNCTIONS ----------------

def record_audio():
    print("Speak now...")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()

    audio = audio.squeeze()
    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(tmp.name, SAMPLE_RATE, (audio * 32767).astype("int16"))
    return tmp.name


def ollama_reason(user_text):
    prompt = (
        "You are a calm, concise personal assistant. "
        "Reply briefly and naturally.\n\n"
        f"User: {user_text}\nAssistant:"
    )

    result = subprocess.check_output(
        ["ollama", "run", "gemma3:4b", prompt],
        text=True
    )
    return result.strip()


def maya_tts(text, out_path="reply.wav"):
    prompt = f"<voice:default>{text}"

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = maya_model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False
        )

    # IMPORTANT: only audio tokens
    audio_tokens = output[0].cpu().numpy()
    audio = snac.decode(audio_tokens)

    sf.write(out_path, audio, 24000)


def play_audio(path):
    subprocess.run(
        ["ffplay", "-autoexit", "-nodisp", path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


# ---------------- MAIN LOOP ----------------

while True:
    input("\nPress ENTER to talk (Ctrl+C to quit)")

    audio_path = record_audio()

    print("Transcribing...")
    result = asr(audio_path)
    os.remove(audio_path)

    user_text = result["text"]
    print("You:", user_text)

    reply = ollama_reason(user_text)
    print("Agent:", reply)

    maya_tts(reply, "reply.wav")
    play_audio("reply.wav")
