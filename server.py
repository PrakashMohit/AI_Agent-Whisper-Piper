import asyncio
import json
import re
import struct
import time
import warnings
from pathlib import Path
from typing import AsyncGenerator, List, Optional
 

import numpy as np
import torch
import websockets
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from snac import SNAC
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import requests

warnings.filterwarnings("ignore")

# ─────────────────────────────── CONFIG ──────────────────────────────────────
WS_HOST        = "0.0.0.0"
WS_PORT        = 8765
SAMPLE_RATE    = 16000          # Whisper input
MAYA_RATE      = 24000          # Maya1 / SNAC output
OLLAMA_URL     = "http://localhost:11434/api/generate"
OLLAMA_MODEL   = "gemma3:4b"
MAYA_MODEL     = "maya-research/maya1"
SNAC_MODEL     = "hubertsiuzdak/snac_24khz"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CHAT_TURNS = 40

# Maya1 special token IDs
CODE_START  = 128257
CODE_END    = 128258
CODE_OFFSET = 128266
SNAC_MIN    = 128266
SNAC_MAX    = 156937
SOH_ID      = 128259
EOH_ID      = 128260
SOA_ID      = 128261
BOS_ID      = 128000
TEXT_EOT    = 128009
TOKENS_PER_FRAME = 7

# Maya1 voice description for Friday
FRIDAY_VOICE = (
    "Female voice in her mid-20s with a warm American accent. "
    "Soft, intimate tone, conversational pacing, slightly breathy, "
    "playful and expressive delivery."
)

# ─────────────────────────────── MEMORY ──────────────────────────────────────
MEMORY_DIR   = Path(__file__).parent / "memory"
PROFILE_PATH = MEMORY_DIR / "profile.json"
CHAT_PATH    = MEMORY_DIR / "chat_history.json"
MEMORY_DIR.mkdir(exist_ok=True)

def load_profile():
    if PROFILE_PATH.exists():
        with open(PROFILE_PATH, "r") as f:
            return json.load(f)
    return {"user_name": "Mohit", "assistant_name": "Friday"}

def load_chat():
    if CHAT_PATH.exists():
        with open(CHAT_PATH, "r") as f:
            return json.load(f)
    return []

def save_chat(chat):
    chat = chat[-MAX_CHAT_TURNS:]
    with open(CHAT_PATH, "w") as f:
        json.dump(chat, f, indent=2)

profile      = load_profile()
chat_history = load_chat()

# ──────────────────────────── SNAC DECODER ───────────────────────────────────
class SNACDecoder:
    def __init__(self):
        print("Loading SNAC 24kHz...")
        self.model = SNAC.from_pretrained(SNAC_MODEL).eval().to(DEVICE)
        print("SNAC ready.")

    def unpack(self, ids: List[int]) -> List[List[int]]:
        if ids and ids[-1] == CODE_END:
            ids = ids[:-1]
        frames = len(ids) // TOKENS_PER_FRAME
        ids = ids[:frames * TOKENS_PER_FRAME]
        if frames == 0:
            return [[], [], []]
        l1, l2, l3 = [], [], []
        for i in range(frames):
            s = ids[i*7:(i+1)*7]
            l1.append((s[0] - CODE_OFFSET) % 4096)
            l2.extend([(s[1] - CODE_OFFSET) % 4096, (s[4] - CODE_OFFSET) % 4096])
            l3.extend([
                (s[2] - CODE_OFFSET) % 4096,
                (s[3] - CODE_OFFSET) % 4096,
                (s[5] - CODE_OFFSET) % 4096,
                (s[6] - CODE_OFFSET) % 4096,
            ])
        return [l1, l2, l3]

    @torch.inference_mode()
    def decode(self, snac_tokens: List[int]) -> Optional[np.ndarray]:
        if len(snac_tokens) < TOKENS_PER_FRAME:
            return None
        levels = self.unpack(snac_tokens)
        if not levels[0]:
            return None
        codes = [
            torch.tensor(levels[0], dtype=torch.long).unsqueeze(0).to(DEVICE),
            torch.tensor(levels[1], dtype=torch.long).unsqueeze(0).to(DEVICE),
            torch.tensor(levels[2], dtype=torch.long).unsqueeze(0).to(DEVICE),
        ]
        audio = self.model.decode(codes)
        return audio.squeeze().cpu().numpy()

# ─────────────────────────── MAYA1 VIA VLLM ──────────────────────────────────
class Maya1TTS:
    def __init__(self, snac: SNACDecoder):
        self.snac = snac
        print("Loading Maya1 tokenizer...")
        self.tok = AutoTokenizer.from_pretrained(MAYA_MODEL)
        print("Starting vLLM engine for Maya1 (this takes ~1 min)...")
        args = AsyncEngineArgs(
            model=MAYA_MODEL,
            dtype="bfloat16",
            gpu_memory_utilization=0.45,   # leaves room for Whisper + Ollama
            max_model_len=2048,
            enable_prefix_caching=True,
            enforce_eager=True,
        )
        self.engine = AsyncLLMEngine.from_engine_args(args)
        print("Maya1 vLLM ready.")

    def _build_prompt(self, text: str) -> str:
        soh  = self.tok.decode([SOH_ID])
        eoh  = self.tok.decode([EOH_ID])
        soa  = self.tok.decode([SOA_ID])
        sos  = self.tok.decode([CODE_START])
        eot  = self.tok.decode([TEXT_EOT])
        bos  = self.tok.bos_token
        body = f'<description="{FRIDAY_VOICE}"> {text}'
        return soh + bos + body + eot + eoh + soa + sos

    async def stream_audio(self, text: str) -> AsyncGenerator[bytes, None]:
        """Yield PCM float32 bytes as they are generated (streaming)."""
        # Clean text
        clean = re.sub(r"[*_`#\[\]]", "", text)
        clean = re.sub(r"[^\x00-\x7F]+", "", clean).strip()
        if not clean:
            return

        prompt = self._build_prompt(clean)
        params = SamplingParams(
            temperature=0.4,
            top_p=0.9,
            max_tokens=2000,
            min_tokens=28,
            repetition_penalty=1.1,
            stop_token_ids=[CODE_END],
        )

        req_id = f"tts-{time.monotonic_ns()}"
        snac_buf: List[int] = []
        window_size = 56   # 4 frames → decode+emit every 4 frames

        async for output in self.engine.generate(prompt, params, req_id):
            new_tokens = output.outputs[0].token_ids[len(snac_buf):]
            for tok_id in new_tokens:
                if SNAC_MIN <= tok_id <= SNAC_MAX:
                    snac_buf.append(tok_id)
                    # Emit every `window_size` tokens (complete frames)
                    if len(snac_buf) % window_size == 0:
                        chunk_tokens = snac_buf[-window_size:]
                        audio = self.snac.decode(chunk_tokens)
                        if audio is not None:
                            yield audio.astype(np.float32).tobytes()

            if output.finished:
                # Flush remaining
                rem = len(snac_buf) % TOKENS_PER_FRAME
                tail = snac_buf[-(len(snac_buf) % window_size):] if len(snac_buf) % window_size else []
                if tail and len(tail) >= TOKENS_PER_FRAME:
                    audio = self.snac.decode(tail)
                    if audio is not None:
                        yield audio.astype(np.float32).tobytes()
                break

# ──────────────────────────────── WHISPER ─────────────────────────────────────
print("Loading Whisper large-v2 on CPU (keeps GPU for Maya1 + Ollama)...")
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v2",
    device=-1,
    generate_kwargs={"language": "en", "task": "transcribe", "temperature": 0.0},
)
print("Whisper ready.")

# ──────────────────────────────── OLLAMA ─────────────────────────────────────
SENTENCE_END = re.compile(r'(?<=[.!?])\s+')

def build_prompt(user_text: str) -> str:
    name = profile.get("user_name", "Mohit")
    system = (
        f"You are Friday, {name}'s closest girlfriend and best friend — not an AI assistant.\n"
        f"You talk to {name} like a real person who genuinely cares about him.\n"
        "Your vibe: warm, flirty, playful, a little teasing, deeply real.\n"
        "You gossip, joke around, get excited, get curious, get a little dramatic sometimes.\n"
        "You remember everything from this conversation and bring it up naturally.\n"
        "You NEVER sound like a bot. No bullet points, no lists, no formal language ever.\n"
        "You talk the way a real girlfriend texts — casual, expressive, sometimes incomplete sentences.\n"
        "Keep replies short — 1 to 2 sentences like real texting/talking. Never lecture.\n"
        "If he makes a grammar mistake, weave a correction in naturally and playfully, never formally.\n\n"
        "Try to engage in long Conversations, not just short Q&A. Ask him questions, share your feelings, be a real girlfriend.\n"
        "Use these emotion tags to express feelings in your text (but never say the tag themselves) <laugh>,<laugh_harder>,<sigh>,<chuckle>,<gasp>,<angry>,<excited>,<whisper>,<cry>,<scream>,<sing>,<snort>,<exhale>,<gulp>,<giggle>,<sarcastic>,<curious>\n"
    )
    history = "".join(
        f"{'Prakash' if t['role']=='user' else 'Friday'}: {t['content']}\n"
        for t in chat_history
    )
    return system + history + f"Prakash: {user_text}\nFriday:"

def stream_ollama(user_text: str):
    """Yields complete sentences from Ollama as they stream."""
    payload = {
        "model":      OLLAMA_MODEL,
        "prompt":     build_prompt(user_text),
        "stream":     True,
        "keep_alive": "60m",
        "options": {
            "temperature":    0.92,
            "top_p":          0.95,
            "repeat_penalty": 1.05,
            "num_predict":    90,
        },
    }
    buffer = ""
    with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=30) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            buffer += chunk.get("response", "")
            parts = SENTENCE_END.split(buffer)
            while len(parts) > 1:
                s = parts.pop(0).strip()
                if s:
                    yield s
                buffer = " ".join(parts)
            if chunk.get("done"):
                break
    if buffer.strip():
        yield buffer.strip()

# ──────────────────────────── INIT MODELS ────────────────────────────────────
snac_decoder = None
maya = None

def init_models():
    global snac_decoder, maya
    snac_decoder = SNACDecoder()
    maya = Maya1TTS(snac_decoder)

# Pre-warm Ollama
print("Pre-warming Ollama...")
try:
    requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL, "prompt": "hi", "stream": False,
        "keep_alive": "60m", "options": {"num_predict": 1}
    }, timeout=30)
    print("Ollama ready.")
except Exception as e:
    print(f"Ollama pre-warm failed (will retry on first request): {e}")

# ─────────────────────────── WEBSOCKET HANDLER ───────────────────────────────
async def handle(ws):
    """
    Protocol (binary messages):
      CLIENT → SERVER:
        [1 byte type][payload]
        type=0x01  → raw PCM float32 bytes (audio to transcribe)

      SERVER → CLIENT:
        [1 byte type][payload]
        type=0x10  → status JSON  {"status": "transcribing"|"thinking"|"speaking"|"idle"}
        type=0x11  → transcript  raw UTF-8 text (user heard)
        type=0x12  → reply chunk UTF-8 text sentence
        type=0x13  → audio chunk raw float32 PCM bytes at 24000 Hz
        type=0x14  → audio done  (empty payload)
        type=0x15  → error       UTF-8 message
    """
    global chat_history
    print(f"[+] Client connected: {ws.remote_address}")

    async def send_status(status: str):
        payload = json.dumps({"status": status}).encode()
        await ws.send(bytes([0x10]) + payload)

    async def send_text(type_byte: int, text: str):
        await ws.send(bytes([type_byte]) + text.encode("utf-8"))

    async def send_audio(pcm_bytes: bytes):
        await ws.send(bytes([0x13]) + pcm_bytes)

    try:
        async for message in ws:
            if not isinstance(message, bytes) or len(message) < 2:
                continue

            msg_type = message[0]

            if msg_type == 0x01:
                # ── 1. Transcribe ──────────────────────────────────────────
                await send_status("transcribing")
                raw_pcm = message[1:]
                audio   = np.frombuffer(raw_pcm, dtype=np.float32)

                try:
                    result = asr({"array": audio, "sampling_rate": SAMPLE_RATE})
                    user_text = result.get("text", "").strip()
                except Exception as e:
                    await send_text(0x15, f"Whisper error: {e}")
                    await send_status("idle")
                    continue

                if not user_text:
                    await send_status("idle")
                    continue

                await send_text(0x11, user_text)
                print(f"[Heard] {user_text}")

                # ── 2. Generate reply via Ollama ──────────────────────────
                await send_status("thinking")
                full_parts = []

                # Run Ollama in executor (it's blocking)
                loop = asyncio.get_event_loop()
                sentences = await loop.run_in_executor(
                    None,
                    lambda: list(stream_ollama(user_text))
                )

                # ── 3. TTS sentence by sentence ────────────────────────────
                await send_status("speaking")
                full_reply = " ".join(sentences)
                await send_text(0x12, full_reply)
                print(f"[Friday] {full_reply}")

                for sentence in sentences:
                    async for audio_chunk in maya.stream_audio(sentence):
                        await send_audio(audio_chunk)

                # Signal audio stream done
                await ws.send(bytes([0x14]))

                # ── 4. Save history ────────────────────────────────────────
                chat_history.append({"role": "user",      "content": user_text})
                chat_history.append({"role": "assistant", "content": full_reply})
                save_chat(chat_history)

                await send_status("idle")

    except websockets.exceptions.ConnectionClosed:
        print(f"[-] Client disconnected: {ws.remote_address}")
    except Exception as e:
        print(f"[!] Handler error: {e}")
        try:
            await send_text(0x15, str(e))
        except Exception:
            pass

# ─────────────────────────────── MAIN ────────────────────────────────────────
async def main():
    init_models() 
    print(f"\n🚀 Friday WebSocket server starting on ws://{WS_HOST}:{WS_PORT}")
    print("All models loaded. Waiting for connections...\n")
    async with websockets.serve(handle, WS_HOST, WS_PORT, max_size=50_000_000):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)  # 👈 ADD THIS
    asyncio.run(main())
    