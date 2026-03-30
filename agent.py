import asyncio
import queue
import threading
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import websockets
import io

# ─────────────────────────────── CONFIG ──────────────────────────────────────
RUNPOD_WS_URL  = "wss://jgmzprse0jclmr-8765.proxy.runpod.net"   

SAMPLE_RATE    = 16000
MAYA_RATE      = 24000   # server sends 24kHz PCM

# VAD
VAD_SILENCE_THRESHOLD = 0.015
VAD_SILENCE_DURATION  = 1.0
VAD_MAX_DURATION      = 12.0
VAD_MIN_DURATION      = 0.8

# ─────────────────────────── SHARED STATE ────────────────────────────────────
audio_level_queue: queue.Queue = queue.Queue()

agent_state = {
    "status":     "idle",
    "user_text":  "",
    "reply_text": "",
}

# Callback hooks — ui.py assigns these
on_status_change  = None   # fn(status: str)
on_transcript     = None   # fn(text: str)
on_reply          = None   # fn(text: str)

# ─────────────────────────── VAD RECORDING ───────────────────────────────────
def record_audio() -> np.ndarray:
    """Capture mic audio using VAD. Returns float32 array at SAMPLE_RATE."""
    frames          = []
    silence_start   = None
    speech_detected = False
    start_time      = time.time()
    WINDOW_SIZE     = 5

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())
        audio_level_queue.put(float(np.sqrt(np.mean(indata ** 2))))

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="float32", callback=callback):
        while True:
            elapsed = time.time() - start_time
            if elapsed > VAD_MAX_DURATION:
                break
            time.sleep(0.05)
            if not frames:
                continue
            recent = frames[-WINDOW_SIZE:] if len(frames) >= WINDOW_SIZE else frames
            rms = float(np.sqrt(np.mean(np.concatenate(recent) ** 2)))
            if rms > VAD_SILENCE_THRESHOLD:
                speech_detected = True
                silence_start   = None
            elif speech_detected:
                if silence_start is None:
                    silence_start = time.time()
                elif (time.time() - silence_start) >= VAD_SILENCE_DURATION:
                    if elapsed >= VAD_MIN_DURATION:
                        break

    audio = np.concatenate(frames, axis=0).squeeze()
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return audio

# ─────────────────────────── AUDIO PLAYER ────────────────────────────────────
class StreamingPlayer:
    """Plays incoming float32 PCM chunks in real-time with minimal buffering."""

    def __init__(self, sample_rate: int = MAYA_RATE):
        self._started = False
        self._min_buffer = 1   # start fast
        self._max_buffer = 6   # allow growth
        self._q: queue.Queue = queue.Queue()
        self._sr = sample_rate
        self._stream = sd.OutputStream(
            samplerate=self._sr,
            channels=1,
            dtype="float32",
            blocksize=4096,
            callback=self._cb,
        )
        self._buf = np.zeros(0, dtype=np.float32)
        self._stream.start()
        self._done = False

    def _cb(self, outdata, frames, time_info, status):
        # Pre-buffering logic
        if not self._started:
            qsize = self._q.qsize()

            if qsize < self._min_buffer:
                outdata[:, 0] = 0
                return

            # if buffer is healthy → start immediately
            if qsize >= self._min_buffer:
                self._started = True
        needed = frames
        out = np.zeros(needed, dtype=np.float32)
        filled = 0

        while filled < needed:
            if len(self._buf) == 0:
                try:
                    chunk = self._q.get(timeout=0.1)
                    if chunk is None:
                        self._done = True
                        break
                    self._buf = chunk
                except queue.Empty:
                    break
            take = min(needed - filled, len(self._buf))
            out[filled:filled+take] = self._buf[:take]
            self._buf = self._buf[take:]
            filled += take

        outdata[:, 0] = out

    def push(self, pcm_bytes: bytes):
        arr = np.frombuffer(pcm_bytes, dtype=np.float32)
        self._q.put(arr)

    def finish(self):
        self._q.put(None)

    def wait_done(self):
        """Block until playback buffer is drained."""
        while not self._q.empty() or len(self._buf) > 0:
            time.sleep(0.05)
        time.sleep(0.1)

    def close(self):
        self._stream.stop()
        self._stream.close()

# ──────────────────────────── MAIN AGENT CALL ────────────────────────────────
def _set_status(status: str):
    agent_state["status"] = status
    if on_status_change:
        on_status_change(status)


async def _run_agent_async():
    """Core async pipeline: record → send → receive TTS and play."""
    # 1. Record
    _set_status("listening")
    audio = record_audio()

    # 2. Connect and send
    _set_status("transcribing")
    raw_pcm = audio.astype(np.float32).tobytes()
    message = bytes([0x01]) + raw_pcm

    player = None
    user_text  = ""
    reply_text = ""

    try:
        async with websockets.connect(
            RUNPOD_WS_URL,
            max_size=50_000_000,
            ping_interval=20,
            ping_timeout=60,
        ) as ws:
            await ws.send(message)

            async for msg in ws:
                if not isinstance(msg, bytes) or len(msg) < 1:
                    continue
                mtype   = msg[0]
                payload = msg[1:]

                if mtype == 0x10:   # status
                    import json
                    d = json.loads(payload)
                    _set_status(d.get("status", "idle"))

                elif mtype == 0x11:  # transcript
                    user_text = payload.decode("utf-8")
                    agent_state["user_text"] = user_text
                    if on_transcript:
                        on_transcript(user_text)

                elif mtype == 0x12:  # reply text
                    reply_text = payload.decode("utf-8")
                    agent_state["reply_text"] = reply_text
                    if on_reply:
                        on_reply(reply_text)

                elif mtype == 0x13:  # audio chunk
                    if player is None:
                        player = StreamingPlayer(sample_rate=MAYA_RATE)
                    player.push(payload)

                elif mtype == 0x14:  # audio done
                    if player:
                        player.finish()
                        player.wait_done()
                        player.close()
                        player = None
                    break

                elif mtype == 0x15:  # error
                    print(f"[Server Error] {payload.decode()}")
                    break

    except Exception as e:
        print(f"[WebSocket Error] {e}")
        if player:
            player.close()

    finally:
        _set_status("idle")

    return user_text, reply_text


def run_agent_once():
    
    return asyncio.run(_run_agent_async())