from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

tts.tts_to_file(
    text="Prakash, I remember you. And I’m here.",
    speaker_wav="voice.wav",
    file_path="ok.wav",
    language="en"
)

print("OK")
