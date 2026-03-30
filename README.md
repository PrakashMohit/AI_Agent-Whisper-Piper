<h1 align="center" id="title">AI_Agent-Whisper-Piper</h1><br>

<p align="center"><img src="https://socialify.git.ci/PrakashMohit/AI_Agent-Whisper-Piper/image?custom_language=Hugging+Face&amp;font=KoHo&amp;language=1&amp;name=1&amp;owner=1&amp;pattern=Formal+Invitation&amp;stargazers=1&amp;theme=Dark" alt="project-image"></p>


# AI_Agent-Whisper - ~~Maya-1~~ Piper
Created an AI AGENT which uses whisper STT and ~~Maya~~ Piper TTS locally along with Ollama to provide an assistant behaviour vocally.

# Update 1 ----><br>
I am using Piper TTS as i dont have GPU or Collab Pro

# Update 2 ----><br>
Ive added a smaller whisper model and updated my program to play audio throug sounddevice library directly instead of calling ffplay and introduced threading which helped me reduce the latency from 1 minute to 10 seconds.

# Update 3 ----><br>
Added basic memory to get enough context to have a basic chat with the agent without context drop.


# Update 4 ----><br>
Now using Ollama Gemma3:4b along with whisper large and official MAYA 1 as tts and running this on cloud using runpod on A40 48 GB Single Gpu 

<h1>Getting Started With Repo</h1>
<h2>1 Installing Dependencies</h2>

```
pip install -r requirements.txt
```
