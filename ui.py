import tkinter as tk
import threading
import time
import numpy as np
from agent import run_agent_once, audio_level_queue

WIDTH = 800
HEIGHT = 500
WAVE_HEIGHT = 160
BAR_COUNT = 60

class VoiceUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Friday Your Assistant")
        self.root.geometry(f"{WIDTH}x{HEIGHT}")
        self.root.configure(bg="white")

        # ---------- STATUS ----------
        self.status = tk.Label(
            self.root,
            text="Loading models…",
            font=("Segoe UI", 12),
            bg="white",
            fg="black"
        )
        self.status.pack(pady=10)

        # ---------- WAVEFORM ----------
        self.canvas = tk.Canvas(
            self.root,
            width=WIDTH,
            height=WAVE_HEIGHT,
            bg="white",
            highlightthickness=0
        )
        self.canvas.pack()

        self.center_y = WAVE_HEIGHT // 2
        self.bar_width = WIDTH // BAR_COUNT

        self.bars = []
        for i in range(BAR_COUNT):
            x = i * self.bar_width
            bar = self.canvas.create_line(
                x, self.center_y,
                x, self.center_y,
                fill="black",
                width=2
            )
            self.bars.append(bar)

        # ---------- LOG ----------
        self.chat = tk.Text(
            self.root,
            height=8,
            bg="white",
            fg="black",
            font=("Segoe UI", 11),
            state="disabled",
            wrap="word"
        )
        self.chat.pack(fill="x", padx=40, pady=10)

        # ---------- BUTTON ----------
        self.btn = tk.Button(
            self.root,
            text="🎤 Talk",
            font=("Segoe UI", 12),
            command=self.on_talk
        )
        self.btn.pack(pady=10)

        self.status.config(text="Mic ready – click Talk")

    # ---------- UI HELPERS ----------

    def log(self, speaker, text):
        self.chat.configure(state="normal")
        self.chat.insert("end", f"{speaker}: {text}\n\n")
        self.chat.see("end")
        self.chat.configure(state="disabled")

    def update_waveform(self, level):
        level = min(level * 500, WAVE_HEIGHT // 2)
        for i, bar in enumerate(self.bars):
            offset = np.sin(i / 4) * level
            self.canvas.coords(
                bar,
                i * self.bar_width, self.center_y - offset,
                i * self.bar_width, self.center_y + offset
            )

    # ---------- MAIN ACTION ----------

    def on_talk(self):
        self.btn.config(state="disabled")
        threading.Thread(target=self.run_agent, daemon=True).start()
        threading.Thread(target=self.animate_listening, daemon=True).start()

    def animate_listening(self):
        self.status.config(text="Listening… Speak now")
        while not audio_level_queue.empty() or self.btn["state"] == "disabled":
            try:
                level = audio_level_queue.get(timeout=0.1)
                self.update_waveform(level)
            except:
                pass
            time.sleep(0.03)

    def run_agent(self):
        user, reply = run_agent_once()

        self.status.config(text="Thinking…")
        time.sleep(0.5)


        self.status.config(text="Speaking…")

        if user:
            self.log("You", user)
            self.log("Friday", reply)

        self.status.config(text="Mic ready – click Talk")
        self.btn.config(state="normal")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    VoiceUI().run()
