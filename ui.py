import math
import random
import threading
import asyncio
import queue
import json
import time
import ctypes
import numpy as np
import pygame
import pygame.gfxdraw

# ── try to import agent pieces ────────────────────────────────────────────────
try:
    import agent as _agent
    import sounddevice as sd
    import websockets
    _AGENT_OK = True
except ImportError:
    _AGENT_OK = False
    print("[ui] agent/sounddevice/websockets not found — running in demo mode")

# ─────────────────────────── CONFIG ──────────────────────────────────────────
WIN_SIZE   = 200          # window is a square; circle fills it
R          = 88           # orb radius
CX = CY    = WIN_SIZE // 2

FPS        = 60
N_BARS     = 48

SAMPLE_RATE = 16000
MAYA_RATE   = 24000

# ─────────────────────────── PALETTE ─────────────────────────────────────────
C_BG        = (12,  12,  18)
C_RING_OUT  = (60,  60,  90)
C_RING_MID  = (45,  45,  72)
C_RING_IN   = (30,  30,  50)
C_CORE      = (22,  22,  36)
C_CORE_EDGE = (55,  55,  85)
C_BAR_HI    = (160, 160, 220)
C_BAR_MID   = (80,  80,  130)
C_BAR_LO    = (38,  38,  62)
C_GLOW      = (100, 100, 200)
C_GLOW_REC  = (220, 60,  60)
C_GLOW_SPK  = (60,  160, 220)
C_DOT       = (120, 120, 180)
C_WHITE     = (230, 230, 255)
C_RED       = (220, 60,  60)

# ─────────────────────────── HELPERS ─────────────────────────────────────────

def aa_circle(surf, cx, cy, r, color, width=1):
    if r <= 0:
        return
    if width <= 1:
        pygame.gfxdraw.aacircle(surf, int(cx), int(cy), int(r), color)
    else:
        for i in range(width):
            rr = int(r) - i
            if rr > 0:
                pygame.gfxdraw.aacircle(surf, int(cx), int(cy), rr, color)


def filled_aa_circle(surf, cx, cy, r, color):
    if r <= 0:
        return
    pygame.gfxdraw.filled_circle(surf, int(cx), int(cy), int(r), color)
    pygame.gfxdraw.aacircle(surf, int(cx), int(cy), int(r), color)


def glow_circle(surf, cx, cy, r, color, layers=7, max_alpha=80):
    for i in range(layers, 0, -1):
        gr    = int(r + i * 5)
        alpha = int(max_alpha * (1 - i / (layers + 1)))
        tmp   = pygame.Surface((gr * 2 + 2, gr * 2 + 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(tmp, gr + 1, gr + 1, gr, (*color, alpha))
        surf.blit(tmp, (int(cx) - gr - 1, int(cy) - gr - 1))


def aa_arc(surf, cx, cy, r, start_deg, end_deg, color, width=2):
    steps = max(60, int(abs(end_deg - start_deg) * r / 60))
    pts   = []
    for i in range(steps + 1):
        t   = start_deg + (end_deg - start_deg) * i / steps
        rad = math.radians(t)
        pts.append((cx + r * math.cos(rad), cy - r * math.sin(rad)))
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        pygame.draw.line(surf, color, (int(x1), int(y1)), (int(x2), int(y2)), width)
        pygame.gfxdraw.pixel(surf, int(x1), int(y1), color)
        pygame.gfxdraw.pixel(surf, int(x2), int(y2), color)


# ─────────────────────── PTT RECORDING ───────────────────────────────────────
_ptt_frames: list = []
_ptt_level_q: queue.Queue = queue.Queue()
_ptt_stream = None
_ptt_lock   = threading.Lock()


def _ptt_cb(indata, frames, time_info, status):
    _ptt_frames.append(indata.copy())
    _ptt_level_q.put(float(np.sqrt(np.mean(indata ** 2))))


def ptt_start():
    global _ptt_stream, _ptt_frames
    if not _AGENT_OK:
        return
    with _ptt_lock:
        _ptt_frames = []
        _ptt_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1,
            dtype="float32", callback=_ptt_cb
        )
        _ptt_stream.start()


def ptt_stop_and_send(on_status, on_transcript, on_reply, on_done):
    global _ptt_stream
    if not _AGENT_OK:
        on_done()
        return
    with _ptt_lock:
        if _ptt_stream is None:
            on_done()
            return
        _ptt_stream.stop()
        _ptt_stream.close()
        _ptt_stream = None
        frames = list(_ptt_frames)

    def _run():
        asyncio.run(_ws_pipeline(frames, on_status, on_transcript, on_reply, on_done))

    threading.Thread(target=_run, daemon=True).start()


async def _ws_pipeline(frames, on_status, on_transcript, on_reply, on_done):
    if not frames:
        on_done()
        return
    audio = np.concatenate(frames, axis=0).squeeze().astype(np.float32)
    audio /= (np.max(np.abs(audio)) + 1e-8)
    raw_pcm = audio.tobytes()
    message = bytes([0x01]) + raw_pcm
    on_status("transcribing")
    player = None
    try:
        async with websockets.connect(
            _agent.RUNPOD_WS_URL,
            max_size=50_000_000,
            ping_interval=20,
            ping_timeout=60,
        ) as ws:
            await ws.send(message)
            async for msg in ws:
                if not isinstance(msg, bytes) or len(msg) < 1:
                    continue
                mtype, payload = msg[0], msg[1:]
                if mtype == 0x10:
                    on_status(json.loads(payload).get("status", "idle"))
                elif mtype == 0x11:
                    on_transcript(payload.decode("utf-8"))
                elif mtype == 0x12:
                    on_reply(payload.decode("utf-8"))
                elif mtype == 0x13:
                    if player is None:
                        player = _agent.StreamingPlayer(sample_rate=MAYA_RATE)
                    player.push(payload)
                elif mtype == 0x14:
                    if player:
                        player.finish(); player.wait_done(); player.close()
                        player = None
                    break
                elif mtype == 0x15:
                    print(f"[Server Error] {payload.decode()}")
                    break
    except Exception as e:
        print(f"[WebSocket Error] {e}")
        if player:
            player.close()
    finally:
        on_done()


# ─────────────────────────── ORB UI ──────────────────────────────────────────
class FridayOrb:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Friday")

        self.screen = pygame.display.set_mode(
            (WIN_SIZE, WIN_SIZE),
            pygame.NOFRAME | pygame.SRCALPHA
        )
        self.surf = pygame.Surface((WIN_SIZE, WIN_SIZE), pygame.SRCALPHA)
        self.clock  = pygame.time.Clock()

        self._tick      = 0
        self._amp       = 0.0
        self._status    = "idle"
        self._recording = False
        self._busy      = False

        self._phases    = [random.uniform(0, math.pi * 2) for _ in range(N_BARS)]
        self._bar_seeds = [random.uniform(0.4, 1.0)        for _ in range(N_BARS)]

        self._win_rect  = (0, 0)
        self._status_q: queue.Queue = queue.Queue()

        # Position window bottom-right
        info   = pygame.display.Info()
        sw, sh = info.current_w, info.current_h
        wx     = sw - WIN_SIZE - 24
        wy     = sh - WIN_SIZE - 60
        self._move_win(wx, wy)
        self._win_rect = (wx, wy)

    def _move_win(self, x, y):
        try:
            import ctypes
            hwnd = pygame.display.get_wm_info()["window"]
            ctypes.windll.user32.SetWindowPos(
                hwnd, -1, x, y, WIN_SIZE, WIN_SIZE, 0x0001
            )
            self._win_rect = (x, y)
        except Exception:
            pass

    def _set_topmost(self):
        try:
            import ctypes
            hwnd = pygame.display.get_wm_info()["window"]
            ctypes.windll.user32.SetWindowPos(
                hwnd, -1, 0, 0, 0, 0, 0x0002 | 0x0001
            )
        except Exception:
            pass

    # ── CALLBACKS ────────────────────────────────────────────────────────────
    def _on_status(self, s):     self._status_q.put(("status",     s))
    def _on_transcript(self, t): self._status_q.put(("transcript", t))
    def _on_reply(self, t):      self._status_q.put(("reply",      t))
    def _on_done(self):          self._status_q.put(("done",       None))

    def _flush_status_q(self):
        while not self._status_q.empty():
            try:
                kind, val = self._status_q.get_nowait()
                if kind == "status":
                    self._status = val
                elif kind == "transcript":
                    print(f"[YOU] {val}")
                elif kind == "reply":
                    print(f"[FRIDAY] {val}")
                elif kind == "done":
                    self._busy = False
                    self._status = "idle"
            except Exception:
                pass

    # ── PTT ──────────────────────────────────────────────────────────────────
    def _handle_click(self):
        if self._busy:
            return
        if not self._recording:
            self._recording = True
            self._status    = "listening"
            ptt_start()
        else:
            self._recording = False
            self._busy      = True
            self._status    = "transcribing"
            ptt_stop_and_send(
                self._on_status, self._on_transcript,
                self._on_reply,  self._on_done
            )

    # ── DRAW ─────────────────────────────────────────────────────────────────
    def _draw(self):
        surf   = self.surf
        t      = self._tick
        amp    = self._amp
        status = self._status
        surf.fill((0, 0, 0, 0))

        # Glow
        if self._recording:
            gc, ga = C_GLOW_REC, 90 + int(30 * math.sin(t * 0.15))
        elif status == "speaking":
            gc, ga = C_GLOW_SPK, 70 + int(25 * math.sin(t * 0.12))
        elif status == "thinking":
            gc, ga = C_GLOW,     55 + int(20 * math.sin(t * 0.10))
        else:
            gc, ga = C_GLOW,     35 + int(12 * math.sin(t * 0.06))
        glow_circle(surf, CX, CY, R - 4, gc, layers=8, max_alpha=ga)

        r_out = R
        r_mid = int(R * 0.76)
        r_in  = int(R * 0.56)

        ring_col = (*C_RED, 200) if self._recording else (*C_RING_OUT, 160)
        aa_circle(surf, CX, CY, r_out, ring_col, width=1)

        for deg in [20, 70, 130, 200, 280, 340]:
            ang = math.radians(deg)
            dx_ = r_out * math.cos(ang)
            dy_ = r_out * math.sin(ang)
            dc  = (*C_RED, 180) if self._recording else (*C_DOT, 160)
            filled_aa_circle(surf, CX + dx_, CY + dy_, 2, dc)

        # Bars
        bar_w   = max(2, int(R * 0.035))
        bar_gap = bar_w + max(1, int(R * 0.022))
        total_w = N_BARS * bar_gap
        start_x = CX - total_w // 2

        for i in range(N_BARS):
            bx = start_x + i * bar_gap
            if self._recording or status == "listening":
                wave  = math.sin(self._phases[i]) * amp
                bar_h = int(R * (0.08 + 0.58 * max(0, wave)) * self._bar_seeds[i])
                frac  = min(1.0, bar_h / (R * 0.35))
                col   = (
                    int(C_BAR_LO[0] + frac * (C_BAR_HI[0] - C_BAR_LO[0])),
                    int(C_BAR_LO[1] + frac * (C_BAR_HI[1] - C_BAR_LO[1])),
                    int(C_BAR_LO[2] + frac * (C_BAR_HI[2] - C_BAR_LO[2])),
                    220
                )
            elif status in ("thinking", "transcribing"):
                wave  = math.sin(self._phases[i] + t * 0.04)
                bar_h = int(R * (0.06 + 0.20 * abs(wave)) * self._bar_seeds[i])
                col   = (*C_BAR_MID, 180)
            elif status == "speaking":
                wave  = math.sin(self._phases[i]) * amp
                bar_h = int(R * (0.10 + 0.48 * abs(wave)) * self._bar_seeds[i])
                frac  = min(1.0, bar_h / (R * 0.30))
                col   = (60, 140, int(160 + frac * 60), 210)
            else:
                pulse = 0.5 + 0.5 * math.sin(t * 0.04 + i * 0.3)
                bar_h = int(R * 0.055 * self._bar_seeds[i] * (0.7 + 0.3 * pulse))
                col   = (*C_BAR_LO, 140)

            half = max(1, bar_h // 2)
            rr   = max(1, bar_w // 2)
            pygame.draw.rect(surf, col,
                             pygame.Rect(int(bx), CY - half, bar_w, half * 2),
                             border_radius=rr)

        aa_circle(surf, CX, CY, r_mid, (*C_RING_MID, 130), width=1)
        filled_aa_circle(surf, CX, CY, r_in, (*C_CORE, 255))
        aa_circle(surf, CX, CY, r_in, (*C_CORE_EDGE, 180), width=1)

        # Centre icon
        if self._recording:
            pulse = 1.0 + 0.22 * math.sin(t * 0.20)
            pr    = int(R * 0.17 * pulse)
            glow_circle(surf, CX, CY, pr, C_RED, layers=4, max_alpha=80)
            filled_aa_circle(surf, CX, CY, pr, (*C_RED, 240))
        elif status == "idle":
            pulse = 1.0 + 0.10 * math.sin(t * 0.055)
            pr    = int(R * 0.17 * pulse)
            aa_circle(surf, CX, CY, pr, (*C_WHITE, 180), width=2)
        elif status in ("listening", "transcribing"):
            mic_r = int(R * 0.155)
            aa_circle(surf, CX, CY, mic_r, (*C_WHITE, 200), width=2)
            lw = max(1, int(R * 0.032))
            for off in [-lw, 0, lw]:
                pygame.draw.line(surf, (*C_WHITE, 200),
                                 (CX + off, CY - mic_r + 3),
                                 (CX + off, CY + mic_r - 3), lw)
            stand = int(R * 0.085)
            pygame.draw.line(surf, (*C_WHITE, 180),
                             (CX, CY + mic_r), (CX, CY + mic_r + stand), 2)
            pygame.draw.line(surf, (*C_WHITE, 180),
                             (CX - stand, CY + mic_r + stand),
                             (CX + stand, CY + mic_r + stand), 2)
        elif status == "speaking":
            sw2, sh2 = int(R * 0.10), int(R * 0.13)
            pts = [(CX-sw2, CY-sh2//2),(CX,CY-sh2),(CX,CY+sh2),(CX-sw2,CY+sh2//2)]
            pygame.draw.polygon(surf, (*C_GLOW_SPK, 220), pts)
            for wi in range(1, 3):
                wr = int(R * 0.09 * wi)
                aa_arc(surf, CX+2, CY, wr, -50, 50, (*C_GLOW_SPK, 180), width=2)
        elif status == "thinking":
            arc_r   = int(R * 0.19)
            start_a = (t * 3) % 360
            aa_arc(surf, CX, CY, arc_r, start_a, start_a + 260,
                   (*C_WHITE, 210), width=3)
            tip_rad = math.radians(start_a + 260)
            tx = CX + arc_r * math.cos(tip_rad)
            ty = CY - arc_r * math.sin(tip_rad)
            filled_aa_circle(surf, tx, ty, 3, (*C_WHITE, 220))

        # Circular mask
        mask = pygame.Surface((WIN_SIZE, WIN_SIZE), pygame.SRCALPHA)
        mask.fill((0, 0, 0, 0))
        pygame.gfxdraw.filled_circle(mask, CX, CY, R, (255, 255, 255, 255))
        pygame.gfxdraw.aacircle(mask, CX, CY, R, (255, 255, 255, 255))
        sa = pygame.surfarray.pixels_alpha(surf)
        ma = pygame.surfarray.pixels_alpha(mask)
        sa[:] = (sa.astype(np.uint16) * ma // 255).astype(np.uint8)
        del sa, ma

        self.screen.fill((0, 0, 0, 0))
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()

    # ── UPDATE ────────────────────────────────────────────────────────────────
    def _update(self):
        levels = []
        while not _ptt_level_q.empty():
            try: levels.append(_ptt_level_q.get_nowait())
            except Exception: break

        if levels and self._recording:
            self._amp = min(1.0, float(np.mean(levels)) * 50)
        elif self._status in ("thinking", "transcribing"):
            self._amp = 0.15 + 0.10 * math.sin(self._tick * 0.15)
        elif self._status == "speaking":
            self._amp = 0.45 + 0.30 * math.sin(self._tick * 0.18)
        else:
            self._amp *= 0.88
            if self._amp < 0.005: self._amp = 0.0

        for i in range(N_BARS):
            self._phases[i] += 0.04 + 0.025 * ((i % 5) / 5)
        self._tick += 1

    # ── MAIN LOOP ────────────────────────────────────────────────────────────
    def run(self):
        start_mouse = (0, 0)
        start_win   = (0, 0)
        self._set_topmost()

        dragging   = False
        drag_off   = (0, 0)
        press_pos  = (0, 0)
        press_time = 0.0

        running = True
        while running:
            self.clock.tick(FPS)
            self._flush_status_q()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                elif event.type == pygame.MOUSEMOTION:
                    if pygame.mouse.get_pressed()[0]:

                        dx = abs(event.pos[0] - press_pos[0])
                        dy = abs(event.pos[1] - press_pos[1])

                        if dx > 4 or dy > 4:
                            dragging = True

                        if dragging:
                            try:
                                import ctypes, ctypes.wintypes

                                hwnd = pygame.display.get_wm_info()["window"]
                                

                                pt = ctypes.wintypes.POINT()
                                ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))

                                # delta movement
                                dx = pt.x - start_mouse[0]
                                dy = pt.y - start_mouse[1]

                                new_x = start_win[0] + dx
                                new_y = start_win[1] + dy

                                ctypes.windll.user32.SetWindowPos(
                                    hwnd,
                                    0,  # 👈 IMPORTANT: use HWND_TOP (not topmost)
                                    int(new_x),
                                    int(new_y),
                                    0,
                                    0,
                                    0x0001 | 0x0040  # NOSIZE | NOZORDER
                                )

                            except Exception:
                                pass
                    drag_off = event.pos

                elif event.type == pygame.MOUSEMOTION:
                    if pygame.mouse.get_pressed()[0]:
                        dx = abs(event.pos[0] - press_pos[0])
                        dy = abs(event.pos[1] - press_pos[1])

                        if dx > 4 or dy > 4:
                            dragging = True

                        if dragging:
                            try:
                                import ctypes, ctypes.wintypes

                                hwnd = pygame.display.get_wm_info()["window"]

                                pt = ctypes.wintypes.POINT()
                                ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))

                                abs_x = pt.x - drag_off[0]
                                abs_y = pt.y - drag_off[1]

                                ctypes.windll.user32.SetWindowPos(
                                    hwnd, -1, abs_x, abs_y, WIN_SIZE, WIN_SIZE, 0x0001
                                )
                            except Exception:
                                pass

                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    if not dragging:
                        if math.hypot(event.pos[0] - CX, event.pos[1] - CY) <= R:
                            self._handle_click()
                    dragging = False

            self._update()
            self._draw()

        pygame.quit()


if __name__ == "__main__":
    FridayOrb().run()