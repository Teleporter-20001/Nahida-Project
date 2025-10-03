import os
from typing import Iterable, Optional
import multiprocessing as mp
import threading
import time
import numpy as np
import json
from dataclasses import dataclass

from matplotlib import pyplot as plt

from app.common.utils import printgreen, printred, printyellow

# optional torch support
try:
    import torch
except Exception:
    torch = None


@dataclass
class _IPCMessage:
    """Simple container for messages sent to the debug window process."""
    type: str
    payload: object


def _window_process_main(queue: mp.Queue, width: int, height: int, title: str, fps: int = 60):
    """Run in a separate process: create a pygame window and render frames received on the queue.

    This function imports pygame inside the child process to avoid importing SDL at module import
    time in the parent.
    """
    try:
        import pygame
    except Exception as e:
        # cannot initialize pygame in child; print and exit
        print(f"Debugger child: failed to import pygame: {e}")
        return

    try:
        os.environ['SDL_VIDEO_WINDOW_POS'] = "2560, 600"
        pygame.init()
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        os.environ.pop('SDL_VIDEO_WINDOW_POS')
    except Exception as e:
        print(f"Debugger child: pygame init failed: {e}")
        return

    clock = pygame.time.Clock()

    # default data
    data = np.zeros((3, 3), dtype=float)

    running = True
    font = None
    try:
        font = pygame.font.SysFont("Arial", 18)
    except Exception:
        try:
            font = pygame.font.Font(None, 18)
        except Exception:
            font = None

    while running:
        # handle window events (allow user to close)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # drain queue (non-blocking) - keep only the latest frame to avoid backlog
        try:
            while True:
                msg = queue.get_nowait()
                if not isinstance(msg, dict):
                    continue
                if msg.get('type') == 'frame':
                    payload = msg.get('payload')
                    try:
                        arr = np.array(payload, dtype=float)
                        if arr.size == 9:
                            arr = arr.reshape((3, 3))
                        elif arr.shape == (3, 3):
                            pass
                        else:
                            continue
                        data = arr
                    except Exception:
                        continue
                elif msg.get('type') == 'close':
                    running = False
                    break
        except Exception:
            # queue empty
            pass

        # render
        screen.fill((30, 30, 30))
        try:
            minv = float(np.nanmin(data))
            maxv = float(np.nanmax(data))
        except Exception:
            minv, maxv = 0.0, 1.0
        if not np.isfinite(minv) or not np.isfinite(maxv):
            minv, maxv = 0.0, 1.0
        denom = maxv - minv if maxv > minv else 1.0

        cell_margin = 6
        cell_w = (width - cell_margin * 4) / 3.0
        cell_h = (height - cell_margin * 4) / 3.0

        for i in range(3):
            for j in range(3):

                val = float(data[i, j])
                # rel = (val - minv) / denom
                cmap = plt.get_cmap('coolwarm')
                relr, relg, relb, rela = cmap(val)
                r = int(relr * 255)
                g = int(relg * 255)
                b = int(relb * 255)
                # r = int(50 + rel * 205)
                # g = int(50 + rel * 155)
                # b = int(120 - rel * 100)
                color = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
                x = int(cell_margin + j * (cell_w + cell_margin))
                y = int(cell_margin + i * (cell_h + cell_margin))
                w = int(cell_w)
                h = int(cell_h)
                rect = pygame.Rect(x, y, w, h)
                pygame.draw.rect(screen, color, rect)
                txt = f"{val:.3f}"
                if font:
                    brightness = (color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114) / 255.0
                    txt_color = (255, 255, 255) if brightness < 0.5 else (0, 0, 0)
                    surf = font.render(txt, True, txt_color)
                    sw, sh = surf.get_size()
                    tx = x + (w - sw) // 2
                    ty = y + (h - sh) // 2
                    screen.blit(surf, (tx, ty))

        pygame.display.flip()
        clock.tick(fps)

    try:
        pygame.quit()
    except Exception:
        pass


class Debugger:
    """Singleton Debugger client.

    By default, the Debugger runs in "client mode": calls to `add_data()` will push
    the latest frame into an internal multiprocessing.Queue. The child process
    (launched with `start_window()`) reads from the queue and displays the heatmap
    in a separate OS window using pygame. This avoids conflicts with the main
    process's pygame display.
    """
    _instance: Optional["Debugger"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Debugger, cls).__new__(cls)
        return cls._instance

    @classmethod
    def instance(cls) -> "Debugger":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, width: int = 480, height: int = 480, title: str = "Debugger Heatmap"):
        if getattr(self, '_initialized', False):
            return
        self._initialized = True

        self.width = width
        self.height = height
        self.title = title

        # local copy of latest weights (kept for optional in-process render)
        self._weights = np.zeros((3, 3), dtype=float)
        self._lock = threading.Lock()

        # IPC to child
        self._queue: Optional[mp.Queue] = None
        self._process: Optional[mp.Process] = None

        printgreen(f"Debugger (ipc window) constructed: {width}x{height}")

    # -------------------- control child window --------------------
    def start_window(self):
        """Start the child process that displays the independent debug window.

        Safe to call multiple times; subsequent calls have no effect if already started.
        """
        if self._process is not None and self._process.is_alive():
            return
        # create a Queue (multiprocessing-safe)
        self._queue = mp.Queue(maxsize=4)
        self._process = mp.Process(target=_window_process_main, args=(self._queue, self.width, self.height, self.title))
        self._process.daemon = True
        self._process.start()
        printgreen("Debugger child process started")

    def stop_window(self, timeout: float = 1.0):
        """Request child to close and join the process."""
        try:
            if self._queue is not None:
                try:
                    # send close request
                    self._queue.put({'type': 'close', 'payload': None}, block=False)
                except Exception:
                    pass
            if self._process is not None:
                self._process.join(timeout)
                if self._process.is_alive():
                    try:
                        self._process.terminate()
                    except Exception:
                        pass
        finally:
            try:
                if self._queue is not None:
                    self._queue.close()
            except Exception:
                pass
            self._queue = None
            self._process = None
            printgreen("Debugger child process stopped")

    # -------------------- data API (called from main thread) --------------------
    def add_data(self, values: Iterable[float]):
        """Accept 9 floats or a 3x3 iterable. Non-blocking: pushes latest frame to the child queue.

        If the child queue is full or not started, the call is still non-blocking and only
        updates the local buffer.
        """
        # printyellow(f'get values: {values}')
        # support torch tensors directly
        if torch is not None and isinstance(values, torch.Tensor):
            try:
                arr = values.detach().cpu().numpy()
            except Exception:
                arr = np.array(list(values), dtype=float)

        else:
            arr = np.array(list(values), dtype=float)

        if arr.size == 9:
            arr = arr.reshape((3, 3))
        elif arr.shape == (3, 3):
            pass
        else:
            raise ValueError("add_data expects 9 values or a 3x3 iterable")

        # printyellow(f'get data: {arr}')

        with self._lock:
            self._weights[:, :] = arr

        # non-blocking push to child (keep only latest)
        if self._queue is not None:
            try:
                # Use put_nowait; if queue full, discard oldest by doing one get then put
                try:
                    self._queue.put({'type': 'frame', 'payload': arr.tolist()}, block=False)
                except Exception:
                    try:
                        _ = self._queue.get_nowait()
                    except Exception:
                        pass
                    try:
                        self._queue.put({'type': 'frame', 'payload': arr.tolist()}, block=False)
                    except Exception:
                        pass
            except Exception:
                pass

    # backward-compatible in-process render: call from main loop if user prefers
    def render(self, surface):
        """Fallback: in-process render (same as before) if caller prefers to blit into main screen.

        This will use the local buffer updated by `add_data()`.
        """
        try:
            import pygame
        except Exception:
            return
        if surface is None:
            return
        surface.fill((30, 30, 30))
        with self._lock:
            data = self._weights.copy()
        try:
            minv = float(np.nanmin(data))
            maxv = float(np.nanmax(data))
        except Exception:
            minv, maxv = 0.0, 1.0
        if not np.isfinite(minv) or not np.isfinite(maxv):
            minv, maxv = 0.0, 1.0
        denom = maxv - minv if maxv > minv else 1.0
        cell_w = (surface.get_width() - 6 * 4) / 3.0
        cell_h = (surface.get_height() - 6 * 4) / 3.0
        try:
            font = pygame.font.SysFont("Arial", 16)
        except Exception:
            font = pygame.font.Font(None, 16)
        for i in range(3):
            for j in range(3):
                val = float(data[i, j])
                rel = (val - minv) / denom
                r = int(50 + rel * 205)
                g = int(50 + rel * 155)
                b = int(120 - rel * 100)
                color = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
                x = int(6 + j * (cell_w + 6))
                y = int(6 + i * (cell_h + 6))
                w = int(cell_w)
                h = int(cell_h)
                rect = pygame.Rect(x, y, w, h)
                pygame.draw.rect(surface, color, rect)
                txt = f"{val:.3f}"
                brightness = (color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114) / 255.0
                txt_color = (255, 255, 255) if brightness < 0.5 else (0, 0, 0)
                surf = font.render(txt, True, txt_color)
                sw, sh = surf.get_size()
                tx = x + (w - sw) // 2
                ty = y + (h - sh) // 2
                surface.blit(surf, (tx, ty))

    def __del__(self):
        try:
            self.stop_window()
        except Exception:
            pass


if __name__ == '__main__':
    dbg = Debugger.instance()
    dbg.start_window()

    import random

    try:
        while True:
            vals = [random.random() for _ in range(9)]
            dbg.add_data(vals)
            time.sleep(0.15)
    except KeyboardInterrupt:
        dbg.stop_window()