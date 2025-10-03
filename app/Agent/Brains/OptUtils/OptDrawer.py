from collections.abc import Callable
from typing import Optional
import numpy as np
from typing_extensions import Self
import multiprocessing as mp

from app.common.Settings import Settings
from app.common.utils import printgreen, printred, printyellow


def _get_traj(x0, y0, vx0, vy0, ax, ay):
    def traj(step):
        dt = 1.0 / Settings.FPS
        t = dt * step  # 总时间
        x = x0 + vx0 * t + 0.5 * ax * t * t
        y = y0 + vy0 * t + 0.5 * ay * t * t
        return x, y
    return traj


def _window_process_main(queue: mp.Queue, predict_len: int):
    import pygame
    import sys
    import os

    try:
        os.environ['SDL_VIDEO_WINDOW_POS'] = "2500, 600"
        pygame.init()
        screen = pygame.display.set_mode((Settings.window_width, Settings.window_height))
        pygame.display.set_caption("OptBrain Debug Window")
        clock = pygame.time.Clock()
        os.environ.pop('SDL_VIDEO_WINDOW_POS')
    except Exception as e:
        printred(f'Failed to initialize pygame window: {e}')
        return
    
    font = pygame.font.SysFont("Arial", 18)
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        screen.fill(pygame.Color('white'))
        
        try:
            data = queue.get_nowait()
        except Exception as e:
            printred(f'error in reading data from queue: {sys.exc_info()}\n-------------\n{e}')
            data = None
        
        if data is not None:
            player_x, player_y = data["player"]
            boss_x, boss_y, boss_vx, boss_vy, boss_ax, boss_ay = data["boss"]
            bullets = data["bullets"]
            
            # Draw player
            pygame.draw.circle(screen, (127, 127, 127), (int(player_x), int(player_y)), 8)
            
            # Draw boss
            pygame.draw.circle(screen, (0, 0, 255), (int(boss_x + player_x), int(boss_y + player_y)), 50)   # 转回绝对位置，方便看

            # Draw boss trajectory
            boss_traj = _get_traj(boss_x, boss_y, boss_vx, boss_vy, boss_ax, boss_ay)
            for i in range(1, predict_len):
                traj_x, traj_y = boss_traj(i)
                pygame.draw.circle(screen, (0, 0, 90), (int(traj_x + player_x), int(traj_y + player_y)), 5)

            # Draw bullets
            for (bx, by, bvx, bvy, bax, bay) in bullets:
                if np.hypot(bx, by) < 6: 
                    continue  # too close to player, must be a zero initial value
                pygame.draw.circle(screen, (255, 0, 0), (int(bx + player_x), int(by + player_y)), 13)
                btraj = _get_traj(bx, by, bvx, bvy, bax, bay)
                for i in range(1, predict_len):
                    traj_x, traj_y = btraj(i)
                    pygame.draw.circle(screen, (90, 0, 0), (int(traj_x + player_x), int(traj_y + player_y)), 4)

        pygame.display.update()
        clock.tick(Settings.FPS)


class OptDrawer:

    _instance: Optional["OptDrawer"] = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(OptDrawer, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def instance(cls) -> "OptDrawer":
        if cls._instance is None:
            cls._instance = OptDrawer()
        return cls._instance

    def __init__(self):
        
        self.player_x = 0.0
        self.player_y = 0.0
        self.boss_x = 0.0
        self.boss_y = 0.0
        self.boss_vx = 0.0
        self.boss_vy = 0.0
        self.boss_ax = 0.0
        self.boss_ay = 0.0
        self.bullets: list[tuple[float, float, float, float, float, float]] = []  # list of (x, y, vx, vy, ax, ay)
        self.predict_len: int = 10
        
        self._queue: Optional[mp.Queue] = None # pass data to subprocess to draw debug info
        self._process: Optional[mp.Process] = None

    def set_predict_len(self, predict_len: int):
        self.predict_len = predict_len
        
    def write_player_data(self, x: float, y: float):
        self.player_x = x
        self.player_y = y
        
    def write_boss_data(self, x: float, y: float, vx: float, vy: float, ax: float, ay: float):
        self.boss_x = x
        self.boss_y = y
        self.boss_vx = vx
        self.boss_vy = vy
        self.boss_ax = ax
        self.boss_ay = ay
        
    def write_bullet_data(self, idx: int, x: float, y: float, vx: float, vy: float, ax: float, ay: float, traj: Callable[[int], tuple[float, float]]):
        if idx >= len(self.bullets):
            printyellow(f'bullet idx {idx} out of range {len(self.bullets)}, extend bullets list')
            self.bullets.extend([(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)] * (idx - len(self.bullets) + 1))
        self.bullets[idx] = (x, y, vx, vy, ax, ay)

    def draw(self):
        """Call this function in each frame to push data to subprocess and update the debug window
        """
        if self._queue is not None:
            data = {
                "player": (self.player_x, self.player_y),
                "boss": (self.boss_x, self.boss_y, self.boss_vx, self.boss_vy, self.boss_ax, self.boss_ay),
                "bullets": self.bullets,
            }
            self._queue.put(data)
            
    def start_window(self):
        if self._process is not None and self._process.is_alive():
            printyellow('OptDrawer: debug window already running')
            return
        self._queue = mp.Queue(maxsize=4)
        self._process = mp.Process(target=_window_process_main, args=(self._queue, self.predict_len), daemon=True)
        self._process.start()
        printgreen('OptDrawer: debug window started')
        
    def stop_window(self):
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join()
            self._process = None
            self._queue = None
            printgreen('OptDrawer: debug window stopped')
        