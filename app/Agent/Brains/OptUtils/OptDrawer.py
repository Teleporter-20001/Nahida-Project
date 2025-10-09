import numbers
import time
from collections.abc import Callable
from typing import Optional
import numpy as np
from typing_extensions import Self
import multiprocessing as mp
import pickle
import struct
from threading import Lock

from app.common.Settings import Settings
from app.common.utils import printgreen, printred, printyellow, printblue


def _get_traj(x0, y0, vx0, vy0, ax, ay):
    def traj(step):
        dt = 1.0 / Settings.FPS
        t = dt * step  # 总时间
        x = x0 + vx0 * t + 0.5 * ax * t * t
        y = y0 + vy0 * t + 0.5 * ay * t * t
        return x, y
    return traj


class OptimizedDebugData:
    """Optimized data structure for debug information transmission.
    
    Uses binary packing to reduce serialization overhead.
    """
    
    def __init__(self):
        self.player_x: float = 0.0
        self.player_y: float = 0.0
        self.boss_x: float = 0.0
        self.boss_y: float = 0.0
        self.boss_vx: float = 0.0
        self.boss_vy: float = 0.0
        self.boss_ax: float = 0.0
        self.boss_ay: float = 0.0
        self.target_x: float = 0.0
        self.target_y: float = 0.0
        self.bullets: np.ndarray = np.zeros((10, 7), dtype=np.float32)  # (x, y, vx, vy, ax, ay, r)
        self.bullet_count: int = 0
        self.player_traj: list[tuple[float, float]] = []
    
    def pack(self) -> bytes:
        """Pack data into binary format for efficient transmission."""
        # Pack basic data: player(2) + boss(6) + target(2) = 10 floats + 1 int for bullet_count
        header = struct.pack('10f1i', 
                           self.player_x, self.player_y,
                           self.boss_x, self.boss_y, self.boss_vx, self.boss_vy, self.boss_ax, self.boss_ay,
                           self.target_x, self.target_y, self.bullet_count)
        
        # Pack bullet data efficiently
        bullets_data = self.bullets[:self.bullet_count].tobytes()
        
        # Pack trajectory data (simplified - only first few points to reduce size)
        traj_count = min(len(self.player_traj), 20)  # Limit trajectory points
        traj_data = struct.pack('i', traj_count)
        for i in range(traj_count):
            traj_data += struct.pack('2f', self.player_traj[i][0], self.player_traj[i][1])
        
        return header + bullets_data + traj_data
    
    @classmethod
    def unpack(cls, data: bytes) -> 'OptimizedDebugData':
        """Unpack binary data back to debug data structure."""
        obj = cls()
        
        # Unpack header
        header_size = struct.calcsize('10f1i')
        header_data = struct.unpack('10f1i', data[:header_size])
        
        obj.player_x, obj.player_y = header_data[0], header_data[1]
        obj.boss_x, obj.boss_y, obj.boss_vx, obj.boss_vy, obj.boss_ax, obj.boss_ay = header_data[2:8]
        obj.target_x, obj.target_y = header_data[8], header_data[9]
        obj.bullet_count = header_data[10]
        
        # Unpack bullets
        offset = header_size
        bullets_size = obj.bullet_count * 7 * 4  # 7 floats * 4 bytes each
        if bullets_size > 0:
            bullets_bytes = data[offset:offset + bullets_size]
            bullets_array = np.frombuffer(bullets_bytes, dtype=np.float32).reshape(obj.bullet_count, 7)
            obj.bullets[:obj.bullet_count] = bullets_array
        
        # Unpack trajectory
        offset += bullets_size
        traj_count = struct.unpack('i', data[offset:offset + 4])[0]
        offset += 4
        
        obj.player_traj = []
        for i in range(traj_count):
            x, y = struct.unpack('2f', data[offset:offset + 8])
            obj.player_traj.append((x, y))
            offset += 8
        
        return obj


def _window_process_main(queue: mp.Queue, predict_len: int, fps_limit: int = 10):
    import pygame
    import sys
    import os

    try:
        os.environ['SDL_VIDEO_WINDOW_POS'] = "2500, 600"
        pygame.init()
        screen = pygame.display.set_mode((Settings.window_width, Settings.window_height))
        pygame.display.set_caption("OptBrain Debug Window (Optimized)")
        clock = pygame.time.Clock()
        os.environ.pop('SDL_VIDEO_WINDOW_POS')
    except Exception as e:
        printred(f'Failed to initialize pygame window: {e}')
        return
    
    font = pygame.font.SysFont("Arial", 18)
    last_data = None
    frame_skip_counter = 0
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        screen.fill(pygame.Color('white'))
        
        # Try to get new data, but don't block
        new_data = None
        try:
            while True:  # Drain the queue to get the latest data
                new_data = queue.get_nowait()
        except:
            pass  # Queue is empty
        
        if new_data is not None:
            if isinstance(new_data, bytes):
                # Handle optimized binary data
                try:
                    data = OptimizedDebugData.unpack(new_data)
                    last_data = data
                except Exception as e:
                    printred(f'Failed to unpack debug data: {e}')
                    data = last_data
            else:
                # Handle legacy dictionary data
                data = new_data
                last_data = data
        else:
            data = last_data
        
        if data is not None:
            if isinstance(data, OptimizedDebugData):
                # Handle optimized data structure
                player_x, player_y = data.player_x, data.player_y
                boss_x, boss_y = data.boss_x, data.boss_y
                boss_vx, boss_vy, boss_ax, boss_ay = data.boss_vx, data.boss_vy, data.boss_ax, data.boss_ay
                target_x, target_y = data.target_x, data.target_y
                player_traj = data.player_traj
                bullets = [(data.bullets[i, 0], data.bullets[i, 1], data.bullets[i, 2], 
                           data.bullets[i, 3], data.bullets[i, 4], data.bullets[i, 5], data.bullets[i, 6])
                          for i in range(data.bullet_count)]
            else:
                # Handle legacy dictionary data
                player_x, player_y = data["player"]
                boss_x, boss_y, boss_vx, boss_vy, boss_ax, boss_ay = data["boss"]
                bullets = data["bullets"]
                target_x, target_y = data.get("target", (None, None))
                player_traj = data.get("player_traj", [])
            
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
            for (bx, by, bvx, bvy, bax, bay, br) in bullets:
                if np.hypot(bx, by) < 6: 
                    continue  # too close to player, must be a zero initial value
                if not isinstance(br, numbers.Number):
                    printred(f'invalid bullet radius: {br}, type: {type(br)}')
                    continue
                try:
                    # Convert br to integer for pygame.draw.circle
                    # Handle various number types safely by converting to numpy array first
                    br_array = np.array(br)
                    br_scalar = br_array.item()  # extract scalar value safely
                    br_int = int(br_scalar)
                    if br_int <= 0:
                        continue  # skip invalid radius
                    pygame.draw.circle(screen, (255, 0, 0), (int(bx + player_x), int(by + player_y)), br_int)
                    btraj = _get_traj(bx, by, bvx, bvy, bax, bay)
                    for i in range(1, predict_len):
                        traj_x, traj_y = btraj(i)
                        traj_br_int = max(1, br_int // 3)  # ensure minimum radius of 1
                        pygame.draw.circle(screen, (90, 0, 0), (int(traj_x + player_x), int(traj_y + player_y)), traj_br_int)
                except (TypeError, ValueError, OverflowError, AttributeError) as e:
                    printred(f'Error converting bullet radius to int: {br} (type: {type(br)}), error: {e}')
                    continue
                
            # Draw target point
            if target_x is not None and target_y is not None:
                pygame.draw.circle(screen, (0, 255, 0), (int(target_x), int(target_y)), 10, 4)
                text_surface = font.render(f'Target: ({target_x:.1f}, {target_y:.1f})', True, (0, 128, 0))
                screen.blit(text_surface, (10, 10))
                
            # Draw player trajectory
            if player_traj:
                for i in range(1, len(player_traj)):
                    pygame.draw.line(screen, (200, 50, 100), (int(player_traj[i-1][0]), int(player_traj[i-1][1])), (int(player_traj[i][0]), int(player_traj[i][1])), width=4)

            pygame.display.update()
            if Settings.save_record:
                pygame.image.save(screen, os.path.join('data', f'debug_{time.time()}.png'))
            
        # Limit FPS to reduce CPU usage
        clock.tick(fps_limit)


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
        # Check if debug mode is enabled
        self._debug_enabled = Settings.enable_debug_drawer
        self._use_optimized_data = Settings.debug_data_compression
        self._fps_limit = Settings.debug_drawer_fps_limit
        
        # Data storage
        self._debug_data = OptimizedDebugData()
        self.bullets: list[tuple[float, float, float, float, float, float, float]] = []  # legacy format
        self.predict_len: int = 10
        
        # Process communication
        self._queue: Optional[mp.Queue] = None
        self._process: Optional[mp.Process] = None
        self._lock = Lock()  # Thread safety for data updates
        
        # Performance tracking
        self._frame_counter = 0
        self._last_send_time = 0.0
        self._send_interval = 1.0 / max(1, self._fps_limit)  # Minimum interval between sends
        
        if not self._debug_enabled:
            printyellow("OptDrawer: Debug mode disabled in Settings. No performance overhead.")
        else:
            printgreen("OptDrawer: Debug mode enabled with optimizations.")

    def set_predict_len(self, predict_len: int):
        self.predict_len = predict_len
        
    def write_player_data(self, x: float, y: float):
        if not self._debug_enabled:
            return
        with self._lock:
            self._debug_data.player_x = x
            self._debug_data.player_y = y
        
    def write_boss_data(self, x: float, y: float, vx: float, vy: float, ax: float, ay: float):
        if not self._debug_enabled:
            return
        with self._lock:
            self._debug_data.boss_x = x
            self._debug_data.boss_y = y
            self._debug_data.boss_vx = vx
            self._debug_data.boss_vy = vy
            self._debug_data.boss_ax = ax
            self._debug_data.boss_ay = ay
        
    def write_target_data(self, x: float, y: float):
        if not self._debug_enabled:
            return
        with self._lock:
            self._debug_data.target_x = x
            self._debug_data.target_y = y
        
    def write_bullet_data(self, idx: int, x: float, y: float, vx: float, vy: float, ax: float, ay: float, r: int):
        if not self._debug_enabled:
            return
        
        with self._lock:
            if idx < 10:  # Limit to array size
                self._debug_data.bullets[idx] = [x, y, vx, vy, ax, ay, r]
                self._debug_data.bullet_count = max(self._debug_data.bullet_count, idx + 1)
            
            # Also maintain legacy format for compatibility
            if idx >= len(self.bullets):
                if idx >= Settings.consider_bullets_num:
                    return  # Skip instead of error
                self.bullets.extend([(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)] * (idx - len(self.bullets) + 1))
            self.bullets[idx] = (x, y, vx, vy, ax, ay, r)
        
    def write_player_traj(self, traj: list[tuple[float, float]]):
        if not self._debug_enabled:
            return
        with self._lock:
            self._debug_data.player_traj = traj[:20]  # Limit trajectory points for performance

    def draw(self):
        """Call this function in each frame to push data to subprocess and update the debug window
        """
        if not self._debug_enabled or self._queue is None:
            return
            
        # Rate limiting to reduce queue overhead
        current_time = time.time()
        if current_time - self._last_send_time < self._send_interval:
            return
        
        self._last_send_time = current_time
        self._frame_counter += 1
        
        try:
            with self._lock:
                if self._use_optimized_data:
                    # Send binary packed data
                    packed_data = self._debug_data.pack()
                    if self._queue.qsize() < 3:  # Prevent queue overflow
                        self._queue.put_nowait(packed_data)
                else:
                    # Send legacy dictionary data
                    data = {
                        "player": (self._debug_data.player_x, self._debug_data.player_y),
                        "boss": (self._debug_data.boss_x, self._debug_data.boss_y, 
                                self._debug_data.boss_vx, self._debug_data.boss_vy, 
                                self._debug_data.boss_ax, self._debug_data.boss_ay),
                        "bullets": self.bullets,
                        "target": (self._debug_data.target_x, self._debug_data.target_y),
                        "player_traj": self._debug_data.player_traj
                    }
                    if self._queue.qsize() < 3:  # Prevent queue overflow
                        self._queue.put_nowait(data)
        except Exception as e:
            # Silently ignore queue full errors to prevent blocking
            pass
            
    def start_window(self):
        if not self._debug_enabled:
            printblue('OptDrawer: Debug mode disabled, window not started')
            return
            
        if self._process is not None and self._process.is_alive():
            printyellow('OptDrawer: debug window already running')
            return
            
        try:
            # Use smaller queue to reduce memory overhead
            self._queue = mp.Queue(maxsize=2)
            self._process = mp.Process(
                target=_window_process_main, 
                args=(self._queue, self.predict_len, self._fps_limit), 
                daemon=True
            )
            self._process.start()
            printgreen(f'OptDrawer: debug window started (FPS limit: {self._fps_limit}, optimized: {self._use_optimized_data})')
        except Exception as e:
            printred(f'OptDrawer: Failed to start debug window: {e}')
            self._debug_enabled = False
        
    def stop_window(self):
        if self._process is not None and self._process.is_alive():
            try:
                self._process.terminate()
                self._process.join(timeout=1.0)  # Add timeout
                if self._process.is_alive():
                    self._process.kill()  # Force kill if needed
            except Exception as e:
                printred(f'OptDrawer: Error stopping window: {e}')
            finally:
                self._process = None
                self._queue = None
                printgreen('OptDrawer: debug window stopped')
        