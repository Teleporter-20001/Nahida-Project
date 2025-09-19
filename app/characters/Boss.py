from math import *

from app.characters.BaseChar import BaseChar
import random

from app.common.Settings import Settings


class Boss(BaseChar):
    
    ORIGIN_SPEED = 60

    def __init__(
            self,
            x: float,
            y: float,
            radius: int,
            img_path: str,
            health: int = 250,
            vx: float=0,
            vy: float=0,
            target_size: tuple[int, int]=(-1, -1),
    ):
        """
        vx and vy are useless. we use ORIGIN_SPEED and direction to generate velocity.
        """
        super().__init__(x, y, radius, img_path, vx, vy, target_size, no_out=True)
        self.maxy = int(Settings.window_height / 2.5)
        self.health: int = health
        self.max_health: int = health
        self.direction = random.random() * 2 * pi
        
    @property
    def alive(self):
        return self.health > 0
    
    def change_direction(self):
        self.direction = random.random() * 2 * pi
    
    def generate_speed(self):
        self.vx = self.ORIGIN_SPEED * cos(self.direction)
        self.vy = self.ORIGIN_SPEED * sin(self.direction)
        if self.rect.left + self.vx / Settings.FPS < 0 or self.rect.right + self.vx / Settings.FPS > Settings.window_width:
            self.direction = pi - self.direction
            self.vx = -self.vx
        if self.rect.top + self.vy / Settings.FPS < 0 or self.rect.bottom + self.vy / Settings.FPS > self.maxy:
            self.direction = -self.direction
            self.vy = -self.vy
        
    def update(self):
        self.generate_speed()
        super().update()