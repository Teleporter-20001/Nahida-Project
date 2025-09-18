from app.characters.BaseEnemyBullet import BaseEnemyBullet
from math import *

class SprayBullet(BaseEnemyBullet):
    
    ORIGIN_SPEED = 200

    def __init__(self, x: float, y: float, radius: int, img_path: str, theta: float, vx: float = 0, vy: float = 0, target_size: tuple[int, int] = (-1, -1)):
        r = 10
        self.birth_x = x
        self.birth_y = y
        x += r * cos(theta)
        y += r * sin(theta)
        super().__init__(x, y, radius, img_path, vx, vy, target_size)
        
    def generate_speed(self):
        dx = self.posx - self.birth_x
        dy = self.posy - self.birth_y
        r = hypot(dx, dy)
        vel_radial_x = self.ORIGIN_SPEED * dx / r if r != 0 else 0
        vel_radial_y = self.ORIGIN_SPEED * dy / r if r != 0 else 0
        omega = 0.2
        vel_tangential_x = -vel_radial_y * omega
        vel_tangential_y = vel_radial_x * omega
        self.vx = vel_radial_x + vel_tangential_x
        self.vy = vel_radial_y + vel_tangential_y

