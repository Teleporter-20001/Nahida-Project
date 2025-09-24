from math import cos, sin

from app.characters.BaseEnemyBullet import BaseEnemyBullet
from deprecated.Settings import ORIGINSPEED


class StraightEnemyBullet(BaseEnemyBullet):

    ORIGIN_SPEED = 150

    def __init__(self, x: float, y: float, radius: int, img_path: str, direction: float):
        super().__init__(x, y, radius, img_path)
        self.direction = direction

    def generate_speed(self):
        self.vx = self.ORIGIN_SPEED * cos(self.direction)
        self.vy = self.ORIGIN_SPEED * sin(self.direction)