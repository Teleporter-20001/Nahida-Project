from numpy.ma.core import hypot

from app.characters.BaseEnemyBullet import BaseEnemyBullet


class AimatPlayerBullet(BaseEnemyBullet):

    ORIGIN_SPEED = 190  # a little faster than player, so player can't avoid it by moving backward

    def __init__(self, x: float, y: float, radius: int, img_path: str, target_x: float, target_y: float):

        super().__init__(x, y, radius, img_path, target_size=(2*radius, 2*radius))
        self.birth_x = x
        self.birth_y = y
        self.target_x = target_x
        self.target_y = target_y
        dx = self.target_x - self.birth_x
        dy = self.target_y - self.birth_y
        r = hypot(dx, dy)
        self.vx = self.ORIGIN_SPEED * dx / r
        self.vy = self.ORIGIN_SPEED * dy / r

    def generate_speed(self):
        pass
