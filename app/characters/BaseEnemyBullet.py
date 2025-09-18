from app.characters.BaseChar import BaseChar


class BaseEnemyBullet(BaseChar):

    def __init__(
            self,
            x: float,
            y: float,
            radius: int,
            img_path: str,
            vx: float=0,
            vy: float=0,
            target_size: tuple[int, int]=(-1, -1)
    ):
        super().__init__(x, y, radius, img_path, vx, vy, target_size)
        self.age = 0    # the age is measured by num of past frames


    def generate_speed(self):
        raise NotImplementedError('We should not use base enemy bullet')
    
    def update(self):
        self.age += 1
        self.generate_speed()
        super().update()