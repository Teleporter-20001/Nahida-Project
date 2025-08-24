# This project is licensed under the MIT License (Non-Commercial Use Only).
# Please see the license file in this source distribution for full license terms.

import pygame
# from pygame.sprite import _Group
from Settings import *
from math import *

from Settings import IMG_BULLET_ROUND

class Bullet_straight(pygame.sprite.Sprite):

    ORIGINSPEED = 4

    def __init__(self, pos:tuple[int, int]) -> None:
        
        pygame.sprite.Sprite.__init__(self)
        self.radius = 10
        self.image = pygame.transform.scale(
            pygame.image.load(IMG_BULLET_ROUND).convert_alpha(), 
            (self.radius * 2 + 2, self.radius * 2 + 2)
        )
        self.rect = self.image.get_rect()
        self.rect.center = (pos[0], pos[1])
        self.centerx:float = self.rect.centerx
        self.centery:float = self.rect.centery

        self.speed:float = 1
        self.dir_theta:float = 0
        self.aimx:float = 500
        self.aimy:float = 700
        self.destx:float = 0
        self.desty:float = 0
        self.age:float = 0


    def calc_dest(self):
        self.dir_theta = atan2(self.aimy - self.centery, self.aimx - self.centerx)
        self.destx = self.centerx + self.speed * self.ORIGINSPEED * cos(self.dir_theta)
        self.desty = self.centery + self.speed * self.ORIGINSPEED * sin(self.dir_theta)

    
    def move(self):
        self.calc_dest()
        self.centerx = self.destx
        self.centery = self.desty
        self.rect.centerx = round(self.centerx)
        self.rect.centery = round(self.centery)


    def update(self):
        self.move()
        self.age += 1 / FPS


class Bullet_round(Bullet_straight):


    def __init__(self, pos:tuple[int, int], r:float, pos_0:float) -> None:
        
        pygame.sprite.Sprite.__init__(self)

        self.speed:float = 1
        self.pos_theta:float = pos_0
        self.r = r
        self.roundpoint:tuple[int, int] = pos
        self.destx:float = 0
        self.desty:float = 0
        self.age = 0

        self.radius = 10
        self.image = pygame.transform.scale(
            pygame.image.load(IMG_BULLET_ROUND).convert_alpha(), 
            (self.radius * 2 + 2, self.radius * 2 + 2)
        )
        self.rect = self.image.get_rect()
        self.rect.center = (
            int(self.roundpoint[0] + self.r * cos(self.pos_theta)), 
            int(self.roundpoint[1] + self.r * sin(self.pos_theta))
        )
        self.centerx:float = self.rect.centerx
        self.centery:float = self.rect.centery



    def calc_dest(self):
        self.pos_theta += self.speed * self.ORIGINSPEED / self.r
        self.destx = self.roundpoint[0] + self.r * cos(self.pos_theta)
        self.desty = self.roundpoint[1] + self.r * sin(self.pos_theta)


class Bullet(pygame.sprite.Sprite):

    
    def __init__(self, r: int, position_x: int, position_y: int, speed: float, img: str = IMG_BULLET_ROUND) -> None:

        super().__init__()
        self.radius = r
        self.image = pygame.transform.scale(
            pygame.image.load(img).convert_alpha(), 
            (2 * self.radius + 2, 2 * self.radius + 2)
        )
        self.rect = self.image.get_rect()
        self.rect.center = (position_x, position_y)
        self.initial_pos_x , self.initial_pos_y = position_x, position_y
        self.pos_x , self.pos_y = position_x, position_y
        self.border = WINDOW_SIZE

        self.speed = speed * ORIGINSPEED

        self.age: float = 0


    def is_inboard(self):
        if self.dest_x + self.rect.width / 2 < 0 or self.dest_x - self.rect.width / 2 > self.border[0]:
            return False
        elif self.dest_y + self.rect.height / 2 < 0 or self.dest_y - self.rect.height / 2 > self.border[1]:
            return False
        else:
            return True   
        

    def calc_dest(self):
        self.dest_x: float = self.initial_pos_x + self.speed * self.age
        self.dest_y: float = self.initial_pos_y + self.speed * self.age

    
    def move(self):
        self.calc_dest()
        self.pos_x = self.dest_x
        self.pos_y = self.dest_y
        self.rect.centerx = round(self.pos_x)
        self.rect.centery = round(self.pos_y)


    def update(self):
        self.move()
        if not self.is_inboard:
            self.kill()
        self.age += 1 / FPS


class Bullet_roundspray(Bullet):


    def __init__(self, r: int, position_x: int, position_y: int, speed: float, initial_theta: float, img: str = IMG_BULLET_ROUND) -> None:

        super().__init__(r, position_x, position_y, speed, img)

        self.r_1: float = 0
        self.initial_theta = initial_theta
        self.theta_1 = self.initial_theta

    def calc_dest(self):
        self.dest_x = self.initial_pos_x + self.r_1 * cos(self.theta_1)
        self.dest_y = self.initial_pos_y + self.r_1 * sin(self.theta_1)

    def update(self):
        super().update()
        b, a, d = 1.2, 2, 3
        self.r_1        = abs(self.speed * (sqrt(b * self.age + a) + d * self.age))
        self.theta_1    = self.initial_theta + self.speed * 0.01 * self.age


class Bullet_front(Bullet):

    def __init__(self, r: int, position_x: int, position_y: int, speed: float, dir_theta: float, img: str = IMG_BULLET_ROUND) -> None:
        super().__init__(r, position_x, position_y, speed, img)
        self.dir_theta = dir_theta

    def calc_dest(self):
        self.dest_x = self.initial_pos_x + self.speed * self.age * cos(self.dir_theta)
        self.dest_y = self.initial_pos_y + self.speed * self.age * sin(self.dir_theta)
