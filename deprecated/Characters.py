# This project is licensed under the MIT License (Non-Commercial Use Only).
# Please see the license file in this source distribution for full license terms.

from typing import Any
import pygame
from Settings import *

class Boss(pygame.sprite.Sprite):

    def __init__(
        self, 
        health: int = 150, 
        position: tuple[int, int] = (WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 4), 
        img: str = IMG_BOSS
    ) -> None:

        super().__init__()

        self.radius = 50
        self.image = pygame.transform.scale(
            pygame.image.load(img).convert_alpha(), 
            (2 * self.radius, 2 * self.radius)
        )

        self.rect = self.image.get_rect()
        self.rect.center = position
        self.pos_x, self.pos_y = position
        self.dest_x, self.dest_y = position
        self.border = (WINDOW_SIZE[0], WINDOW_SIZE[1] / 2)

        self.health = health
        self.bullets = pygame.sprite.Group()

        self.begin_roundspray()


    def move(self):
        pass


    def update(self):
        if self.health <= 0:
            self.kill()


    def begin_roundspray(self):
        pygame.time.set_timer(BOSS_BEGIN_ROUNDSPRAY_EVENT, BOSS_BEGIN_ROUNDSPRAY_TIME)
        

class Traveller(pygame.sprite.Sprite):

    ORIGINSPEED: int = 7
    DIAGONAL_SPEED_ADAPT = {
        (0, 0)  : 1, 
        (0, 1)  : 1, 
        (0, -1) : 1, 
        (-1, 0) : 1, 
        (1, 0)  : 1, 
        (1, 1)  : 0.707, 
        (1, -1) : 0.707, 
        (-1, 1) : 0.707, 
        (-1, -1): 0.707
    }

    def __init__(self, img = IMG_TRAVELLER, size: int = 50) -> None:

        super().__init__()

        self.image = pygame.transform.scale(
            pygame.image.load(img).convert_alpha(), 
            (size, int(size * 1222 / 720))
        )

        self.rect = self.image.get_rect()
        self.rect.center = (WINDOW_SIZE[0] // 2, int(WINDOW_SIZE[1] * 0.8))
        self.pos_x, self.pos_y = self.rect.center
        self.direction_x = 0
        self.direction_y = 0
        self.dest_x = 0
        self.dest_y = 0

        self.speed = self.ORIGINSPEED
        self.is_slow = False
        self.border = WINDOW_SIZE

        self.radius = 5
        self.health = 10
        self.is_strongest = False
        self.is_dead = False

        self.is_shooting = False

    def is_inboard(self):
        res = [True, True]
        if self.dest_x - self.rect.width / 2 < 0 or self.dest_x + self.rect.width / 2 > self.border[0]:
            res[0] = False
        if self.dest_y - self.rect.height / 2 < 0 or self.dest_y + self.rect.height / 2 > self.border[1]:
            res[1] = False
        return res


    def calc_dest(self):
        adaptor = self.DIAGONAL_SPEED_ADAPT.get((self.direction_x, self.direction_y))
        if adaptor is None:
            adaptor = 1
        self.dest_x = self.pos_x + self.speed * self.direction_x * adaptor
        self.dest_y = self.pos_y + self.speed * self.direction_y * adaptor


    def AI_give_direction(self, x: float, y: float):
        self.direction_x = x
        self.direction_y = y


    def update(self):

        self.calc_dest()
        # print(self.dest_x, self.dest_y)

        movable = self.is_inboard()
        if movable[0]:
            self.pos_x = self.dest_x
            self.rect.centerx = round(self.pos_x)
        if movable[1]:
            self.pos_y = self.dest_y
            self.rect.centery = round(self.pos_y)
        # if not (movable[0] and movable[1]):
        # print(movable, self.pos_x, self.pos_y)

        # if self.health <= 0 and not self.is_dead:
        #     print("post died event")
        #     self.is_dead = True
        #     pygame.event.post(pygame.event.Event(TRAVELLER_DIED_EVENT))
            


class AI_Traveller(Traveller):

    def AI_give_direction(self, x: float, y: float):
        self.direction_x = x
        self.direction_y = y

    def calc_dest(self):
        # self.AI_give_direction()
        super().calc_dest()