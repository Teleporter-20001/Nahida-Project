import pygame

from app.common.Settings import Settings
from app.common.utils import to_ints


class BaseChar(pygame.sprite.Sprite):

    ORIGIN_SPEED = 5

    def __init__(self, x: float, y: float, radius: int, image_path: str, vx: float=0, vy: float=0, target_size: tuple[int, int] = (-1, -1), no_out: bool = False):
        pygame.sprite.Sprite.__init__(self)
        self._posx = x
        self._posy = y
        self.vx = vx
        self.vy = vy
        self._radius = radius
        self._is_alive = True
        temp_img = pygame.image.load(image_path).convert_alpha()
        if target_size == (-1, -1):
            target_size = temp_img.get_size()
        self.image = pygame.transform.scale(temp_img, target_size)
        self.rect = self.image.get_rect()
        self.rect.center = to_ints((self._posx, self._posy))
        self._scale = target_size
        self.no_out = no_out

    @property
    def posx(self):
        return self._posx
    @property
    def posy(self):
        return self._posy
    @property
    def radius(self):
        return self._radius
    @property
    def is_alive(self):
        return self._is_alive
    @property
    def scale(self):
        return self._scale



    def update(self):
        self._posx += self.vx / Settings.FPS
        if self.no_out and (self.posx < 0 or self.posx > Settings.window_width):
            self._posx -= self.vx / Settings.FPS
        self.rect.centerx = round(self._posx)
        self._posy += self.vy / Settings.FPS
        if self.no_out and (self.posy < 0 or self.posy > Settings.window_height):
            self._posy -= self.vy / Settings.FPS
        self.rect.centery = round(self._posy)


    def draw(self, surface: pygame.Surface):
        surface.blit(self.image, self.rect)