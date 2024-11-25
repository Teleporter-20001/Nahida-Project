# imports
import pygame
from Settings import *
from Bullets import *


class Game(object):

    def __init__(self) -> None:
        
        pygame.init()                                           # Game init
        self.fps_clock = pygame.time.Clock()
        self.running: bool = True
        self.is_pause: bool = False

        self.window_size = WINDOW_SIZE                           # Window create
        self.window = pygame.display.set_mode(self.window_size)
        self.window.fill("purple")
        pygame.display.set_caption(VERSION)

        self.bullets = pygame.sprite.Group()                    # Characters init
        self.bullets.add(
            Bullet(
                10, 
                0, 100, 
                10
            )
        )
        for i in range(24):
            self.bullets.add(
                Bullet_roundspray(
                    7, 
                    300, 200, 
                    4, 
                    i * pi / 12
                )
            )
        for i in range(24):
            self.bullets.add(
                Bullet_roundspray(
                    7, 
                    300, 200, 
                    -4, 
                    i * pi / 12, 
                    IMG_BULLET_SUPER
                )
            )
        

        self.score = 0                                          # Score init

        pygame.display.flip()                                   # Scene init


    # 4. event detection
    def event_handle(self):

        for event in pygame.event.get():
            
            if event.type == pygame.QUIT:
                exit()


    def collision_handle(self):
        pass


    def game_update(self):

        pygame.display.update()
        self.window.fill("purple")
        self.bullets.update()
        self.bullets.draw(self.window)
        pygame.display.update()
        self.fps_clock.tick(FPS)


    def gameover(self):

        print(f"Your final score: {self.score}")