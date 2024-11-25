# imports
import pygame
from Settings import *
from Bullets import *
from Characters import *
from Event_manager import *


class Game(object):
    

    def __init__(self) -> None:
        
        pygame.init()                                           # Game init
        self.fps_clock = pygame.time.Clock()

        self.window_size = WINDOW_SIZE                           # Window create
        self.window = pygame.display.set_mode(self.window_size)
        self.window.fill(BACKGROUND_COLOR)
        pygame.display.set_caption(VERSION)


        self.boss: Boss = Boss()                                # Characters init
        self.bosses = pygame.sprite.Group()
        self.bosses.add(self.boss)
        self.player: Traveller = Traveller()
        self.players = pygame.sprite.Group()
        self.players.add(self.player)
        self.bullets_boss = pygame.sprite.Group()
        self.bullets_player = pygame.sprite.Group()

        self.event_manager = Event_manager(self.boss, self.bullets_boss, self.player, self.bullets_player)
       

        self.score = 0                                          # Score init

        pygame.display.flip()                                   # Scene init



    def collision_handle(self):
        pass


    def game_update(self):

        pygame.display.update()
        self.window.fill(BACKGROUND_COLOR)

        self.bosses.update()
        self.bosses.draw(self.window)
        self.players.update()
        self.players.draw(self.window)
        self.bullets_boss.update()
        self.bullets_boss.draw(self.window)
        self.bullets_player.update()
        self.bullets_player.draw(self.window)

        pygame.display.update()
        self.fps_clock.tick(FPS)


    def gameover(self):

        print(f"Your final score: {self.score}")
        pygame.quit()





if __name__ == "__main__":

    gameobj = Game()

    while gameobj.event_manager.state_of_game == GAME_1:
        gameobj.event_manager.event_handle()
        if not gameobj.event_manager.is_pause:
            gameobj.collision_handle()
            gameobj.game_update()
