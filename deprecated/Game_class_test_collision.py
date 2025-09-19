# This project is licensed under the MIT License (Non-Commercial Use Only).
# Please see the license file in this source distribution for full license terms.

# imports
import pygame
import sys
from Settings import *
from Bullets import *
from Characters import *
from Event_manager import *
from Collision_manager import *
from Scene_updater import * 


class Game(object):
    

    def __init__(self) -> None:
        
        pygame.init()                                           # Game init

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

        self.collision_manager = Collision_manager(
            self.player, 
            self.bullets_player, 
            self.boss, 
            self.bullets_boss)
        self.scene_updater = Scene_updater(
            self.window, 
            self.player, 
            self.players, 
            self.bullets_player, 
            self.boss, 
            self.bosses, 
            self.bullets_boss
        )
        self.event_manager = Event_manager(
            self.player, 
            self.bullets_player, 
            self.boss, 
            self.bullets_boss, 
            self.collision_manager, 
            self.scene_updater
        )
             
        pygame.display.flip()                                   # Scene init


    def gameover(self):

        print(f"Your final score: {self.scene_updater.score}")
        pygame.quit()
        sys.exit()


if __name__ == "__main__":

    gameobj = Game()

    while gameobj.event_manager.state_of_game == GAME_1:
        gameobj.event_manager.event_handle()
        if not gameobj.event_manager.is_pause:
            gameobj.collision_manager.collision_handle()
            gameobj.scene_updater.game_update()

    gameobj.player.is_dead = False
    gameobj.player.health = 3
    print("change game state", gameobj.event_manager.state_of_game)

    while gameobj.event_manager.state_of_game == GAME_2:
        print("game2start", gameobj.event_manager.state_of_game, gameobj.player.health)
        gameobj.event_manager.event_handle()
        if not gameobj.event_manager.is_pause:
            gameobj.collision_manager.collision_handle()
            gameobj.scene_updater.game_update()
    
    gameobj.gameover()
