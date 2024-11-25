# imports
import pygame
import random
from Settings import *
from Bullets import *
from Characters import *
from Collision_manager import * 
from Scene_updater import * 


class Event_manager(object):

    def __init__(
        self, 
        window: pygame.Surface, 
        player: Traveller, 
        bullets_player: pygame.sprite.Group, 
        boss: Boss, 
        bullets_boss: pygame.sprite.Group, 
        collision_manager: Collision_manager, 
        scene_updater: Scene_updater
    ) -> None:

        self.state_of_game = GAME_1
        self.is_pause = False

        self.window = window
        self.boss = boss
        self.bullets_boss = bullets_boss
        self.player = player
        self.bullets_player = bullets_player

        self.collision_manager = collision_manager
        self.scene_updater = scene_updater

        self.up_keydown = False
        self.down_keydown = False
        self.left_keydown = False
        self.right_keydown = False

        self.behit_position = open('AI_data/behit_pos.txt', 'a')

        self.EVENT_DICT = {
            pygame.QUIT                 : self.quit, 
            TRAVELLER_DIED_EVENT        : self.next_stage, 
            pygame.KEYDOWN              : self.keydown_handle, 
            pygame.KEYUP                : self.keyup_handle, 
            BOSS_BEGIN_ROUNDSPRAY_EVENT : self.boss_begin_roundspray, 
            BOSS_ROUNDSPRAY_EVENT       : self.boss_roundspray, 
            TRAVELLER_SHOOT_EVENT       : self.player_shoot, 
            TRAVELLER_BEHIT_EVENT       : self.player_be_hit, 
            CLEAR_BULLET_BOSS_EVENT     : self.clear_bullet_boss, 
            TRAVELLER_NOT_STRONG_EVENT  : self.player_is_not_strong
        }

        self.KEYDOWN_DICT = {
            pygame.K_ESCAPE     : self.pause_switch, 
            pygame.K_LSHIFT     : self.player_slow, 
            pygame.K_LEFT       : self.player_left, 
            pygame.K_RIGHT      : self.player_right, 
            pygame.K_UP         : self.player_up, 
            pygame.K_DOWN       : self.player_down, 
            pygame.K_z          : self.player_is_shooting
        }

        self.KEYUP_DICT = {
            pygame.K_LSHIFT     : self.player_fast, 
            pygame.K_LEFT       : self.player_de_left, 
            pygame.K_RIGHT      : self.player_de_right, 
            pygame.K_UP         : self.player_de_up, 
            pygame.K_DOWN       : self.player_de_down, 
            pygame.K_z          : self.player_isnot_shooting
        }


    def quit(self, event: pygame.event.Event):
        self.state_of_game = GAMEOVER

    def next_stage(self, event: pygame.event.Event):
        self.state_of_game += 1
        print("game state add", self.state_of_game)
        

    # Keydown and keyup event

    def pause_switch(self):
        self.is_pause = not self.is_pause

    def player_slow(self):
        self.player.is_slow = True
        self.player.speed = self.player.ORIGINSPEED * 0.4
        # print("slow")

    def player_fast(self):
        self.player.is_slow = False
        self.player.speed = self.player.ORIGINSPEED
        # print("fast")

    def player_up(self):
        self.up_keydown = True
        self.player.direction_y = -1
        # print("up")

    def player_down(self):
        self.down_keydown = True
        self.player.direction_y = 1
        # print("down")

    def player_left(self):
        self.left_keydown = True
        self.player.direction_x = -1
        # print("left")

    def player_right(self):
        self.right_keydown = True
        self.player.direction_x = 1
        # print("right")

    def player_de_up(self):
        self.up_keydown = False
        if self.down_keydown:
            self.player.direction_y = 1
        else:
            self.player.direction_y = 0

    def player_de_down(self):
        self.down_keydown = False
        if self.up_keydown:
            self.player.direction_y = -1
        else:
            self.player.direction_y = 0

    def player_de_left(self):
        self.left_keydown = False
        if self.right_keydown:
            self.player.direction_x = 1
        else:
            self.player.direction_x = 0

    def player_de_right(self):
        self.right_keydown = False
        if self.left_keydown:
            self.player.direction_x = -1
        else:
            self.player.direction_x = 0

    def player_is_shooting(self):
        print("shoot")
        pygame.time.set_timer(
            TRAVELLER_SHOOT_EVENT, 
            TRAVELLER_SHOOT_TIME
        )
        # print("shoot")

    def player_isnot_shooting(self):
        pygame.time.set_timer(
            TRAVELLER_SHOOT_EVENT, 
            0
        )
        # print("not shoot")



    # Event with no info

    def boss_begin_roundspray(self, event: pygame.event.Event):
        pygame.time.set_timer(BOSS_ROUNDSPRAY_EVENT, BOSS_ROUNDSPRAY_TIME, 3)

    def boss_roundspray(
            self, 
            event: pygame.event.Event, 
            num: int = 24
        ):
        rand = random.random()
        for i in range(num):
            self.bullets_boss.add(
                Bullet_roundspray(
                    7, 
                    self.boss.pos_x, self.boss.pos_y,  
                    4, 
                    i * pi / num * 2 + rand
                )
            )
        for i in range(num):
            self.bullets_boss.add(
                Bullet_roundspray(
                    7, 
                    self.boss.pos_x, self.boss.pos_y, 
                    -4, 
                    i * pi / num * 2 + rand,
                    IMG_BULLET_SUPER
                )
            )

    def player_shoot(self, event: pygame.event.Event):
        self.bullets_player.add(
            Bullet_front(
                12, 
                self.player.pos_x, self.player.rect.top, 
                150, 
                -0.5 * pi, 
                IMG_BUTTERFLY
            )
        )

    def player_be_hit(self, event: pygame.event.Event):
        if not self.player.is_strongest:
            # print("behit!")
            self.behit_position.write(f'({self.player.pos_x}, {self.player.pos_y}), \n')
            pygame.image.save(self.window, f'AI_data/{pygame.time.get_ticks()}_{self.player.health}.png')
            self.scene_updater.begin_behit_time = pygame.time.get_ticks()
            self.player.health -= 1
            self.player.is_strongest = True
            self.player.image.set_alpha(100)
            pygame.time.set_timer(CLEAR_BULLET_BOSS_EVENT, CLEAR_BULLET_BOSS_TIME, 1)
            pygame.time.set_timer(TRAVELLER_NOT_STRONG_EVENT, TRAVELLER_NOT_STRONG_TIME, 1)
            if self.player.health <= 0:
                # print("died")
                pygame.time.set_timer(TRAVELLER_DIED_EVENT, TRAVELLER_DIED_TIME, 1)


    def clear_bullet_boss(self, event: pygame.event.Event):
        self.bullets_boss.empty()

    def player_is_not_strong(self, event: pygame.event.Event):
        # print("lose strong!")
        self.player.is_strongest = False
        self.player.image.set_alpha(256)

    def keydown_handle(self, event: pygame.event.Event):
        handler = self.KEYDOWN_DICT.get(event.key)
        if handler:
            handler()

    def keyup_handle(self, event: pygame.event.Event):
        handler = self.KEYUP_DICT.get(event.key)
        if handler:
            handler()        


    def event_handle(self):

        for event in pygame.event.get():
            handler = self.EVENT_DICT.get(event.type)
            if handler:
                handler(event)
