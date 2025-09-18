# This project is licensed under the MIT License (Non-Commercial Use Only).
# Please see the license file in this source distribution for full license terms.

# imports
import pygame
import sys
from Settings import *
from Bullets import *
from Characters import *
from Collision_manager import * 

class Scene_updater(object):

    def __init__(
        self, 
        window: pygame.Surface, 
        player: Traveller, 
        players: pygame.sprite.Group, 
        bullets_player: pygame.sprite.Group, 
        boss: Boss, 
        bosses: pygame.sprite.Group, 
        bullets_boss: pygame.sprite.Group, 
        collision_manager: Collision_manager
    ) -> None:
        
        self.window = window
        self.fps_clock = pygame.time.Clock()
        self.font = pygame.font.Font("app/resources/Alibaba_R.ttf", 25)

        self.player = player
        self.players = players
        self.bullets_player = bullets_player
        self.boss = boss
        self.bosses = bosses
        self.bullets_boss = bullets_boss

        self.collision_manager = collision_manager

        self.begin_behit_time = 0

        self.score = 0

    def draw_judge_point(self):
        if self.player.is_slow:
            pygame.draw.circle(self.window, "white", self.player.rect.center, self.player.radius), 
            pygame.draw.circle(self.window, "black", self.player.rect.center, self.player.radius + 1, 1), 
            pygame.draw.circle(self.window, "grey", self.player.rect.center, self.player.radius // 2)

    def draw_score(self):
        self.score_display = self.font.render("score: " + str(self.score), True, "black", None)
        self.window.blit(self.score_display, (10, 0))

    def draw_player_health(self):
        self.health_display = self.font.render("health: " + str(self.player.health), True, "black", None)
        self.window.blit(self.health_display, (10, WINDOW_SIZE[1] - 50))

    def draw_force(self):
        for x, y in zip(self.collision_manager.fx_i, self.collision_manager.fy_i):
            pygame.draw.line(
                self.window, 
                'blue', 
                (self.player.pos_x, self.player.pos_y), 
                (self.player.pos_x + 50000 * x, self.player.pos_y + 50000 * y), 
                6
            )
        pygame.draw.line(
            self.window, 
            'red', 
            (self.player.pos_x, self.player.pos_y), 
            (self.player.pos_x + 50000 * self.collision_manager.fx_wall, self.player.pos_y + 50000 * self.collision_manager.fy_wall), 
            6
        )

    def play_player_behit_animation(self):
        delta_t = pygame.time.get_ticks() - self.begin_behit_time
        self.uplayer = pygame.Surface((WINDOW_SIZE)).convert_alpha()
        self.uplayer.fill('blue')
        if 0 <= delta_t < 300:
            pygame.draw.circle(
                self.uplayer, 
                pygame.Color('red'), 
                (self.player.pos_x, self.player.pos_y), 
                int(delta_t / 2)
            )
        elif 300<= delta_t < 900:
            pygame.draw.circle(
                self.uplayer, 
                pygame.Color('red'), 
                (self.player.pos_x, self.player.pos_y), 
                150
            )
            pygame.draw.circle(
                self.uplayer, 
                pygame.Color('blue'), 
                (self.player.pos_x, self.player.pos_y), 
                int((delta_t - 300) * 0.375)
            )
        self.uplayer.set_colorkey(pygame.Color('blue'))
        self.uplayer.set_alpha(164)
        self.window.blit(self.uplayer, (0, 0))


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

        self.draw_judge_point()

        if self.player.is_strongest:
            self.play_player_behit_animation()

        self.draw_score()
        self.draw_player_health()
        self.draw_force()

        pygame.display.update()
        self.fps_clock.tick(FPS)
