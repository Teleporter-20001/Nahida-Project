import os
import sys
import pygame
from math import sin, cos, pi
import numpy as np

from app.characters.Nahida import Nahida
from app.characters.Boss import Boss
from app.characters.OurBullet import OurBullet
from app.characters.SprayBullet import SprayBullet
from app.common.Settings import Settings
from app.Agent.DataStructure import State, Action


class TouhouEnv:
    """A minimal RL-style environment wrapper around the existing game objects.

    Implements the basic RL interface: reset, step, render, close.
    Observation is provided via `State` instances. Actions are the `Action` enum.
    """

    PLAYER_SHOOT_EVENT = pygame.USEREVENT + 1
    BOSS_CHANGE_DIR_EVENT = pygame.USEREVENT + 2
    BOSS_SPRAY_EVENT = pygame.USEREVENT + 3
    
    def __init__(self, settings=Settings):
        pygame.init()
        self.settings = settings
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((settings.window_width, settings.window_height))
        pygame.display.set_caption('TouhouEnv')

        # sprite group holding the player (environment owns entities)
        self.players = pygame.sprite.Group()
        self.player = Nahida(
            int(settings.window_width * .5),
            int(settings.window_height * .75),
            8,
            os.path.join('resources', 'nahida_2.png'),
            target_size=(80, 130)
        )
        self.players.add(self.player)   # type: ignore
        # sprite group holding player's bullets
        self.our_bullets: pygame.sprite.Group = pygame.sprite.Group()
        self.enemies = pygame.sprite.Group()
        self.boss = Boss(
            self.settings.window_width / 2,
            self.settings.window_height / 5,
            50,
            os.path.join('resources', 'boss.png'),
            health=250, 
            target_size=(100, 100)
        )
        self.enemies.add(self.boss) # type: ignore
        self.enemy_bullets = pygame.sprite.Group()

        # RL-style metadata
        self.action_space = list(Action)
        # observation_space is informal here (no gym dependency)
        self.observation_space = {
            'player_pos': (settings.window_width, settings.window_height),
        }

        self._terminated = False
        # custom pygame event for timed player shooting

    def _get_observation(self) -> State:
        obs = {
            'player_x': self.player.rect.centerx,
            'player_y': self.player.rect.centery,
            'player_rect': self.player.rect.copy(),
            'screen_width': self.settings.window_width,
            'screen_height': self.settings.window_height,
        }
        return State(observation=obs)

    def reset(self) -> State:
        """Reset environment to initial state and return initial observation (State)."""
        # reset player position to bottom center
        self.player.rect.centerx = int(self.settings.window_width * .5)
        self.player.rect.centery = int(self.settings.window_height * .75)
        self._terminated = False
        # start automatic shooting timer (300 ms)
        self.start_shoot_timer(130)
        self.start_boss_change_dir_timer(2000)
        self.start_boss_spray_timer(700)
        return self._get_observation()


    def step(self, action: Action):
        """Apply action (an Action enum) and step environment.

        Returns: (next_state: State, reward: float, done: bool, info: dict)
        """
        if self._terminated:
            return self._get_observation(), 0.0, True, {}

        # apply movement
        if isinstance(action, Action):
            self.player.set_action(action)

        # update groups
        self.players.update()
        self.our_bullets.update()
        self.enemies.update()
        self.enemy_bullets.update()

        # placeholder reward: small negative step penalty to encourage efficiency
        reward = 0.01
        done = False

        # player bullets vs enemies collision: circle-vs-circle using radius fields
        # collision detection: use a simple circle (player center, radius from BaseChar) vs bullet centers
        try:
            # check our bullets against enemies
            for bullet in list(self.our_bullets):
                try:
                    bx = bullet.rect.centerx
                    by = bullet.rect.centery
                except Exception:
                    continue
                br = getattr(bullet, 'radius', max(bullet.rect.width, bullet.rect.height) / 2.0)
                for enemy in list(self.enemies):
                    try:
                        ex = enemy.rect.centerx
                        ey = enemy.rect.centery
                    except Exception:
                        continue
                    er = getattr(enemy, 'radius', getattr(enemy, '_radius', 0))
                    dx = bx - ex
                    dy = by - ey
                    if dx * dx + dy * dy <= (br + er) ** 2:
                        try:
                            bullet.kill()
                        except Exception:
                            pass
                        try:
                            if hasattr(enemy, 'health'):
                                enemy.health -= 1
                                reward += 0.1
                                if getattr(enemy, 'health', 0) <= 0:
                                    enemy.kill()
                                    reward += 10.0
                        except Exception:
                            pass
                        # bullet hit an enemy; move to next bullet
                        break
        except Exception as e:
            print(f'Error in bullet-vs-enemy collision: {e}')
            
        if not self.boss.alive:
            reward += 1000.0
            self._terminated = True
            done = True
        
        try:
            px = self.player.posx
            py = self.player.posy
            for bullet in self.enemy_bullets:
                # try to get bullet center; fallback to rect if attribute missing
                try:
                    bx = bullet.rect.centerx
                    by = bullet.rect.centery
                except Exception:
                    # if bullet has no rect, skip it
                    continue

                # use bullet.radius property when available (BaseEnemyBullet provides it)
                br = getattr(bullet, 'radius', max(bullet.rect.width, bullet.rect.height) / 2.0)

                dx = px - bx
                dy = py - by
                dist_sq = dx * dx + dy * dy
                # collision if distance between centers <= (player radius + bullet radius)
                if dist_sq <= (self.player.radius + br) ** 2:
                    reward += -1000
                    done = True
                    break
        except Exception as e:
            # defensive: if sprites/groups aren't set up as expected, don't crash the env
            print(f'Error in collision detection: {e}')

        info = {}

        return self._get_observation(), reward, done, info

    def render(self):
        """Render one frame to the screen."""
        self.screen.fill(self.settings.window_background_color)
        self.players.draw(self.screen)
        # draw a small gray circle at the player's center for hit-area visualization
        try:
            pygame.draw.circle(self.screen, (128, 128, 128), (self.player.rect.centerx, self.player.rect.centery), self.player.radius)
        except Exception:
            pass
        self.our_bullets.draw(self.screen)
        self.enemies.draw(self.screen)
        self.enemy_bullets.draw(self.screen)
        pygame.display.update()
        self.clock.tick(self.settings.FPS)

    def close(self):
        # stop timers before quitting
        try:
            self.stop_shoot_timer()
            self.stop_boss_change_dir_timer()
            self.stop_boss_spray_timer()
        except Exception as e:
            print(f'Error stopping timers: {e}')
        pygame.quit()
        # do not call sys.exit() here; let caller decide

    # -----------------------------------

    def handle_event(self, event):
        """Handle pygame events delegated from the main loop. If a shoot event is received,
        spawn a bullet via _playerShoot.
        """
        if event.type == TouhouEnv.PLAYER_SHOOT_EVENT:
            self._player_shoot()
        elif event.type == TouhouEnv.BOSS_CHANGE_DIR_EVENT:
            self._boss_change_direction()
        elif event.type == TouhouEnv.BOSS_SPRAY_EVENT:
            self._boss_spray()
            
            
    def start_shoot_timer(self, interval_ms: int = 300):
        """Start a pygame timer that will generate events every interval_ms milliseconds."""
        pygame.time.set_timer(TouhouEnv.PLAYER_SHOOT_EVENT, int(interval_ms))

    def stop_shoot_timer(self):
        """Stop the shooting timer."""
        pygame.time.set_timer(TouhouEnv.PLAYER_SHOOT_EVENT, 0)

    def _player_shoot(self):
        self.our_bullets.add(OurBullet( # type: ignore
            self.player.posx,
            self.player.posy,
            10,
            os.path.join('resources', 'butterfly.png'),
            0,
            -1000,
            (20, 20)
        ))
        
    def start_boss_change_dir_timer(self, interval_ms: int = 1000):
        """Start a pygame timer that will generate events every interval_ms milliseconds."""
        pygame.time.set_timer(TouhouEnv.BOSS_CHANGE_DIR_EVENT, int(interval_ms))

    def stop_boss_change_dir_timer(self):
        """Stop the boss change direction timer."""
        pygame.time.set_timer(TouhouEnv.BOSS_CHANGE_DIR_EVENT, 0)
        
    def _boss_change_direction(self):
        self.boss.change_direction()
        
    def start_boss_spray_timer(self, interval_ms: int = 1000):
        """Start a pygame timer that will generate events every interval_ms milliseconds."""
        pygame.time.set_timer(TouhouEnv.BOSS_SPRAY_EVENT, int(interval_ms))

    def stop_boss_spray_timer(self):
        """Stop the boss spraying timer."""
        pygame.time.set_timer(TouhouEnv.BOSS_SPRAY_EVENT, 0)

    def _boss_spray(self):
        for theta in np.linspace(0, 2 * np.pi, num=24, endpoint=False):
            self.enemy_bullets.add(SprayBullet( # type: ignore
                self.boss.posx,
                self.boss.posy,
                15,
                os.path.join('resources', 'bullet_super.png'),
                float(theta),
                target_size=(30, 30)
            ))