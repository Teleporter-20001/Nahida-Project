import os
import random
import sys
import pygame
from math import sin, cos, pi
import numpy as np
import heapq

from app.characters.AimatPlayerBullet import AimatPlayerBullet
from app.characters.Nahida import Nahida
from app.characters.Boss import Boss
from app.characters.OurBullet import OurBullet
from app.characters.SprayBullet import SprayBullet
from app.characters.StraightEnemyBullet import StraightEnemyBullet
from app.common.Settings import Settings
from app.Agent.DataStructure import State, Action
from app.env.RewardSet import RewardSet


class TouhouEnv:
    """A minimal RL-style environment wrapper around the existing game objects.

    Implements the basic RL interface: reset, step, render, close.
    Observation is provided via `State` instances. Actions are the `Action` enum.
    """

    PLAYERDIR_TO_ACTION_DICT = {
        (-1, -1): Action.LEFTUP,
        (0, -1): Action.UP,
        (1, -1): Action.RIGHTUP,
        (-1, 0): Action.LEFT,
        (0, 0): Action.NOMOVE,
        (1, 0): Action.RIGHT,
        (-1, 1): Action.LEFTDOWN,
        (0, 1): Action.DOWN,
        (1, 1): Action.RIGHTDOWN,
    }

    PLAYER_SHOOT_EVENT = pygame.USEREVENT + 1
    BOSS_CHANGE_DIR_EVENT = pygame.USEREVENT + 2
    BOSS_SPRAY_EVENT = pygame.USEREVENT + 3

    PLAYER_SHOOT_PERIOD = int(Settings.FPS / 8)    # player shoot once every ? frames
    BOSS_CHANGE_DIR_PERIOD = int(Settings.FPS * 2.5)
    BOSS_SPRAY_PERIOD = int(Settings.FPS * 4)
    BOSS_AIMAT_PLAYER_SHOOT_PERIOD = int(Settings.FPS * 4)
    BOSS_RAIN_PERIOD = int(Settings.FPS * 3)
    
    def __init__(self, settings=Settings):

        pygame.init()
        self.settings = settings
        self.clock = pygame.time.Clock()
        # request window position (SDL reads this env var when creating the window)
        # os.environ['SDL_VIDEO_WINDOW_POS'] = "1600,800"
        self.screen = pygame.display.set_mode((settings.window_width, settings.window_height))
        # optional: remove env var so it doesn't affect later windows
        # os.environ.pop('SDL_VIDEO_WINDOW_POS', None)
        pygame.display.set_caption('TouhouEnv')

        # font for FPS display (initialized once)
        try:
            self.font = pygame.font.SysFont(None, 24)
        except Exception:
            self.font = None

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
        # self.observation_space = {
        #     'player_pos': (settings.window_width, settings.window_height),
        # }

        self._terminated = False
        self.worldAge: int = 0

        # for human teaching
        self.human_action: Action = Action.NOMOVE
        self.isleft = False # for key state
        self.isright = False
        self.isup = False
        self.isdown = False
        self.playerdirx = 0 # for player direction
        self.playerdiry = 0

    def _get_observation(self) -> State:
        
        # Find nearest k enemy bullets to the player and return their features
        k = 10
        px = self.player.posx
        py = self.player.posy

        bullets = []  # list of (dist_sq, x, y, vx, vy, radius)
        for b in self.enemy_bullets:
            try:
                bx = getattr(b, 'posx', b.rect.centerx)
                by = getattr(b, 'posy', b.rect.centery)
                bvx = getattr(b, 'vx', 0.0)
                bvy = getattr(b, 'vy', 0.0)
                br = getattr(b, 'radius', max(getattr(b, 'rect', pygame.Rect(0,0,1,1)).width, getattr(b, 'rect', pygame.Rect(0,0,1,1)).height) / 2.0)
            except Exception as e:
                print(f'Warning: Error getting bullet attributes: {e}')
                continue
            dx = bx - px
            dy = by - py
            dist_sq = dx * dx + dy * dy
            bullets.append((dist_sq, float(dx), float(dy), float(bvx), float(bvy), float(br)))

        # take k nearest by distance using a heap for O(n log k) performance
        if len(bullets) <= k:
            nearest = bullets
        else:
            nearest = heapq.nsmallest(k, bullets, key=lambda x: x[0])

        # build numpy array (k,5) with rows [x,y,vx,vy,radius], pad with zeros if fewer than k
        arr = np.zeros((k, 5), dtype=np.float32)
        for i, item in enumerate(nearest):
            _, dx, dy, bvx, bvy, br = item
            arr[i, :] = (dx, dy, bvx, bvy, br)

        # get human advice:
        self.human_action = self.PLAYERDIR_TO_ACTION_DICT.get((self.playerdirx, self.playerdiry), Action.RIGHTUP)

        obs = {
            'player_x': self.player.posx, 
            'player_y': self.player.posy,
            'boss_from_player_x': self.boss.posx - self.player.posx,
            'boss_from_player_y': self.boss.posy - self.player.posy,
            'boss_velx': self.boss.vx,
            'boss_vely': self.boss.vy,
            'nearest_bullets': arr, # shape (k,5) array of nearest enemy bullets, each row is (x, y(relative to player), vx, vy, radius)
            'human_action': self.human_action,
        }
        return State(observation=obs)

    def reset(self) -> State:
        """Reset environment to initial state and return initial observation (State)."""
        # reset player position to bottom center
        self.player._posx = int(self.settings.window_width * .5)
        self.player._posy = int(self.settings.window_height * .75)

        # self.boss._posx = int(self.settings.window_width / 4)
        self.boss._posx = int(random.randint(self.boss.rect.width/2 + 5, self.settings.window_width - self.boss.rect.width/2 - 5))
        self.boss._posy = int(self.settings.window_height / 5)
        self.boss.health = self.boss.max_health

        self.our_bullets.empty()
        self.enemy_bullets.empty()

        self._terminated = False
        # restart automatic timer
        # self.stop_shoot_timer()
        # self.stop_boss_change_dir_timer()
        # self.stop_boss_spray_timer()
        # self.start_shoot_timer(130)
        # self.start_boss_change_dir_timer(2000)
        # self.start_boss_spray_timer(900)
        self.worldAge = 0

        self.human_action: Action = Action.NOMOVE

        return self._get_observation()


    def step(self, action: Action):
        """Apply action (an Action enum) and step environment.

        Returns: (next_state: State, reward: float, done: bool, info: dict)
        """
        reward: RewardSet = RewardSet()

        if self._terminated:
            return self._get_observation(), 0.0, True, {}

        # -------------------- handle events -------------------
        _offset = int(Settings.FPS / 4)
        if self.worldAge % self.PLAYER_SHOOT_PERIOD == 0:
            self._player_shoot()
        if self.worldAge % self.BOSS_CHANGE_DIR_PERIOD == 0:
            self._boss_change_direction()
        if (self.worldAge + _offset) % self.BOSS_SPRAY_PERIOD == 0:
            self._boss_spray()
        if (self.worldAge + _offset) % self.BOSS_AIMAT_PLAYER_SHOOT_PERIOD == 0:
            self._boss_aimat_player_shoot()
        if self.worldAge % self.BOSS_RAIN_PERIOD == 0:
            self._boss_rain()

        # apply movement
        if isinstance(action, Action):
            self.player.set_action(action)

        # update groups
        self.players.update()
        self.our_bullets.update()
        self.enemies.update()
        self.enemy_bullets.update()

        # placeholder reward: small positive reward to encourage alive
        reward.survive()
        done = False
        
        # punish player going to edges
        if self.player.posx < self.player.minx + self.settings.BORDER_BUFFER:
            reward.edge_punish((1 - (self.player.posx - self.player.minx) / self.settings.BORDER_BUFFER) * self.settings.BORDER_PUNISH)
        if self.player.posx > self.player.maxx - self.settings.BORDER_BUFFER:
            reward.edge_punish((1 - (self.player.maxx - self.player.posx) / self.settings.BORDER_BUFFER) * self.settings.BORDER_PUNISH)
        if self.player.posy < self.player.miny + self.settings.BORDER_BUFFER:
            reward.edge_punish((1 - (self.player.posy - self.player.miny) / self.settings.BORDER_BUFFER) * self.settings.BORDER_PUNISH)
        if self.player.posy > self.player.maxy - self.settings.BORDER_BUFFER:
            reward.edge_punish((1 - (self.player.maxy - self.player.posy) / self.settings.BORDER_BUFFER) * self.settings.BORDER_PUNISH)

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
                                reward.hit_enemy()
                                # if getattr(enemy, 'health', 0) <= 0:
                                    # enemy.kill()
                                    # reward += 10.0
                        except Exception:
                            pass
                        # bullet hit an enemy; move to next bullet
                        break

                # punish our bullets going out of screen, not shooting boss
                if bullet.posy < bullet.miny - bullet.rect.height/2:
                    try:
                        # reward += -0.1
                        bullet.kill()
                    except Exception as e:
                        print(f'an exception occured during clearing out-screen bullet: {e}')

        except Exception as e:
            print(f'Error in bullet-vs-enemy collision: {e}')
            
        if not self.boss.alive:
            reward.killenemy()
            self._terminated = True
            done = True

        # player vs enemy bullets collision: circle-vs-circle using radius fields
        min_dist = np.infty
        try:
            px = self.player.posx
            py = self.player.posy
            for bullet in self.enemy_bullets:
                if done:
                    break
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
                dist = np.sqrt(dist_sq)
                if dist < min_dist:
                    min_dist = dist
                # encourage to avoid bullets closely: $ [px - bx, py - by] \dot [vx, vy] < 0$
                if dist < self.player.radius + br + 40 and np.dot([dx, dy], [bullet.vx, bullet.vy]) < 0:
                    reward.avoid()
                # collision if distance between centers <= (player radius + bullet radius)
                if dist_sq <= (self.player.radius + br) ** 2:
                    reward.behit()
                    done = True
            # reward += -3. / min_dist
        except Exception as e:
            # defensive: if sprites/groups aren't set up as expected, don't crash the env
            print(f'Error in collision detection: {e}')

        info = {
            'reward_details': reward
        }

        self.worldAge += 1

        return self._get_observation(), reward.value, done, info

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
        # draw boss health bar at the top (left-aligned) after sprites so the empty area is transparent
        try:
            boss = self.boss
            current_hp = float(getattr(boss, 'health', 0))
            # use explicit max_health field as requested
            max_hp = float(getattr(boss, 'max_health', 1.0))
            if max_hp <= 0:
                max_hp = 1.0
            hp_ratio = max(0.0, min(1.0, current_hp / max_hp))

            bar_height = 18
            padding = 4
            bar_y = padding
            bar_x = 0
            bar_full_width = self.settings.window_width
            # filled portion only; empty area is left transparent to show game beneath
            filled_width = int(bar_full_width * hp_ratio)
            if filled_width > 0:
                pygame.draw.rect(self.screen, (200, 30, 30), (bar_x, bar_y, filled_width, bar_height))
            # border around full width
            pygame.draw.rect(self.screen, (0, 0, 0), (bar_x, bar_y, bar_full_width, bar_height), 2)
        except Exception:
            pass

        # draw FPS at bottom-left for debugging/performance monitoring
        try:
            font = getattr(self, 'font', None)
            if font is not None:
                fps = self.clock.get_fps()
                fps_surf = font.render(f'FPS: {fps:.1f}', True, (255, 255, 255))
                fps_x = 6
                fps_y = self.settings.window_height - fps_surf.get_height() - 60
                # draw a small dark background rectangle for readability
                pygame.draw.rect(self.screen, (0, 0, 0), (fps_x - 4, fps_y - 2, fps_surf.get_width() + 8, fps_surf.get_height() + 4))
                self.screen.blit(fps_surf, (fps_x, fps_y))
        except Exception:
            pass
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
        # ------------- events based on timer may not be used anymore -------------
        elif event.type == pygame.KEYDOWN:
            self._handle_keydown(event)
        elif event.type == pygame.KEYUP:
            self._handle_keyup(event)
            
            
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
            -1800,
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
        r = random.randint(6, 12)
        bullet_num = random.randint(3, 5)
        for theta in np.linspace(0, 2 * np.pi, num=bullet_num, endpoint=False):
            self.enemy_bullets.add(SprayBullet( # type: ignore
                self.boss.posx,
                self.boss.posy,
                r,
                os.path.join('resources', 'bullet_super.png'),
                float(theta),
                target_size=(2*r, 2*r)
            ))
            self.enemy_bullets.add(SprayBullet( # type: ignore
                self.boss.posx,
                self.boss.posy,
                r,
                os.path.join('resources', 'bullet_super.png'),
                float(theta),
                target_size=(2*r, 2*r),
                inverse=True
            ))

    def _handle_keydown(self, event):
        key = event.key
        if key == pygame.K_UP:
            self.isup = True
            self.playerdiry = -1
        elif key == pygame.K_DOWN:
            self.isdown = True
            self.playerdiry = 1
        elif key == pygame.K_LEFT:
            self.isleft = True
            self.playerdirx = -1
        elif key == pygame.K_RIGHT:
            self.isright = True
            self.playerdirx = 1

    def _handle_keyup(self, event):
        key = event.key
        if key == pygame.K_UP:
            self.isup = False
            if self.isdown:
                self.playerdiry = 1
            else:
                self.playerdiry = 0
        elif key == pygame.K_DOWN:
            self.isdown = False
            if self.isup:
                self.playerdiry = -1
            else:
                self.playerdiry = 0
        elif key == pygame.K_LEFT:
            self.isleft = False
            if self.isright:
                self.playerdirx = 1
            else:
                self.playerdirx = 0
        elif key == pygame.K_RIGHT:
            self.isright = False
            if self.isleft:
                self.playerdirx = -1
            else:
                self.playerdirx = 0

    def _boss_aimat_player_shoot(self):
        self.enemy_bullets.add(AimatPlayerBullet(   # type: ignore
            self.boss.posx,
            self.boss.posy,
            random.randint(6, 14),
            os.path.join('resources', 'bullet_super.png'),
            self.player.posx,
            self.player.posy
        ))

    def _boss_rain(self):
        bullet_num = random.randint(3, 5)
        room_size_x = int(Settings.window_width / bullet_num) - 1
        room_size_y = int(Settings.window_height / bullet_num) - 1
        for _dir in [1, -1]:
            begin_x = random.randint(0, room_size_x)
            begin_y = random.randint(0, room_size_y)
            for i in range(bullet_num):
                self.enemy_bullets.add(StraightEnemyBullet( # type: ignore
                    begin_x + i * room_size_x,
                    Settings.window_height / 2 - Settings.window_height / 2 * 0.95 * _dir,
                    random.randint(8, 12),
                    os.path.join('resources', 'bullet_super.png'),
                    pi / 2 * _dir
                ))
                self.enemy_bullets.add(StraightEnemyBullet( # type: ignore
                    Settings.window_width / 2 - Settings.window_width / 2 * 0.95 * _dir,
                    begin_y + i * room_size_y,
                    random.randint(8, 12),
                    os.path.join('resources', 'bullet_super.png'),
                    pi / 2 * (_dir - 1)
                ))

