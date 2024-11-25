import pygame
from math import * 
from collections import deque
from Bullets import * 
from Characters import * 
from Settings import * 


class Point(object):
    def __init__(self, x, y) -> None:
        self.pos_x = x
        self.pos_y = y


class SpatialHash:

    def __init__(self, cell_size=10):
        self.cell_size = cell_size
        self.hash_table = {}
        self.checked = set()

    def _hash(self, x, y):
        try:
            return int(x / self.cell_size), int(y / self.cell_size)
        except Exception as e:
            print(f"Error occurred while hashing: {e}")
            return None

    def insert(self, obj):
        try:
            # 计算对象的边界框覆盖的所有格子
            left_top = self._hash(obj.pos_x - obj.radius, obj.pos_y - obj.radius)
            right_bottom = self._hash(obj.pos_x + obj.radius, obj.pos_y + obj.radius)
            for x in range(left_top[0], right_bottom[0] + 1):
                for y in range(left_top[1], right_bottom[1] + 1):
                    cell = (x, y)
                    if cell not in self.hash_table:
                        self.hash_table[cell] = []
                    self.hash_table[cell].append(obj)
        except Exception as e:
            print(f"Error occurred while inserting object: {e}")

    def remove(self, obj):
        try:
            # 从对象的边界框覆盖的所有格子中删除对象
            left_top = self._hash(obj.pos_x - obj.radius, obj.pos_y - obj.radius)
            right_bottom = self._hash(obj.pos_x + obj.radius, obj.pos_y + obj.radius)
            for x in range(left_top[0], right_bottom[0] + 1):
                for y in range(left_top[1], right_bottom[1] + 1):
                    cell = (x, y)
                    if cell in self.hash_table:
                        self.hash_table[cell].remove(obj)
            self.checked.remove(obj)
        except Exception as e:
            print(f"Error occurred while removing object: {e}")

    def retrieve(self, x, y):
        try:
            cell = self._hash(x, y)
            if cell in self.hash_table:
                # 从结果中排除已经检测过的对象
                return [obj for obj in self.hash_table[cell] if obj not in self.checked]
            return []
        except Exception as e:
            print(f"Error occurred while retrieving objects: {e}")
            return []
        
    def clear_checked(self):
        self.checked.clear()

    def clear(self):
        try:
            # 清空所有的元素
            self.hash_table.clear()
            self.checked.clear()
        except Exception as e:
            print(f"Error occurred while clearing SpatialHash: {e}")



class Collision_manager(object):

    def __init__(
        self, 
        window: pygame.Surface, 
        player: Traveller, 
        bullets_player: pygame.sprite.Group, 
        boss: Boss, 
        bullets_boss: pygame.sprite.Group
    ) -> None:
        
        self.window = window

        self.player = player
        self.bullets_player = bullets_player
        self.boss = boss
        self.bullets_boss = bullets_boss

        self.player_bossbullet_hash: SpatialHash = SpatialHash(50)
        self.boss_playerbullet_hash: SpatialHash = SpatialHash(50)

        self.is_shooting = False
        self.sumFx = 0
        self.sumFy = 0
        self.K_BULLET = 9
        self.K_WALL = 3.8e-8
        self.CENTER = Point(WINDOW_SIZE[0] // 2, int(WINDOW_SIZE[1] * 0.7))
        self.fx_i = []
        self.fy_i = []
        self.fx_wall = 0
        self.fy_wall = 0
        self.history_sumFx = deque(maxlen = 4)
        self.history_sumFy = deque(maxlen = 4)



    def get_player_cells(self):
        # 计算对象的边界框覆盖的所有格子
        player_cells: tuple[int, int] = []
        left_top = self.player_bossbullet_hash._hash(self.player.pos_x - self.player.radius, self.player.pos_y - self.player.radius)
        right_bottom = self.player_bossbullet_hash._hash(self.player.pos_x + self.player.radius, self.player.pos_y + self.player.radius)
        for x in range(left_top[0], right_bottom[0] + 1):
            for y in range(left_top[1], right_bottom[1] + 1):
                cell = (x, y)
                if cell not in self.player_bossbullet_hash.hash_table:
                    self.player_bossbullet_hash.hash_table[cell] = []
                player_cells.append(cell)
        return player_cells


    def get_more_player_cells(self):
        # 计算对象的边界框覆盖的所有格子
        player_cells: tuple[int, int] = []
        left_top = self.player_bossbullet_hash._hash(self.player.pos_x - self.player.radius, self.player.pos_y - self.player.radius)
        right_bottom = self.player_bossbullet_hash._hash(self.player.pos_x + self.player.radius, self.player.pos_y + self.player.radius)
        for x in range(left_top[0] - 2, right_bottom[0] + 2 + 1):
            for y in range(left_top[1] - 2, right_bottom[1] + 2 + 1):
                cell = (x, y)
                if cell not in self.player_bossbullet_hash.hash_table:
                    self.player_bossbullet_hash.hash_table[cell] = []
                player_cells.append(cell)
        return player_cells


    def player_bossbullet_collision_handle(self):

        self.player_bossbullet_hash.clear()

        self.player_bossbullet_hash.insert(self.player)
        for bullet in self.bullets_boss:
            self.player_bossbullet_hash.insert(bullet)

        player_cells = self.get_player_cells()
        # print(player_cells)

        # to_check_bullets = []
        # player_cells: tuple[int, int] = []
        # left_top = self.player_bossbullet_hash._hash(self.player.pos_x - self.player.radius, self.player.pos_y - self.player.radius)
        # right_bottom = self.player_bossbullet_hash._hash(self.player.pos_x + self.player.radius, self.player.pos_y + self.player.radius)
        # for x in range(left_top[0], right_bottom[0] + 1):
        #     for y in range(left_top[1], right_bottom[1] + 1):
        #         to_check_bullets += self.player_bossbullet_hash.retrieve(x, y)
                
        break_flag = False
        for cell in player_cells:
            for bullet in self.player_bossbullet_hash.hash_table[cell]:
                if bullet != self.player and pygame.sprite.collide_circle(self.player, bullet):
                    # print(f"{len(self.bullets_boss)}, nahida is hit at {self.player.pos_x, self.player.pos_y, bullet.pos_x, bullet.pos_y}")
                    pygame.event.post(pygame.event.Event(TRAVELLER_BEHIT_EVENT))
                    break_flag = True
                if break_flag:
                    break
            if break_flag:
                break 
                
        # print(len(to_check_bullets))
        # for bullet in to_check_bullets:
        #     if bullet != self.player and pygame.sprite.collide_circle(self.player, bullet):
        #         print(f"{len(self.bullets_boss)}, nahida is hit at {self.player.pos_x, self.player.pos_y, bullet.pos_x, bullet.pos_y}")
        #         pygame.event.post(pygame.event.Event(TRAVELLER_BEHIT_EVENT))
        #         break


    def AI_calc_direction(self):

        if not self.is_shooting:
            pygame.event.post(
                pygame.event.Event(pygame.K_z)
            )
            self.is_shooting = True

        def calc_dest(obj1, obj2):
            return sqrt(
                (obj1.pos_x - obj2.pos_x) ** 2 + 
                (obj1.pos_y - obj2.pos_y) ** 2
            )
        
        def calc_theta(obj1, obj2):
            return atan2(
                obj2.pos_y - obj1.pos_y, 
                obj2.pos_x - obj1.pos_x
            )
        
        self.sumFx = 0
        self.sumFy = 0
        self.fx_wall = 0
        self.fy_wall = 0
        self.fx_i = []
        self.fy_i = []

        # Forces that bullets give
        player_cells = self.get_more_player_cells()
        for cell in player_cells:
            for bullet in self.player_bossbullet_hash.hash_table[cell]:
                if bullet is not self.player and bullet is not None:
                    f_i = self.K_BULLET / calc_dest(self.player, bullet) ** 2
                    theta_i = calc_theta(self.player, bullet)
                    self.sumFx += -f_i * cos(theta_i)
                    self.sumFy += -f_i * sin(theta_i)
                    self.fx_i.append(-f_i * cos(theta_i))
                    self.fy_i.append(-f_i * sin(theta_i))

        # Forces that the walls give
        # self.sumFy -= self.K_WALLY / abs(WINDOW_SIZE[1] - self.player.pos_y)
        # self.sumFy += self.K_WALLY / (self.player.pos_y * 0.6 )
        # self.sumFx -= self.K_WALLX / abs(WINDOW_SIZE[0] - self.player.pos_x)
        # self.sumFx += self.K_WALLX / self.player.pos_x 
        f_i = self.K_WALL * (len(self.fx_i) + 2) * calc_dest(self.player, self.CENTER) ** 2.1
        theta_i = calc_theta(self.player, self.CENTER)
        # print(f_i)
        if calc_dest(self.player, self.CENTER) > 32:
            self.sumFx += f_i * cos(theta_i)
            self.sumFy += f_i * sin(theta_i)
            self.fx_wall = f_i * cos(theta_i)
            self.fy_wall = f_i * sin(theta_i)
        
        sumF = sqrt(self.sumFx ** 2 + self.sumFy ** 2)
        # print(len(self.fx_i), sumF)
        if sumF:
            self.sumFx /= sumF
            self.sumFy /= sumF
        else:
            self.sumFx = 0
            self.sumFy = 0
        self.history_sumFx.append(self.sumFx)
        self.history_sumFy.append(self.sumFy)
 
        resx = 0
        resy = 0
        for x in self.history_sumFx:
            resx += x
        for y in self.history_sumFy:
            resy += y
        resF = sqrt(resx ** 2 + resy ** 2)
        resF *= 1.5
        # print(len(self.fx_i), sumF)
        if resF:
            resx /= resF
            resy /= resF
        else:
            resx = 0
            resy = 0

        self.player.direction_x = resx
        self.player.direction_y = resy

                
    def collision_handle(self):
        self.player_bossbullet_collision_handle()
        self.AI_calc_direction()

