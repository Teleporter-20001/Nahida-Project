# This project is licensed under the MIT License (Non-Commercial Use Only).
# Please see the license file in this source distribution for full license terms.

import pygame

VERSION = "Nahida project 6.0 by zhz"
BACKGROUND_COLOR = (255, 140, 0)

# Key config
UPARROW = 1073741906
DOWNARROW = 1073741905
LEFTARROW = 1073741904
RIGHTARROW = 1073741903
SHIFT = 1073742049          # left shift
KEY_SPACE = 32
KEY_Z = 122
KEY_X = 120
KEY_C = 99
KEY_V = 118

# Event config
TRAVELLER_SHOOT_EVENT = pygame.USEREVENT + 3
TRAVELLER_SHOOT_TIME = 100
TRAVELLER_BEHIT_EVENT = pygame.USEREVENT + 4
TRAVELLER_NOT_STRONG_EVENT = pygame.USEREVENT + 5
TRAVELLER_NOT_STRONG_TIME = 4000
CLEAR_BULLET_BOSS_EVENT = pygame.USEREVENT + 6
CLEAR_BULLET_BOSS_TIME = 600
TRAVELLER_DIED_EVENT = pygame.USEREVENT + 7
TRAVELLER_DIED_TIME = 500
BOSS_BEGIN_ROUNDSPRAY_EVENT = pygame.USEREVENT + 1
BOSS_BEGIN_ROUNDSPRAY_TIME = 4000
BOSS_ROUNDSPRAY_EVENT = pygame.USEREVENT + 2
BOSS_ROUNDSPRAY_TIME = 1000

FPS = 60

# Image config
IMG_TRAVELLER           = "resources/nahida_2.png"
IMG_BULLET_ROUND        = "resources/bullet_round.png"
IMG_BULLET_SUPER        = "resources/bullet_super.png"
IMG_BOSS                = "resources/boss.png"
IMG_BUTTERFLY           = "resources/butterfly.png"
IMG_BUTTERFLY_RED       = "resources/butterfly_red.png"


TEAM_DICT = {
    'player'    : 0, 
    'monster'   : 1
}

# Game state config
START_MENU      = 0
GAME_1          = 1
GAME_2          = 2
GAMEOVER        = -1

ORIGINSPEED = 10

WINDOW_SIZE = (600, 800)