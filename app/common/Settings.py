from dataclasses import dataclass

import pygame


class Settings:
    FPS: int = 60
    window_width: int = 600
    window_height: int = 800
    window_background_color: pygame.color.Color = pygame.Color(80, 160, 150)