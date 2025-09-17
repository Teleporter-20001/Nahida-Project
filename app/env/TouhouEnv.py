import os
import sys
import pygame

from app.characters.Nahida import Nahida
from app.common.Settings import Settings
from app.Agent.DataStructure import State, Action


class TouhouEnv:
    """A minimal RL-style environment wrapper around the existing game objects.

    Implements the basic RL interface: reset, step, render, close.
    Observation is provided via `State` instances. Actions are the `Action` enum.
    """

    def __init__(self, settings=Settings):
        pygame.init()
        self.settings = settings
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((settings.window_width, settings.window_height))
        pygame.display.set_caption('TouhouEnv')

        # sprite group holding the player (environment owns entities)
        self.lives = pygame.sprite.Group()
        self.player = Nahida(
            int(settings.window_width * .5),
            int(settings.window_height * .75),
            10,
            os.path.join('resources', 'nahida_2.png'),
            target_size=(80, 130)
        )
        self.lives.add(self.player)

        # RL-style metadata
        self.action_space = list(Action)
        # observation_space is informal here (no gym dependency)
        self.observation_space = {
            'player_pos': (settings.window_width, settings.window_height),
        }

        self._terminated = False

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
        self.lives.update()

        # placeholder reward: small negative step penalty to encourage efficiency
        reward = -0.01
        done = False
        info = {}

        return self._get_observation(), reward, done, info

    def render(self):
        """Render one frame to the screen."""
        self.screen.fill(self.settings.window_background_color)
        self.lives.draw(self.screen)
        pygame.display.update()
        self.clock.tick(self.settings.FPS)

    def close(self):
        pygame.quit()
        # do not call sys.exit() here; let caller decide
