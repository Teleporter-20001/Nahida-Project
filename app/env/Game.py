import time

import pygame
import sys

from app.Agent.Brains.BaseBrain import BaseBrain
from app.common.utils import printpurple, printred, printgreen
from app.env.TouhouEnv import TouhouEnv
from app.common.Settings import Settings


class Game:
    """Top-level manager that coordinates an Agent and the TouhouEnv.

    Responsibilities:
    - create the environment
    - run episodes where it asks agent for actions and steps the env
    - handle pygame events and rendering
    """

    def __init__(self, env: TouhouEnv | None = None):
        # pygame.init()
        self.env: TouhouEnv = env or TouhouEnv(Settings)

    def run_episode(self, agent: BaseBrain, max_steps: int = 100000, render: bool = True):
        """Run a single episode with the provided agent.

        Agent must implement decide_action(state: State) -> Action.
        Returns cumulative reward for the episode.
        """
        state = self.env.reset()
        if hasattr(agent, 'reset_hidden_state'):
            agent.reset_hidden_state()
        total_reward = 0.0

        for step in range(max_steps):
            t1 = time.time()
            # handle events and forward to environment (so env can process timers)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    sys.exit()
                # forward other events to environment for handling (e.g., timed shooting)
                try:
                    self.env.handle_event(event)
                except AttributeError:
                    pass

            action = agent.decide_action(state)
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward
            state = next_state

            if render:
                self.env.render()

            if done:
                printred(f'Agent had kept alive for {step / Settings.FPS} seconds.')
                if self.env.boss.health <= 0:
                    printgreen('You win!')
                break

            t2 = time.time()
            printpurple(f'Episode {step + 1} took {(t2 - t1) * 1000} milliseconds')

        return total_reward

    def close(self):
        self.env.close()