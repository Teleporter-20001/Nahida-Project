import pygame
import sys

from app.Agent.BaseBrain import BaseBrain
from app.Agent.RandomBrain import RandomBrain
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

    def run_episode(self, agent: BaseBrain, max_steps: int = 1000, render: bool = True):
        """Run a single episode with the provided agent.

        Agent must implement decide_action(state: State) -> Action.
        Returns cumulative reward for the episode.
        """
        state = self.env.reset()
        total_reward = 0.0

        for step in range(max_steps):
            # handle quit events so window can be closed
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    sys.exit()

            action = agent.decide_action(state)
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward
            state = next_state

            if render:
                self.env.render()

            if done:
                break

        return total_reward

    def close(self):
        self.env.close()