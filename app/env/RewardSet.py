from dataclasses import dataclass

from app.common.Settings import Settings

@dataclass
class RewardSet:

    def __init__(self):
        self.survive_reward: float = 0
        self.edge_reward: float = 0
        self.hit_reward: float = 0
        self.kill_reward: float = 0
        self.behit_reward: float = 0
        self.avoid_reward: float = 0

        self.settings: Settings = Settings()

    def survive(self):
        self.survive_reward += self.settings.alive_reward

    def edge_punish(self, value: float):
        self.edge_reward += value

    def hit_enemy(self):
        self.hit_reward += self.settings.hit_reward

    def killenemy(self):
        self.kill_reward += self.settings.kill_boss_reward

    def behit(self):
        self.behit_reward += self.settings.behit_reward

    def avoid(self):
        self.avoid_reward += self.settings.avoid_reward

    @property
    def value(self):
        return self.survive_reward + self.edge_reward + self.hit_reward + self.kill_reward + self.behit_reward + self.avoid_reward

    @property
    def attrs(self):
        return self.survive_reward, self.edge_reward, self.hit_reward, self.kill_reward, self.behit_reward, self.avoid_reward

    def __add__(self, other):
        if not isinstance(other, RewardSet):
            return NotImplemented
        result = RewardSet()
        result.survive_reward = self.survive_reward + other.survive_reward
        result.edge_reward = self.edge_reward + other.edge_reward
        result.hit_reward = self.hit_reward + other.hit_reward
        result.kill_reward = self.kill_reward + other.kill_reward
        result.behit_reward = self.behit_reward + other.behit_reward
        result.avoid_reward = self.avoid_reward + other.avoid_reward
        return result
    
    