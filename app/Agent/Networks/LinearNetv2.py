import torch
from torch import nn
import torch.nn.functional as F

from app.common.Settings import Settings


class LinearNetv2(nn.Module):

    def __init__(self, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, bullet_num=10, num_actions=9):
        super().__init__()
        self.bullet_num = bullet_num
        self.num_actions = num_actions
        self.fc1 = nn.Linear(6 + self.bullet_num * 7, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, hidden_dim4)
        self.fc5 = nn.Linear(hidden_dim4, num_actions)

    def forward(self, obs):
        """
        obs 是 dict:
            {
                'player_x': tensor(batch,),
                'player_y': tensor(batch,),
                'boss_from_player_x': tensor(batch,),
                'boss_from_player_y': tensor(batch,),
                'boss_velx': tensor(batch,),
                'boss_vely': tensor(batch,),
                'nearest_bullets': tensor(batch, k, 5),
            }
        """
        other = self._obs_to_tensor(obs)
        batch_size = other.shape[0]
        # (batch, k, 5) -> (batch, k*5)
        bullet_feats = obs['nearest_bullets'].reshape(batch_size, self.bullet_num * 7)
        other_p = torch.cat([other, bullet_feats], dim=1)

        x = F.selu(self.fc1(other_p))
        x = F.selu(self.fc2(x))
        x = F.selu(self.fc3(x))
        x = F.selu(self.fc4(x))
        q_values = self.fc5(x)

        return q_values

    def _obs_to_tensor(self, obs: dict) -> torch.Tensor:
        return torch.stack([
            (obs['player_x'] - Settings.window_width / 2) * 2 / Settings.window_width,
            (obs['player_y'] - Settings.window_height / 2) * 2 / Settings.window_height,
            obs['boss_from_player_x'],
            obs['boss_from_player_y'],
            obs['boss_velx'],
            obs['boss_vely']
        ], dim=1)