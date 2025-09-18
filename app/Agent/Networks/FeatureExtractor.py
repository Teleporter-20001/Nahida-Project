import torch
from torch import nn as nn
from torch.nn import functional as F

class BulletFeatureExtractor(nn.Module):
    """共享子弹特征提取器，对每发子弹提取 embedding"""
    def __init__(self, input_dim=5, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: (batch, k, 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))   # (batch, k, hidden_dim)
        return x

class QNetwork(nn.Module):
    def __init__(self, num_actions=9, bullet_num=10, bullet_feat_dim=32):
        super().__init__()
        self.bullet_extractor = BulletFeatureExtractor(input_dim=5, hidden_dim=bullet_feat_dim)

        # 其他特征: player_x, player_y, boss_from_player_x, boss_from_player_y, boss_velx, boss_vely
        self.other_feat_dim = 6
        self.num_actions = num_actions
        self.bullet_num = bullet_num
        self.bullet_feat_dim = bullet_feat_dim

        # 最终拼接向量: other_feat (6) + pooled_bullet_feat (32)
        self.fc1 = nn.Linear(self.other_feat_dim + bullet_feat_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, num_actions)

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

        # (batch, 6)
        other = self._obs_to_tensor(obs)

        # (batch, k, 5) → (batch, k, 32)
        bullet_feats = self.bullet_extractor(obs['nearest_bullets'])

        # 池化: mean pooling（也可以试 max pooling）
        bullet_pooled = bullet_feats.mean(dim=1)   # (batch, 32)

        # 拼接
        x = torch.cat([other, bullet_pooled], dim=1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc_out(x)   # (batch, num_actions)

        return q_values

    def _obs_to_tensor(self, obs: dict) -> torch.Tensor:
        return torch.stack([
            obs['player_x'],
            obs['player_y'],
            obs['boss_from_player_x'],
            obs['boss_from_player_y'],
            obs['boss_velx'],
            obs['boss_vely']
        ], dim=1)


