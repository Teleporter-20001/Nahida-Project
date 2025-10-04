import os
from app.Agent.Brains.BaseBrain import BaseBrain
from app.Agent.DataStructure import State, Action

import torch

from app.Agent.Networks.LinearNet import LinearNet
from app.common.Settings import Settings
from app.common.utils import printyellow


class SmartBrain1(BaseBrain):

    def __init__(self, teached=False):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'using device: {self.device}')
        # self.net = QNetwork(num_actions=len(Action), bullet_num=10, bullet_feat_dim=32).to(self.device)
        self.net = LinearNet(
            512,
            256,
            128,
            64,
            bullet_num=Settings.consider_bullets_num,
            num_actions=len(Action)
        ).to(self.device)
        self.model_path = os.path.join('Agent', 'models', f'LinearNet_{Settings.begin_episode}.pth')
        if self.model_path:
            try:
                self.net.load_state_dict(torch.load(self.model_path))
                print(f'loaded model from {self.model_path}')
            except Exception as e:
                printyellow(f'failed to load model from {self.model_path} because of: {e}')
        self.teached = teached

    def decide_action(self, state: State) -> Action:
        obs = state.observation
        if self.teached:
            return obs.get('human_action', Action.LEFTUP)
        with torch.no_grad():
            # 将 observation 转为 tensor，并添加 batch 维度
            obs_tensor = {
                'player_x': torch.tensor([obs['player_x']], dtype=torch.float32, device=self.device),
                'player_y': torch.tensor([obs['player_y']], dtype=torch.float32, device=self.device),
                'boss_from_player_x': torch.tensor([obs['boss_from_player_x']], dtype=torch.float32, device=self.device),
                'boss_from_player_y': torch.tensor([obs['boss_from_player_y']], dtype=torch.float32, device=self.device),
                'boss_velx': torch.tensor([obs['boss_velx']], dtype=torch.float32, device=self.device),
                'boss_vely': torch.tensor([obs['boss_vely']], dtype=torch.float32, device=self.device),
                # obs['nearest_bullets'] is typically a numpy array shape (k,5) or a list;
                # use torch.as_tensor to avoid creating a list-of-ndarray which triggers a warning
                'nearest_bullets': torch.as_tensor(obs['nearest_bullets'], dtype=torch.float32, device=self.device).unsqueeze(0),  # (1, k, 5)
            }
            q_values = self.net(obs_tensor)  # (1, num_actions)
            action_idx = q_values.argmax(dim=1).item()
            return list(Action)[action_idx]