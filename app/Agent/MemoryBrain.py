import os
from typing import Deque

import numpy as np
import torch

from app.Agent.BaseBrain import BaseBrain
from app.Agent.DataStructure import State, Action
from app.Agent.Networks.LinearNetv2 import LinearNetv2
from app.common.Settings import Settings


class MemoryBrain(BaseBrain):

    def __init__(self, be_teached=False):
        super().__init__()
        self.be_teached_ = be_teached
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'using device: {self.device}')
        self.net = LinearNetv2(
            512,
            256,
            128,
            64,
            bullet_num=10,
            num_actions=len(Action)
        ).to(self.device)

        self.load_model = False
        self.model_path = os.path.join('Agent', 'models', f'LinearNetv2_{Settings.begin_episode}.pth')
        if self.load_model and os.path.exists(self.model_path):
            try:
                self.net.load_state_dict(torch.load(self.model_path, map_location=self.device))
                print(f'model loaded from {self.model_path}')
            except RuntimeError as e:
                print(f'model not found at {self.model_path}, \nor there was an error: {e}')

        self.bullet_history = Deque(maxlen=2)

    def _match_and_compute_features(self, cur_bullets, prev_bullets):
        """
        cur_bullets: (k, 5) ndarray, each row [dx, dy, vx, vy, radius]
        prev_bullets: (k, 5) ndarray
        return: (k, 7) ndarray, each row [dx, dy, vx, vy, radius, dvx, dvy]
        """
        if prev_bullets is None or len(prev_bullets) == 0:
            # 没有历史帧，Δv=0
            zeros = np.zeros((cur_bullets.shape[0], 2), dtype=np.float32)
            return np.concatenate([cur_bullets, zeros], axis=1)

        # 最近邻匹配
        extended_feats = []
        for b in cur_bullets:
            dx, dy, vx, vy, r = b
            # 找到上一帧最近的子弹
            dists = np.linalg.norm(prev_bullets[:, :2] - b[:2], axis=1)
            j = np.argmin(dists)
            prev_vx, prev_vy = prev_bullets[j, 2:4]
            dvx, dvy = vx - prev_vx, vy - prev_vy
            extended_feats.append([dx, dy, vx, vy, r, dvx, dvy])
        return np.array(extended_feats, dtype=np.float32)

    def decide_action(self, state: State) -> Action:
        obs = state.observation
        if self.be_teached_:
            return obs.get('human_action', Action.LEFTUP)

        # 当前子弹信息
        cur_bullets = np.array(obs['nearest_bullets'], dtype=np.float32)  # (k,5), k=10
        prev_bullets = self.bullet_history[-1] if len(self.bullet_history) > 0 else np.zeros_like(cur_bullets)

        # 构造增强特征
        extended_bullets = self._match_and_compute_features(cur_bullets, prev_bullets)

        # TODO: 在这里就把另2维特征加进去，不要在下面处理，下面函数给外面调的时候历史是乱的，在这里加的历史才是对的，加速度才是对的

        # 更新历史
        self.bullet_history.append(cur_bullets)

        with torch.no_grad():
            obs_tensor = {
                'player_x': torch.tensor([obs['player_x']], dtype=torch.float32, device=self.device),
                'player_y': torch.tensor([obs['player_y']], dtype=torch.float32, device=self.device),
                'boss_from_player_x': torch.tensor([obs['boss_from_player_x']], dtype=torch.float32, device=self.device),
                'boss_from_player_y': torch.tensor([obs['boss_from_player_y']], dtype=torch.float32, device=self.device),
                'boss_velx': torch.tensor([obs['boss_velx']], dtype=torch.float32, device=self.device),
                'boss_vely': torch.tensor([obs['boss_vely']], dtype=torch.float32, device=self.device),
                # (1, k, 7)
                'nearest_bullets': torch.as_tensor(extended_bullets, dtype=torch.float32, device=self.device).unsqueeze(0),
            }
            q_values = self.net(obs_tensor)  # (1, num_actions)
            action_idx = q_values.argmax(dim=1).item()
            return list(Action)[action_idx]

    def augment_bullet_features(self, bullets_5d):
        """
        输入: bullets_5d, shape (k, 5)，每行 [rel_x, rel_y, vel_x, vel_y, radius]
        输出: bullets_7d, shape (k, 7)，额外增加 [accel_x, accel_y]
        """
        if isinstance(bullets_5d, torch.Tensor):
            bullets_np = bullets_5d.cpu().numpy()
        else:
            bullets_np = np.asarray(bullets_5d, dtype=np.float32)

        k = bullets_np.shape[0]
        bullets_7d = np.zeros((k, 7), dtype=np.float32)
        bullets_7d[:, :5] = bullets_np

        accel_x, accel_y = np.zeros(k, dtype=np.float32), np.zeros(k, dtype=np.float32)

        prev_bullets = self.bullet_history[-1] if len(self.bullet_history) > 0 else np.zeros_like(bullets_np)
        # 用索引追踪 (假设每次输入的 bullets 顺序一致)
        for i in range(k):
            vx, vy = bullets_np[i, 2], bullets_np[i, 3]
            if i in prev_bullets:
                prev_vx, prev_vy = prev_bullets[i, 2], prev_bullets[i, 3]
                accel_x[i] = vx - prev_vx
                accel_y[i] = vy - prev_vy
            # prev_bullets[i] = (vx, vy)

        bullets_7d[:, 5] = accel_x
        bullets_7d[:, 6] = accel_y
        return bullets_7d
