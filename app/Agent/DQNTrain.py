import os.path
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import atexit

from app.Agent.Networks.LinearNet import LinearNet
from app.Agent.SmartBrain1 import State, Action, SmartBrain1
from app.Agent.Networks.FeatureExtractor import QNetwork

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:

    filepath = os.path.join('Agent', 'models', 'replay_buffer.pkl')

    def __init__(self, capacity=4000000):
        self.buffer = deque(maxlen=capacity)
        isload = True
        if isload:
            self.load()
        atexit.register(self.save)

    def push(self, *args):
        self.buffer.append(Transition(*args))
        if len(self.buffer) % 1000 == 0:
            print(f'Replay Buffer has {len(self.buffer)} transitions now')
        if len(self.buffer) % 10000 == 0:
            try:
                self.save()
                print(f'Saved replay buffer to {self.filepath}')
            except Exception as e:
                print(f'Failed to save replay buffer due to an error: {e}')

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

    def save(self):
        torch.save(self.buffer, self.filepath)

    def load(self):
        try:
            data = torch.load(self.filepath)
            self.buffer = deque(data, maxlen=self.buffer.maxlen)
            print(f'loaded replay buffer from {self.filepath}, now buffer size {len(self.buffer)}')
        except EOFError as e:
            print(f'Failed to load replay buffer from {self.filepath} due to EOFError: {e}')
        except FileNotFoundError as e:
            print(f'Failed to load replay buffer from {self.filepath} due to FileNotFoundError: {e}')


class DQNTrainer:
    def __init__(self, brain: SmartBrain1, gamma=0.99, lr=1e-3, batch_size=1024, target_update=1500):
        self.brain = brain
        self.device = brain.device
        self.policy_net = brain.net
        # self.target_net = QNetwork(num_actions=len(Action), bullet_num=10, bullet_feat_dim=32).to(self.device)
        self.target_net = LinearNet(
            256,
            512,
            128,
            bullet_num=10,
            num_actions=len(Action)
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.buffer = ReplayBuffer()
        self.steps_done = 0

    def select_action(self, state: State, epsilon=0.1):
        """epsilon-greedy"""
        if random.random() < epsilon:
            # return random.randrange(len(Action))
            return random.choice(list(Action))
        else:
            with torch.no_grad():
                # obs_tensor = self.brain.net._obs_to_tensor(state.observation)
                # q_values = self.policy_net(obs_tensor)
                # return q_values.argmax(dim=1).item()
                return self.brain.decide_action(state)

    def optimize_model(self):
        if len(self.buffer) < self.batch_size:
            return

        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # 组装 batch 数据
        state_batch = self._batch_obs(batch.state)
        next_state_batch = self._batch_obs(batch.next_state)
        # batch.action 里应是 action 的整数索引。如果不确定类型，做一次安全转换：
        from app.Agent.DataStructure import Action as _ActionEnum

        def _to_idx(a):
            # 如果已经是 int 或可以转换为 int，直接返回
            if isinstance(a, int):
                return int(a)
            # 如果是 Action enum，返回其在 Action 列表中的索引
            if isinstance(a, _ActionEnum):
                return list(_ActionEnum).index(a)
            # 其他可转换类型（例如 numpy.int64）也尝试直接转换
            try:
                return int(a)
            except Exception:
                raise ValueError(f"Unsupported action type in buffer: {type(a)}")

        action_indices = [_to_idx(a) for a in batch.action]
        action_batch = torch.tensor(action_indices, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=self.device)

        # 当前 Q(s, a)
        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()

        # 目标 Q 值
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        # 损失
        loss = nn.MSELoss()(q_values, target_q)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # 更新 target 网络
        if self.steps_done % self.target_update == 0:
            print('network updated')
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.steps_done += 1

    def _batch_obs(self, obs_list):
        """把一批 obs dict 转成 batch tensor dict"""
        return {
            'player_x': torch.tensor([o['player_x'] for o in obs_list], dtype=torch.float32, device=self.device),
            'player_y': torch.tensor([o['player_y'] for o in obs_list], dtype=torch.float32, device=self.device),
            'boss_from_player_x': torch.tensor([o['boss_from_player_x'] for o in obs_list], dtype=torch.float32, device=self.device),
            'boss_from_player_y': torch.tensor([o['boss_from_player_y'] for o in obs_list], dtype=torch.float32, device=self.device),
            'boss_velx': torch.tensor([o['boss_velx'] for o in obs_list], dtype=torch.float32, device=self.device),
            'boss_vely': torch.tensor([o['boss_vely'] for o in obs_list], dtype=torch.float32, device=self.device),
            'nearest_bullets': torch.tensor([o['nearest_bullets'] for o in obs_list], dtype=torch.float32, device=self.device),
        }
