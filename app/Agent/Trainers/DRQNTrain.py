import os
import random
import atexit
from collections import deque, namedtuple

import numpy as np
import torch
from torch import optim, nn
from torch.nn.utils import clip_grad_norm_

from app.Agent.Brains.BaseNetBrain import BaseNetBrain
from app.Agent.Brains.MemoryBrainV2 import MemoryBrainV2
from app.Agent.DataStructure import Action, State
from app.Agent.Networks.RecurrentQNet import RecurrentQNet
from app.Agent.Trainers.DQNTrain import ReplayBuffer
from app.common.Settings import Settings
from app.common.utils import printgreen

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class SeqReplayBuffer(ReplayBuffer):
    def __init__(self, capacity=300000, chunk_size=5000, seq_len=10):
        super().__init__(capacity, chunk_size)
        self.seq_len = seq_len

        # 全局 step 计数器（不会回绕，永远递增）
        self._counter = 0

        # 存储合法起点（全局索引，不受 deque 丢弃影响）
        self.valid_starts = deque()

        # 记录 buffer 中数据的起始全局索引
        self._start_index = 0

        # 如果加载过旧数据，尝试重建
        if len(self.buffer) > 0:
            self._counter = len(self.buffer)
            self._rebuild_valid_starts()

    def _rebuild_valid_starts(self):
        """全量重建 valid_starts（只在 load 时或必要时调用）"""
        self.valid_starts.clear()
        buf = list(self.buffer)
        for i in range(len(buf) - self.seq_len):
            if not any(tr.done for tr in buf[i:i+self.seq_len]):
                self.valid_starts.append(self._start_index + i)

    def push(self, *args):
        """
        append 一个 Transition，并维护 valid_starts。
        """
        super().push(*args)
        idx = self._counter  # 当前 transition 的全局索引
        self._counter += 1

        # 如果 buffer 溢出，意味着 _start_index 也要右移
        if len(self.buffer) == self.buffer.maxlen:
            self._start_index = self._counter - len(self.buffer)

        # done=True 时，清理所有受影响的起点
        if args[-1] is True:
            # 移除所有跨过该 done 的起点
            cutoff = idx - self.seq_len + 1
            while self.valid_starts and self.valid_starts[-1] >= cutoff:
                self.valid_starts.pop()
            return

        # 如果 buffer 不够长，不构成合法序列
        if len(self.buffer) < self.seq_len:
            return

        # 最近序列的起点
        start = idx - self.seq_len + 1

        # 只要起点 >= buffer 起始位置，并且最近 seq 内没有 done，就合法
        if start >= self._start_index:
            seq = [self.buffer[start - self._start_index + j] for j in range(self.seq_len)]
            if not any(tr.done for tr in seq):
                self.valid_starts.append(start)

    def sample(self, batch_size):
        """
        从 valid_starts 中采样 batch_size 个序列。
        """
        if len(self.valid_starts) == 0:
            # raise ValueError("No valid sequence available in RNNReplayBuffer")
            return None

        starts = random.sample(self.valid_starts, batch_size)
        batch = []
        for s in starts:
            # 把全局索引 s 换算到 buffer 内部索引
            offset = s - self._start_index
            seq = [self.buffer[offset + j] for j in range(self.seq_len)]
            batch.append(seq)
        return batch

    def __len__(self):
        return len(self.buffer)

    def num_valid_sequences(self):
        """返回当前合法序列数"""
        return len(self.valid_starts)


class DRQNTrainer:
    
    def __init__(self, brain: MemoryBrainV2, seq_len: int, gamma=Settings.gamma, lr=Settings.gamma, batch_size=Settings.batch_size, target_update=Settings.target_update):

        self.brain = brain
        self.device = brain.device
        self.policy_net = brain.net
        self.target_net = RecurrentQNet(
            256, 512, 256,
            bullet_num=10,
            num_actions=len(Action)
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        self.seq_len = seq_len
        self.buffer = SeqReplayBuffer(seq_len=seq_len)
        self.steps_done = 0
        self.has_begun_to_replay = False
        
    def select_action(self, state: State, epsilon=0.1):
        """epsilon-greedy"""
        if random.random() < epsilon:
            return random.choice(list(Action))
        else:
            with torch.no_grad():
                return self.brain.decide_action(state)
            
            
    def optimize_model(self):
        
        if len(self.buffer) < self.batch_size or self.buffer.num_valid_sequences() < self.batch_size:
            return
        
        # transitions = self.buffer.sample(self.batch_size)  # list of [Transition,...] seq_len
        # batch = Transition(*zip(*transitions))
        sequences = self.buffer.sample(self.batch_size)  # list of [Transition,...] seq_len
        if sequences is None:
            return

        # really can begin to replay
        if not self.has_begun_to_replay:
            printgreen('Begin to replay sequences...')
            self.has_begun_to_replay = True
        
         # 1. 将序列列表转换为状态、动作等各自的列表
        # obs_sequences will be a list of lists of observation dicts
        obs_sequences = [[tr.state for tr in seq] for seq in sequences]
        action_sequences = [[tr.action for tr in seq] for seq in sequences]
        reward_sequences = [[tr.reward for tr in seq] for seq in sequences]
        next_obs_sequences = [[tr.next_state for tr in seq] for seq in sequences]
        done_sequences = [[tr.done for tr in seq] for seq in sequences]

        # 2. 将数据批处理为正确的张量形状 (batch, seq_len, ...)
        state_batch = self._batch_sequences(obs_sequences)
        next_state_batch = self._batch_sequences(next_obs_sequences)

        # 动作、奖励和完成标志通常只关心序列的最后一步
        # 或者需要根据你的模型设计来处理整个序列
        # 这里我们假设使用序列的最后一个动作和奖励
        last_actions = [seq[-1] for seq in action_sequences]
        last_rewards = [seq[-1] for seq in reward_sequences]
        last_dones = [seq[-1] for seq in done_sequences]
        
        # 组装 batch 数据
        # state_batch = self._batch_obs(batch.state)
        # next_state_batch = self._batch_obs(batch.next_state)
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

        # action_indices = [_to_idx(a) for a in batch.action]
        # action_batch = torch.tensor(action_indices, dtype=torch.long, device=self.device).unsqueeze(1)
        # reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        # done_batch = torch.tensor(batch.done, dtype=torch.float32, device=self.device)
        action_batch = torch.tensor([_to_idx(a) for a in last_actions], dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(last_rewards, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(last_dones, dtype=torch.float32, device=self.device)

        # 当前 Q(s, a)
        # q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()
        
        # policy_net 需要接收整个序列
        q_values, _ = self.policy_net(state_batch) # (batch, num_actions)
        q_s_a = q_values.gather(1, action_batch).squeeze()

        # 目标 Q 值
        with torch.no_grad():
            # next_q_values = self.target_net(next_state_batch).max(1)[0]
            # target_q = reward_batch + self.gamma * next_q_values * (1 - done_batch)
            
            # target_net 也接收整个序列
            next_q_values, _ = self.target_net(next_state_batch)
            max_next_q = next_q_values.max(1)[0]
            target_q = reward_batch + self.gamma * max_next_q * (1 - done_batch)

        # 损失
        # loss = nn.MSELoss()(q_values, target_q)
        loss = nn.MSELoss()(q_s_a, target_q)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # 更新 target 网络
        if self.steps_done % self.target_update == 0:
            print('network updated')
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.steps_done += 1
        
    def _batch_sequences(self, seq_list):
        """将 observation 的序列列表转换为批处理张量字典"""
        batch_size = len(seq_list)
        seq_len = len(seq_list[0])
        
        # Flatten the list of sequences to a list of observations
        flat_obs = [obs for seq in seq_list for obs in seq]
        
        # Use the existing _batch_obs to convert the flat list to tensors
        batched_obs_flat = self._batch_obs(flat_obs)
        
        # Reshape the tensors to (batch_size, seq_len, ...)
        final_batch = {}
        for key, tensor in batched_obs_flat.items():
            if key == 'nearest_bullets':
                # (batch*seq, k, 5) -> (batch, seq, k, 5)
                k = tensor.shape[1]
                feat_dim = tensor.shape[2]
                final_batch[key] = tensor.view(batch_size, seq_len, k, feat_dim)
            else:
                # (batch*seq) -> (batch, seq)
                final_batch[key] = tensor.view(batch_size, seq_len)
        return final_batch
        
    def _batch_obs(self, obs_list):
        """把一批 obs dict 转成 batch tensor dict"""
        return {
            'player_x': torch.tensor([o['player_x'] for o in obs_list], dtype=torch.float32, device=self.device),
            'player_y': torch.tensor([o['player_y'] for o in obs_list], dtype=torch.float32, device=self.device),
            'boss_from_player_x': torch.tensor([o['boss_from_player_x'] for o in obs_list], dtype=torch.float32, device=self.device),
            'boss_from_player_y': torch.tensor([o['boss_from_player_y'] for o in obs_list], dtype=torch.float32, device=self.device),
            'boss_velx': torch.tensor([o['boss_velx'] for o in obs_list], dtype=torch.float32, device=self.device),
            'boss_vely': torch.tensor([o['boss_vely'] for o in obs_list], dtype=torch.float32, device=self.device),
            # nearest_bullets may be a list of numpy arrays; convert once to a single numpy array to avoid
            # the slow path of creating a tensor from a list of ndarrays (warned by PyTorch).
            'nearest_bullets': self._nearest_bullets_tensor([o['nearest_bullets'] for o in obs_list]),
        }
        
    def _nearest_bullets_tensor(self, nb_list):
        """Convert a list of nearest_bullets (possibly numpy arrays) to a single torch tensor on device.

        Accepts: nb_list: list of arrays or lists, shape per-item = (bullet_num, bullet_feat_dim)
        Returns: torch.FloatTensor of shape (batch, bullet_num, bullet_feat_dim) on self.device
        """
        # Fast path: if already a numpy array
        if isinstance(nb_list, np.ndarray):
            arr = nb_list
        else:
            # Try to stack into a numpy array. Use dtype float32 for compatibility with torch.float32.
            try:
                arr = np.array(nb_list, dtype=np.float32)
            except Exception:
                # Last-resort: build with python list comprehension and explicit conversion
                arr = np.stack([np.array(x, dtype=np.float32) for x in nb_list], axis=0)

        # Ensure dtype float32
        if arr.dtype != np.float32:
            try:
                arr = arr.astype(np.float32)
            except Exception:
                arr = arr.astype(np.float32, copy=False)

        # Convert to torch tensor and move to device
        try:
            t = torch.from_numpy(arr).to(self.device)
        except Exception:
            # Fallback: create tensor from list (slower), but keep dtype/device consistent
            t = torch.tensor(arr, dtype=torch.float32, device=self.device)
        return t
            
            
    # def optimize_model(self):
    #     if self.buffer.num_valid_sequences() < self.batch_size:
    #         return

    #     batches = self.buffer.sample(self.batch_size)  # list of [Transition,...] seq_len
    #     # 按照你的网络输入格式组装batch
    #     obs_batch = {  # shape: (batch, seq_len, ...)
    #         'player_x': [],
    #         'player_y': [],
    #         'boss_from_player_x': [],
    #         'boss_from_player_y': [],
    #         'boss_velx': [],
    #         'boss_vely': [],
    #         'nearest_bullets': [],
    #     }
    #     action_batch, reward_batch, done_batch, next_obs_batch = [], [], [], {k: [] for k in obs_batch}

    #     for seq in batches:
    #         obs_seq = [tr.state for tr in seq]
    #         next_obs_seq = [tr.next_state for tr in seq]
    #         action_seq = [tr.action for tr in seq]
    #         reward_seq = [tr.reward for tr in seq]
    #         done_seq = [tr.done for tr in seq]

    #         for k in obs_batch:
    #             obs_batch[k].append([o[k] for o in obs_seq])
    #             next_obs_batch[k].append([o[k] for o in next_obs_seq])
    #         action_batch.append(action_seq)
    #         reward_batch.append(reward_seq)
    #         done_batch.append(done_seq)

    #     # 转为tensor
    #     for k in obs_batch:
    #         obs_batch[k] = torch.tensor(obs_batch[k], dtype=torch.float32, device=self.device)
    #         next_obs_batch[k] = torch.tensor(next_obs_batch[k], dtype=torch.float32, device=self.device)
    #     action_batch = torch.tensor(action_batch, dtype=torch.long, device=self.device)
    #     reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=self.device)
    #     done_batch = torch.tensor(done_batch, dtype=torch.float32, device=self.device)

    #     # RNN网络forward
    #     q_values, _ = self.policy_net(obs_batch)  # (batch, num_actions)
    #     # 只取最后一步的动作
    #     last_actions = action_batch[:, -1]
    #     last_rewards = reward_batch[:, -1]
    #     last_dones = done_batch[:, -1]
    #     q_s_a = q_values.gather(1, last_actions.unsqueeze(1)).squeeze()

    #     with torch.no_grad():
    #         next_q_values, _ = self.target_net(next_obs_batch)
    #         max_next_q = next_q_values.max(1)[0]
    #         target_q = last_rewards + self.gamma * max_next_q * (1 - last_dones)

    #     loss = nn.MSELoss()(q_s_a, target_q)

    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     clip_grad_norm_(self.policy_net.parameters(), 1.0)
    #     self.optimizer.step()

    #     if self.steps_done % self.target_update == 0:
    #         self.target_net.load_state_dict(self.policy_net.state_dict())
    #     self.steps_done += 1