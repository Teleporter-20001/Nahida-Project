import os
import time
import random
import traceback

import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch.optim as optim
from collections import deque, namedtuple
import atexit
import numpy as np

from app.Agent.Networks.LinearNet import LinearNet
from app.Agent.Brains.SmartBrain1 import State, Action, SmartBrain1
from app.common.Settings import Settings

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:

    # store per-chunk files under app/Agent/transitions
    transitions_dir = os.path.join('Agent', 'transitions')
    # legacy single-file path (kept for backward compatibility on load)
    # legacy_filepath = os.path.join('Agent', 'models', 'replay_buffer.pkl')
    legacy_filepath = ''

    def __init__(self, capacity=300000, chunk_size=5000):
        self.buffer = deque(maxlen=capacity)
        self.chunk_size = chunk_size
        # internal counter to track how many new transitions since last on-disk save
        self._since_last_save = 0
        # try load existing saved transitions
        isload = True
        if isload:
            self.load()
        atexit.register(self.save)

    def push(self, *args):
        """Append a transition and save to disk every chunk_size new transitions.

        Uses an internal counter so we only save when new data has been added since
        the last save. This avoids repeatedly saving identical data when the buffer
        is full and older items are being dropped.
        """
        self.buffer.append(Transition(*args))
        # progress log
        if self._since_last_save % 1000 == 0:
            print(f'Replay Buffer has got {self._since_last_save} more transitions now')

        # increase counter of new transitions since last save
        self._since_last_save += 1

        # save when we've accumulated chunk_size new transitions
        if self._since_last_save >= self.chunk_size:
            try:
                saved = self.save()
                if saved:
                    print(f'Saved latest replay chunk {saved} to transitions directory')
            except Exception as e:
                print(f'Failed to save replay buffer due to an error: {e}')
            finally:
                # reset counter regardless of save success to avoid tight retry loops
                self._since_last_save = 0

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

    def _ensure_dir(self):
        try:
            os.makedirs(self.transitions_dir, exist_ok=True)
        except Exception:
            # best-effort; if it fails, higher-level save will raise
            pass

    def save(self):
        """
        Save only the latest chunk (size=self.chunk_size) into a new file
        under app/Agent/transitions. Returns the path of saved file or None.
        """
        # self._ensure_dir()
        try:
            # take the latest chunk_size transitions
            if len(self.buffer) == 0:
                return None
            # convert to list to slice
            buf_list = list(self.buffer)
            last_chunk = buf_list[-self.chunk_size:]
            # filename with timestamp and total-size to help ordering
            ts = int(time.time() * 1000)
            fname = f'replay_chunk_{ts}_{len(self.buffer)}.pkl'
            fpath = os.path.join(self.transitions_dir, fname)
            torch.save(last_chunk, fpath)
            return fpath
        except Exception as e:
            # bubble up to caller
            traceback.print_exc()
            print(f'An error occurred during saving replay chunk: {e}')
            raise

    def load(self):
        """
        Load all saved chunk files from transitions_dir (sorted by filename)
        into the deque. Also fallback to legacy single-file path if present.
        后读入的内容覆盖前面的内容，保证buffer里是最新的transition。
        """
        loaded = 0
        # 先清空buffer，保证不会混入旧数据
        self.buffer.clear()
        # 先加载legacy文件（最旧）
        try:
            if os.path.exists(self.legacy_filepath):
                try:
                    data = torch.load(self.legacy_filepath)
                    items = list(data) if not isinstance(data, deque) else list(data)
                    for t in items:
                        self.buffer.append(t)
                    loaded += len(items)
                    print(f'loaded legacy replay buffer from {self.legacy_filepath}, now buffer size {len(self.buffer)}')
                except Exception as e:
                    print(f'Failed to load legacy replay buffer from {self.legacy_filepath}: {e}')
        except Exception:
            pass

        # 再加载chunk文件（按时间顺序，后面的覆盖前面的）
        try:
            if os.path.isdir(self.transitions_dir):
                files = [f for f in os.listdir(self.transitions_dir) if f.endswith('.pkl')]
                files.sort()
                for fname in files:
                    fpath = os.path.join(self.transitions_dir, fname)
                    try:
                        data = torch.load(fpath)
                        # 追加新数据
                        for t in data:
                            self.buffer.append(t)
                        loaded += len(data)
                        # 如果超出容量，保留最新的
                        if self.buffer.maxlen is not None and len(self.buffer) > self.buffer.maxlen:
                            # deque自动丢弃最旧的，无需手动裁剪
                            pass
                    except EOFError as e:
                        print(f'Failed to load {fpath} due to EOFError: {e}')
                    except Exception as e:
                        print(f'Failed to load {fpath}: {e}')
                if loaded > 0:
                    print(f'loaded {loaded} transitions from {self.transitions_dir}, now buffer size {len(self.buffer)}')
        except Exception as e:
            print(f'Error while loading transitions from {self.transitions_dir}: {e}')


class DQNTrainer:
    def __init__(self, brain: SmartBrain1, gamma=Settings.gamma, lr=Settings.lr, batch_size=Settings.batch_size, target_update=Settings.target_update):
        self.brain = brain
        self.device = brain.device
        self.policy_net = brain.net
        # self.target_net = QNetwork(num_actions=len(Action), bullet_num=10, bullet_feat_dim=32).to(self.device)
        self.target_net = LinearNet(
            512,
            256,
            128,
            64,
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
            return random.choice(list(Action))
        else:
            with torch.no_grad():
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
        clip_grad_norm_(self.policy_net.parameters(), 1.0)
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

    # def _nearest_bullets_tensor2(self, nb_list):
    #     """Convert list of nearest_bullets (shape=(bullet_num,5)) to (batch, bullet_num,7)"""
    #     # 用智能体的函数统一处理
    #     processed = []
    #     for bullets in nb_list:
    #         # TODO: 检测特征是否已经是7维，如果是就跳过，如果不是就转化一次并**写入原来的地方**
    #         bullets_7d = self.brain.augment_bullet_features(bullets)
    #         processed.append(bullets_7d)
    #
    #     arr = np.stack(processed, axis=0).astype(np.float32)
    #     t = torch.from_numpy(arr).to(self.device)
    #     return t

