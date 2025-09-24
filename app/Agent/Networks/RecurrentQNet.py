import torch
from torch import nn
import torch.nn.functional as F

from app.common.Settings import Settings
from app.common.utils import printyellow


class RecurrentQNet(nn.Module):
    def __init__(self, hidden_dim1, hidden_dim2, hidden_dim3,
                 bullet_num=10, num_actions=9, rnn_hidden_dim=512, rnn_type="gru"):
        super().__init__()
        self.bullet_num = bullet_num
        self.num_actions = num_actions
        self.rnn_hidden_dim = rnn_hidden_dim

        input_dim = 6 + self.bullet_num * 5  # 与 LinearNet 相同

        # RNN 层
        if rnn_type.lower() == "gru":
            self.rnn = nn.GRU(input_dim, rnn_hidden_dim, batch_first=True)
        elif rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(input_dim, rnn_hidden_dim, batch_first=True)
        else:
            raise ValueError("rnn_type 必须是 'gru' 或 'lstm'")

        # 后续 MLP 层
        self.fc1 = nn.Linear(rnn_hidden_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc_out = nn.Linear(hidden_dim3, num_actions)

    def forward(self, obs, hidden_state=None):
        """
        obs: dict
            {
                'player_x': (batch, seq_len),
                'player_y': (batch, seq_len),
                'boss_from_player_x': (batch, seq_len),
                'boss_from_player_y': (batch, seq_len),
                'boss_velx': (batch, seq_len),
                'boss_vely': (batch, seq_len),
                'nearest_bullets': (batch, seq_len, k, 5),
            }
        hidden_state:
            - 对于 GRU: (1, batch, rnn_hidden_dim)
            - 对于 LSTM: (h, c), 各是 (1, batch, rnn_hidden_dim)
            - 可以为 None（表示用 0 初始化）
        """
        other = self._obs_to_tensor(obs)  # (batch, seq_len, 6)
        if other.dim() == 2:
            printyellow('seq dim lost for other')
            other = other.unsqueeze(1)  # (batch, 1, 6)
        b, t, _ = other.shape
        
        bullet_tensor = obs['nearest_bullets']
        # 兼容 (batch, k, 5) 情况
        if bullet_tensor.dim() == 3:
            printyellow('seq dim lost for bullets')
            bullet_tensor = bullet_tensor.unsqueeze(1)  # -> (batch, 1, k, 5)

        # (batch, seq_len, k, 5) -> (batch, seq_len, k*5)
        bullet_feats = bullet_tensor.reshape(b, t, self.bullet_num * 5)

        # 拼接
        x = torch.cat([other, bullet_feats], dim=-1)  # (batch, seq_len, input_dim)

        # 送入 RNN
        rnn_out, new_hidden = self.rnn(x, hidden_state)  # rnn_out: (batch, seq_len, rnn_hidden_dim)

        # 我们只取最后一步的输出作为 Q 值的输入
        last_out = rnn_out[:, -1, :]  # (batch, rnn_hidden_dim)

        # MLP
        x = F.selu(self.fc1(last_out))
        x = F.selu(self.fc2(x))
        x = F.selu(self.fc3(x))
        q_values = self.fc_out(x)
        # q_values = torch.as_tensor(q_values, device=q_values.device)

        return q_values, new_hidden

    def _obs_to_tensor(self, obs: dict) -> torch.Tensor:
        """
        把玩家和 Boss 的特征组合成 (batch, seq_len, 6) 张量
        """
        return torch.stack([
            (obs['player_x'] - Settings.window_width / 2) * 2 / Settings.window_width,
            (obs['player_y'] - Settings.window_height / 2) * 2 / Settings.window_height,
            obs['boss_from_player_x'],
            obs['boss_from_player_y'],
            obs['boss_velx'],
            obs['boss_vely']
        ], dim=-1)
