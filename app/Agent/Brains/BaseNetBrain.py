import torch
from app.Agent.Brains.BaseBrain import BaseBrain
from app.Agent.DataStructure import State, Action
from app.common.utils import printgreen, printred


class BaseNetBrain(BaseBrain):
    
    def __init__(self, be_teached=False):
        super().__init__()
        self.be_teached = be_teached
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        printgreen(f'using device: {self.device}')
        self._create_network()
        self._hidden_state = torch.zeros(1, 1, 512, device=self.device)
        
    def _create_network(self):
        """子类需实现此方法以创建网络"""
        self.net: torch.nn.Module | None = None

        raise NotImplementedError("_create_network 方法未被实现")
    
    def decide_action(self, state: State) -> Action:

        if self.net is None:
            raise ValueError("网络尚未创建")

        obs = state.observation
        if self.be_teached:
            return obs.get('human_action', Action.LEFTUP)
        
        with torch.no_grad():
            
            # 原标量先做 batch 维 (1,) -> 再加 seq 维 (1,1)
            def _scalar(v):
                return torch.tensor([[v]], dtype=torch.float32, device=self.device)  # (1,1)
            # 子弹: (k,5) -> (1,k,5) -> (1,1,k,5)
            bullets = torch.as_tensor(obs['nearest_bullets'], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            obs_tensor = {
                'player_x': _scalar(obs['player_x']),
                'player_y': _scalar(obs['player_y']),
                'boss_from_player_x': _scalar(obs['boss_from_player_x']),
                'boss_from_player_y': _scalar(obs['boss_from_player_y']),
                'boss_velx': _scalar(obs['boss_velx']),
                'boss_vely': _scalar(obs['boss_vely']),
                'nearest_bullets': bullets,
            }

            q_values, self._hidden_state = self.net(obs_tensor, self._hidden_state)  # (1, num_actions) and hidden state
            temperature = 2.0
            weights = torch.softmax(q_values / temperature, dim=-1, dtype=torch.float32)
            try:
                action_idx: int = torch.multinomial(weights, num_samples=1).item()
                # action_idx = q_values.argmax(dim=1).item()
            except AttributeError as e:
                printred(f'An error occured while determining action: {e}')
                raise
            return list(Action)[action_idx]