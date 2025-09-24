import os

import torch

from app.Agent.Brains.BaseNetBrain import BaseNetBrain
from app.Agent.Networks.RecurrentQNet import RecurrentQNet
from app.common.Settings import Settings
from app.common.utils import printgreen


class MemoryBrainV2(BaseNetBrain):

    def __init__(self, be_teached=False):
        super().__init__(be_teached)

    def _create_network(self):
        self.net = RecurrentQNet(
            256,
            512,
            256
        ).to(self.device)
        self.model_path = os.path.join('Agent', 'models', f'{Settings.net_name}_{Settings.begin_episode}.pth')
        if os.path.exists(self.model_path):
            try:
                self.net.load_state_dict(torch.load(self.model_path))
                printgreen(f'Successfully loaded model {self.model_path}')
            except Exception as e:
                printgreen(f'Failed to load model {self.model_path} due to error: {e}')