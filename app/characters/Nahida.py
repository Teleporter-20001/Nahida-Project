import pygame

from app.Agent.BaseBrain import BaseBrain
from app.Agent.DataStructure import Action, State
from app.Agent.RandomBrain import RandomBrain
from app.characters.BaseChar import BaseChar


class Nahida(BaseChar):

    ORIGIN_SPEED = 100

    def __init__(
            self,
            x: float,
            y: float,
            radius: int,
            img_path: str,
            vx: float=0,
            vy: float=0,
            target_size: tuple[int, int]=(-1, -1),
    ):
        super().__init__(x, y, radius, img_path, vx, vy, target_size)
        self.now_action: Action | None = None
        # self.brain: BaseBrain = RandomBrain()
        

    # def choose_action(self, state: State) -> Action: # TODO：从状态到动作的决策过程应该出现在智能体中，但这是环境中的一个部分。要把它移植过去
    #     return self.brain.decide_action(state)

    def set_action(self, action: Action):
        self.now_action = action

    def update(self):
        if self.now_action:
            xfac, yfac = self.now_action.xfactor, self.now_action.yfactor
            self.vx, self.vy = xfac * Nahida.ORIGIN_SPEED, yfac * Nahida.ORIGIN_SPEED

        super().update()

