from random import choice
from app.Agent.Brains.BaseBrain import BaseBrain
from app.Agent.DataStructure import State, Action


class RandomBrain(BaseBrain):

    def decide_action(self, state: State) -> Action:
        return choice(list(Action))