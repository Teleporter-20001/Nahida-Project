from enum import Enum

class State:

    def __init__(self, observation: dict | None = None):
        """Simple container for observations returned by the environment.

        observation: a dict containing observation fields (positions, etc.).
        """
        self.observation = observation or {}


class Action(Enum):
    LEFTUP = (-0.707, -0.707)
    UP = (0, -1)
    RIGHTUP = (0.707, -0.707)
    LEFT = (-1, 0)
    NOMOVE = (0, 0)
    RIGHT = (1, 0)
    LEFTDOWN = (-0.707, 0.707)
    DOWN = (0, 1)
    RIGHTDOWN = (0.707, 0.707)

    @property
    def xfactor(self):
        return self.value[0]
    @property
    def yfactor(self):
        return self.value[1]