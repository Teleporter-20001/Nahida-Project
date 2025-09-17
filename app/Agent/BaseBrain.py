from app.Agent.DataStructure import State, Action


class BaseBrain:

    def __init__(self):
        pass


    def decide_action(self, state: State) -> Action:
        raise NotImplementedError('Do not use base class')
