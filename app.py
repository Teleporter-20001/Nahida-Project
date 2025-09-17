from app.Agent.BaseBrain import BaseBrain
from app.Agent.RandomBrain import RandomBrain
from app.env.Game import Game

if __name__ == '__main__':
    game: Game = Game()
    agent: BaseBrain = RandomBrain()
    game.run_episode(agent)