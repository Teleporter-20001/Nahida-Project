from app.Agent.BaseBrain import BaseBrain
from app.Agent.RandomBrain import RandomBrain
from app.env.Game import Game

if __name__ == '__main__':
    game: Game = Game()
    agent: BaseBrain = RandomBrain()
    once_reward = game.run_episode(agent, max_steps=100000, render=True)
    print(f'reward in one game: {once_reward}')